# %% [markdown]
# # 04 · 5-model ensemble + post-processing search (CPU only, no GPU quota)
#
# **Setup**
# 1. Competition page -> Code -> New Notebook. Accelerator = **None**.
# 2. Attach BOTH earlier runs: right-hand panel -> **Add Input** -> **Your Work** -> the 02 notebook;
#    then Add Input again -> the 03 notebook. The Input panel should list three items: the competition
#    data and the two notebooks. (03's log said "computed folds", which means 02 wasn't attached then.)
# 3. **Save Version -> Save & Run All**. About 20-40 minutes on Kaggle's 4 CPU cores.
#
# **What it does**
# 1. Collects every out-of-fold map: fold 0 from 02, folds 1-4 from 03, i.e. all 707 training images,
#    each scored by the model that never saw it. Checks that the folds don't overlap.
# 2. Stage A: threshold x minimum area x limb cut x minimum confidence on a wider grid (03's best
#    confidence, 0.8, sat at the top of its grid).
# 3. Stage B, around stage A's best: joining fragments separated by small gaps (the model splits a
#    labelled filament ~4x more often than it merges two), with or without filling holes. "group" keeps
#    the predicted pixels and just counts nearby pieces as one filament; "close" also paints the gap,
#    like annotators do.
# 4. Averages the 5 fold models' test maps, applies the best settings, writes submission.csv
#    (and submission_stageA.csv if stage B changed the answer, for a leaderboard comparison).

# %% 0 · Config ------------------------------------------------------------------------------
import os, re, sys, json, math, time, collections, itertools, subprocess, warnings
from pathlib import Path
import multiprocessing as mp

CFG = dict(
    stage_a=dict(thr=[0.3, 0.35, 0.4, 0.45, 0.5], min_area=[200, 300, 400, 500, 600, 800],
                 r_max=[0.88, 0.90, 0.92, 0.95, 1.01], min_score=[0.7, 0.75, 0.8, 0.85, 0.9]),
    stage_b=dict(merge=[["none", 0], ["group", 4], ["group", 8], ["group", 12], ["group", 20],
                        ["close", 4], ["close", 8], ["close", 12], ["close", 20]],
                 fill=[False, True], thr_delta=[-0.05, 0.0, 0.05], area_mult=[0.75, 1.0, 1.25],
                 score_delta=[-0.05, 0.0, 0.05]),
    workers=max(1, min(4, os.cpu_count() or 1)),
)
CFG.update(json.loads(os.environ.get("BASE_CFG", "{}")))      # overrides, used for local tests
INPUT_ROOT = Path(os.environ.get("BASE_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("BASE_OUT", "/kaggle/working/ensemble"))
OUT.mkdir(parents=True, exist_ok=True)
SUB_PATH = Path(os.environ.get("BASE_SUB", "/kaggle/working/submission.csv"))
H, W = 2048, 2048

import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from pycocotools import mask as mask_utils
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], check=False)
    from pycocotools import mask as mask_utils
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **_):
        return it
cv2.setNumThreads(1)
print(json.dumps(CFG, indent=1))

# %% 1 · Helpers (same as 02/03) ----------------------------------------------------------------
OBS_RE = re.compile(r"(\d{14})([A-Za-z])h")
IMG_EXTS = [".fits.fz", ".fits", ".fts", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]


def ext_of(p):
    return ".fits.fz" if p.name.lower().endswith(".fits.fz") else p.suffix.lower()


def parse_obs(s):
    m = OBS_RE.search(str(s))
    return (m.group(0) if m else Path(str(s)).name.split(".")[0])


def split_of(path):
    text = " ".join(t.lower() for t in Path(path).relative_to(INPUT_ROOT).parts)
    for name in ("test", "val", "train"):
        if re.search(rf"(?<![a-z]){name}", text):
            return name
    return "unknown"


def load_gray(path):
    path = str(path)
    if path.lower().endswith((".fits", ".fts", ".fz")):
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = next(h.data for h in hdul if h.data is not None and np.ndim(h.data) == 2)
        d = np.nan_to_num(np.asarray(data, np.float32))
        lo, hi = np.percentile(d, [0.5, 99.9])
        return np.clip((d - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        from PIL import Image
        img = np.asarray(Image.open(path).convert("L"))
    return img


def decode_crop(a, h=H, w=W, pad=2):
    seg = a.get("segmentation")
    if not seg:
        return None
    if isinstance(seg, list):
        polys = [np.asarray(s, float).reshape(-1, 2) for s in seg if len(s) >= 6]
        if not polys:
            return None
        allp = np.concatenate(polys)
        x0 = int(max(0, math.floor(allp[:, 0].min()) - pad)); y0 = int(max(0, math.floor(allp[:, 1].min()) - pad))
        x1 = int(min(w, math.ceil(allp[:, 0].max()) + pad + 1)); y1 = int(min(h, math.ceil(allp[:, 1].max()) + pad + 1))
        shifted = [(q - [x0, y0]).ravel().tolist() for q in polys]
        m = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(shifted, y1 - y0, x1 - x0))).astype(bool)
    else:
        rle = dict(seg)
        rle.setdefault("size", [h, w])
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, *rle["size"])
        elif isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode()
        full = mask_utils.decode(rle).astype(bool)
        rows, cols = np.flatnonzero(full.any(1)), np.flatnonzero(full.any(0))
        if rows.size == 0:
            return None
        y0, x0 = max(0, rows[0] - pad), max(0, cols[0] - pad)
        m = full[y0: rows[-1] + pad + 1, x0: cols[-1] + pad + 1]
    return (m, y0, x0) if m.any() else None


def rle_encode(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def _inter(a, b):
    (ma, ya, xa), (mb, yb, xb) = a, b
    y0, x0 = max(ya, yb), max(xa, xb)
    y1, x1 = min(ya + ma.shape[0], yb + mb.shape[0]), min(xa + ma.shape[1], xb + mb.shape[1])
    if y0 >= y1 or x0 >= x1:
        return 0
    return int(np.logical_and(ma[y0 - ya: y1 - ya, x0 - xa: x1 - xa], mb[y0 - yb: y1 - yb, x0 - xb: x1 - xb]).sum())


def pq_tables(gt, pred):
    ag = np.array([m.sum() for m, _, _ in gt], float)
    ap = np.array([m.sum() for m, _, _ in pred], float)
    inter = np.zeros((len(gt), len(pred)))
    if len(gt) and len(pred):
        bg = np.array([(y, x, y + m.shape[0], x + m.shape[1]) for m, y, x in gt])
        bp = np.array([(y, x, y + m.shape[0], x + m.shape[1]) for m, y, x in pred])
        ov = ((bg[:, None, 0] < bp[None, :, 2]) & (bp[None, :, 0] < bg[:, None, 2]) &
              (bg[:, None, 1] < bp[None, :, 3]) & (bp[None, :, 1] < bg[:, None, 3]))
        for i, j in zip(*np.nonzero(ov)):
            inter[i, j] = _inter(gt[i], pred[j])
    return inter, ag, ap


def score_tables(inter, ag, ap, keep=None):
    """[tp, fp, fn, sum IoU of matches, splits, merges]; matches need IoU > 0.5, one-to-one."""
    if keep is not None:
        inter, ap = inter[:, keep], ap[keep]
    union = ag[:, None] + ap[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    tp_iou, used_g, used_p = [], set(), set()
    for gi, pj in sorted(zip(*np.nonzero(iou > 0.5)), key=lambda t: -iou[t]):
        if gi not in used_g and pj not in used_p:
            used_g.add(gi); used_p.add(pj); tp_iou.append(iou[gi, pj])
    tp = len(tp_iou)
    in_gt = np.divide(inter, ap[None, :], out=np.zeros_like(inter), where=ap[None, :] > 0) >= 0.5
    in_pred = np.divide(inter, ag[:, None], out=np.zeros_like(inter), where=ag[:, None] > 0) >= 0.5
    return np.array([tp, len(ap) - tp, len(ag) - tp, float(np.sum(tp_iou)),
                     int((in_gt.sum(1) >= 2).sum()), int((in_pred.sum(0) >= 2).sum())])


def pq_parts(tp, fp, fn, sum_iou, *_):
    den = tp + 0.5 * fp + 0.5 * fn
    return (sum_iou / den if den else np.nan, sum_iou / tp if tp else np.nan, tp / den if den else np.nan)


def disk_geometry(img, f=2, r_range=(820, 1060), n_rays=180):
    """Solar disk centre (y, x) and radius, robust to the off-disk halo (ray edges + circle fit)."""
    small = cv2.resize(img, (img.shape[1] // f, img.shape[0] // f), interpolation=cv2.INTER_AREA).astype(np.float32)
    s8 = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, cents = cv2.connectedComponentsWithStats((th > 0).astype(np.uint8))
    if n <= 1:
        return img.shape[0] / 2, img.shape[1] / 2, 909.0
    cx, cy = cents[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    radii = np.arange(r_range[0] / f, r_range[1] / f, 0.5)
    th_ = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    for _ in range(2):
        ys = cy + np.sin(th_)[:, None] * radii[None, :]
        xs = cx + np.cos(th_)[:, None] * radii[None, :]
        prof = ndi.map_coordinates(small, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(len(th_), -1)
        k = np.argmin(np.diff(prof, axis=1), axis=1)
        ey, ex = ys[np.arange(len(th_)), k], xs[np.arange(len(th_)), k]
        keep = np.ones(len(th_), bool)
        for _ in range(3):
            A = np.column_stack([ex[keep], ey[keep], np.ones(keep.sum())])
            (a1, a2, a3), *_ = np.linalg.lstsq(A, ex[keep] ** 2 + ey[keep] ** 2, rcond=None)
            cx_, cy_ = a1 / 2, a2 / 2
            r_ = math.sqrt(max(a3 + cx_ ** 2 + cy_ ** 2, 1e-6))
            res = np.abs(np.hypot(ex - cx_, ey - cy_) - r_)
            keep = res < max(3.0, 2.5 * np.median(res[keep]))
        cy, cx = cy_, cx_
    return cy * f + (f - 1) / 2, cx * f + (f - 1) / 2, r_ * f


def to_full(p_u8):
    return cv2.resize(p_u8, (W, H), interpolation=cv2.INTER_LINEAR)


# %% 2 · Post-processing with optional fragment joining and hole filling ---------------------------
def components_pp(prob_u8, thr, merge="none", d=0, fill=False, min_pix=10):
    """Instances from a full-resolution probability map.
    merge="group": pieces whose gap is <= d px count as one filament (pixels unchanged, may be multi-part).
    merge="close": morphological closing with a d-px disk first, so the gap itself is painted in.
    fill: fill holes inside each instance. Specks under `min_pix` px are ignored.
    Returns crops [(mask, y0, x0)], feats (n, 2) = (area px, mean probability), centroids (n, 2) = (x, y)."""
    binary = (prob_u8 >= int(round(thr * 255))).astype(np.uint8)
    if merge != "none" and d > 0:
        r = int(math.ceil(d / 2))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        if merge == "close":
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    if merge == "group" and d > 0:
        # label the dilated mask; its bounding boxes contain each group's original pixels
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(binary, k), connectivity=8)
        areas = np.bincount(lab[binary > 0], minlength=n)        # original pixels per group
        lab = lab * binary
    else:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
    ids = [j for j in range(1, n) if areas[j] >= min_pix]
    if not ids:
        return [], np.zeros((0, 2)), np.zeros((0, 2))
    comps, feats, cents = [], [], []
    for j in ids:                                   # everything per crop: no extra full-image passes
        x, y, w, h = stats[j, :4]
        sl = (slice(y, y + h), slice(x, x + w))
        m = lab[sl] == j
        score = prob_u8[sl][m].mean() / 255         # confidence = mean probability of the predicted pixels
        if fill:
            m = ndi.binary_fill_holes(m)
        ys, xs = np.nonzero(m)
        comps.append((m, sl[0].start, sl[1].start))
        feats.append((ys.size, score))
        cents.append((sl[1].start + xs.mean(), sl[0].start + ys.mean()))
    return comps, np.array(feats, float), np.array(cents, float)


def keep_mask(feats, cents, geom, p):
    if not len(feats):
        return np.zeros(0, bool)
    cy, cx, R = geom
    rr = np.hypot(cents[:, 1] - cy, cents[:, 0] - cx) / R
    return (feats[:, 0] >= p["min_area"]) & (rr <= p["r_max"]) & (feats[:, 1] >= p["min_score"])


# %% 3 · Data: ground truth, disk fits, out-of-fold and test maps --------------------------------
all_files = sorted(p for p in INPUT_ROOT.rglob("*") if p.is_file())
img_files = [p for p in all_files if ext_of(p) in IMG_EXTS and OBS_RE.search(p.name)
             and not re.search(r"mask|label|annot", str(p.parent.relative_to(INPUT_ROOT)).lower())]
files = pd.DataFrame([dict(obs=parse_obs(p.name), path=str(p), split=split_of(p), rank=IMG_EXTS.index(ext_of(p)))
                      for p in img_files], columns=["obs", "path", "split", "rank"])
best_path = files.sort_values("rank").drop_duplicates("obs").set_index("obs")["path"].to_dict()
SETS = collections.defaultdict(list)
for p in [f for f in all_files if ext_of(f) == ".json"]:
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if not (isinstance(d, dict) and d.get("annotations") and split_of(p) != "test"):
        continue
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    for im in d["images"]:
        SETS[parse_obs(im.get("file_name") or im["id"])].append(by_img.get(im["id"], []))
train_obs = sorted(o for o in SETS if o in best_path)
test_obs = sorted(set(files.loc[files.split == "test", "obs"]))
print(f"train images: {len(train_obs):,} | test images: {len(test_obs):,}")

OOF, OOF_FOLD, TEST = {}, {}, collections.defaultdict(dict)
for p in all_files:
    m = re.fullmatch(r"val_probs_fold(\d+)\.npz", p.name)
    if not m:
        continue
    k = int(m.group(1))
    if k in set(OOF_FOLD.values()):
        print(f"fold {k} already loaded; ignoring the copy in {p.parent.relative_to(INPUT_ROOT)}")
        continue
    d = np.load(p)
    obs = [str(o) for o in d["obs"]]
    clash = [o for o in obs if o in OOF]
    if clash:
        print(f"SKIPPING {p.relative_to(INPUT_ROOT)}: {len(clash)} of its images are already in another fold, "
              f"so the folds don't match")
        continue
    for o, pr in zip(obs, d["probs"]):
        OOF[o], OOF_FOLD[o] = pr, k
    tp = p.with_name(f"test_probs_fold{k}.npz")
    if not tp.exists() and (p.parent / "test_probs.npz").exists():
        tp = p.parent / "test_probs.npz"                          # 02's name for fold 0's test maps
    if tp.exists():
        dt = np.load(tp)
        TEST[k] = {str(o): pr for o, pr in zip(dt["obs"], dt["probs"])}
    print(f"fold {k}: {len(obs)} validation maps from {p.parent.relative_to(INPUT_ROOT)}"
          f"{' + test maps' if k in TEST else ' (no test maps found)'}")
FOLDS = sorted(set(OOF_FOLD.values()))
eval_obs = [o for o in train_obs if o in OOF]
print(f"out-of-fold maps for {len(eval_obs):,} of {len(train_obs):,} train images (folds {FOLDS}); "
      f"test maps from folds {sorted(TEST)}")
if len(FOLDS) < 5:
    print("NOTE: fewer than 5 folds found. Check that both the 02 and 03 notebooks are attached as inputs.")
assert eval_obs, "No out-of-fold maps found: attach the 02 and/or 03 notebook outputs as inputs."

GT, GEOM = {}, {}
for obs in tqdm(sorted(set(eval_obs) | set(test_obs)), desc="ground truth + disk fits"):
    img = load_gray(best_path[obs])
    GEOM[obs] = disk_geometry(img)
    if obs in SETS:
        GT[obs] = [[r for r in (decode_crop(a, *img.shape) for a in anns) if r] for anns in SETS[obs]]

# %% 4 · Parallel evaluation machinery -----------------------------------------------------------
_CONFIGS, _COMBOS = [], []          # set before each pool starts; forked workers inherit them


def eval_image(i):
    """PQ counts for one image, every (config, filter combo): vs each annotator and vs pooled annotators."""
    obs = eval_obs[i]
    prob = to_full(OOF[obs])
    n_combo = max(len(c) for c in _COMBOS)
    ann = np.zeros((len(_CONFIGS), n_combo, 6)); pool = np.zeros((len(_CONFIGS), n_combo, 6))
    for ci, (merge, d, fill, thr) in enumerate(_CONFIGS):
        comps, feats, cents = components_pp(prob, thr, merge, d, fill)
        tables = [pq_tables(g, comps) for g in GT[obs]]
        pooled = (np.vstack([t[0] for t in tables]), np.concatenate([t[1] for t in tables]), tables[0][2])
        for cj, (a, r, s) in enumerate(_COMBOS[ci]):
            keep = keep_mask(feats, cents, GEOM[obs], dict(min_area=a, r_max=r, min_score=s))
            for t in tables:
                ann[ci, cj] += score_tables(*t, keep)
            pool[ci, cj] += score_tables(*pooled, keep)
    return i, ann, pool


def run_stage(configs, combos, desc):
    """Returns per-image count arrays: (n_images, n_configs, n_combos, 6) for annotator and pooled scoring."""
    global _CONFIGS, _COMBOS
    _CONFIGS, _COMBOS = configs, combos
    n_combo = max(len(c) for c in combos)
    ANN = np.zeros((len(eval_obs), len(configs), n_combo, 6)); POOL = np.zeros_like(ANN)
    t0 = time.time()
    try:
        if CFG["workers"] <= 1:
            raise RuntimeError("single process requested")
        with mp.get_context("fork").Pool(CFG["workers"]) as pool:
            for i, a, p in tqdm(pool.imap_unordered(eval_image, range(len(eval_obs)), chunksize=2),
                                total=len(eval_obs), desc=desc):
                ANN[i], POOL[i] = a, p
    except Exception as e:
        print(f"  ({e}; running in one process)")
        for i in tqdm(range(len(eval_obs)), desc=desc):
            _, ANN[i], POOL[i] = eval_image(i)
    print(f"  {desc}: {len(configs)} configs x {n_combo} filters on {len(eval_obs)} images in {time.time() - t0:.0f}s")
    return ANN, POOL


def table(ANN, POOL, configs, combos):
    tot_a, tot_p = ANN.sum(0), POOL.sum(0)
    rows = []
    for ci, (merge, d, fill, thr) in enumerate(configs):
        for cj, (a, r, s) in enumerate(combos[ci]):
            pq, sq, rq = pq_parts(*tot_a[ci, cj])
            tp, fp, fn, _, spl, mer = tot_a[ci, cj]
            rows.append(dict(ci=ci, cj=cj, merge=merge, d=d, fill=fill, thr=thr, min_area=a, r_max=r, min_score=s,
                             PQ=pq, SQ=sq, RQ=rq, PQ_pooled_gt=pq_parts(*tot_p[ci, cj])[0],
                             tp=tp, fp=fp, fn=fn, splits=spl, merges=mer))
    return pd.DataFrame(rows).sort_values("PQ", ascending=False).reset_index(drop=True)


# %% 5 · Reference points: default post-processing and human agreement ----------------------------
ref_cfg = [("none", 0, False, 0.5)]
ANN0, POOL0 = run_stage(ref_cfg, [[(0, 1.01, 0.0)]], "default post-processing")
DEFAULT_PQ = pq_parts(*ANN0.sum(0)[0, 0])
hum = np.zeros(6)
for o in eval_obs:
    for i, j in itertools.combinations(range(len(GT[o])), 2):
        hum += score_tables(*pq_tables(GT[o][i], GT[o][j]))
HUMAN = pq_parts(*hum)
print(f"default post-processing: PQ {DEFAULT_PQ[0]:.3f} | human vs human on these images: PQ {HUMAN[0]:.3f}")

# %% 6 · Stage A: threshold x min area x limb cut x min confidence ----------------------------------
ga = CFG["stage_a"]
cfg_a = [("none", 0, False, t) for t in ga["thr"]]
combos_a = [list(itertools.product(ga["min_area"], ga["r_max"], ga["min_score"]))] * len(cfg_a)
ANN_A, POOL_A = run_stage(cfg_a, combos_a, "stage A")
tab_a = table(ANN_A, POOL_A, cfg_a, combos_a)
tab_a.drop(columns=["ci", "cj"]).to_csv(OUT / "oof_grid_stageA.csv", index=False)
A = tab_a.iloc[0]
print(tab_a.drop(columns=["ci", "cj", "merge", "d", "fill"]).head(8).round(4).to_string(index=False))

# %% 7 · Stage B: fragment joining and hole filling around stage A's best ----------------------------
gb = CFG["stage_b"]
thr_b = sorted({round(min(0.95, max(0.05, A.thr + dt)), 3) for dt in gb["thr_delta"]})
areas_b = sorted({int(round(A.min_area * m)) for m in gb["area_mult"]})
scores_b = sorted({round(min(0.99, max(0.0, A.min_score + ds)), 3) for ds in gb["score_delta"]})
cfg_b = [(m, int(d), bool(f), t) for (m, d) in gb["merge"] for f in gb["fill"] for t in thr_b]
combos_b = [[(a, float(A.r_max), s) for a in areas_b for s in scores_b]] * len(cfg_b)
ANN_B, POOL_B = run_stage(cfg_b, combos_b, "stage B")
tab_b = table(ANN_B, POOL_B, cfg_b, combos_b)
tab_b.drop(columns=["ci", "cj"]).to_csv(OUT / "oof_grid_stageB.csv", index=False)
B = tab_b.iloc[0]
print(tab_b.drop(columns=["ci", "cj"]).head(10).round(4).to_string(index=False))
best_by_merge = tab_b.sort_values("PQ", ascending=False).drop_duplicates(["merge", "d", "fill"])
print("\nbest PQ for each joining / hole-filling option:")
print(best_by_merge[["merge", "d", "fill", "thr", "min_area", "min_score", "PQ", "SQ", "RQ", "splits", "merges"]]
      .round(4).to_string(index=False))

# %% 8 · Choose, report per fold, write the submission(s) ----------------------------------------
if B.PQ > A.PQ:
    best, ANN_best, POOL_best, ci, cj = B, ANN_B, POOL_B, int(B.ci), int(B.cj)
else:
    best, ANN_best, POOL_best, ci, cj = A, ANN_A, POOL_A, int(A.ci), int(A.cj)
oof_per_img = POOL_best[:, ci, cj, :2].sum() / len(eval_obs)      # tp + fp vs pooled GT = predictions kept
P = dict(merge=str(best.merge), d=int(best.d), fill=bool(best.fill), thr=float(best.thr),
         min_area=int(best.min_area), r_max=float(best.r_max), min_score=float(best.min_score))
PA = dict(merge="none", d=0, fill=False, thr=float(A.thr), min_area=int(A.min_area), r_max=float(A.r_max),
          min_score=float(A.min_score))
per_fold = {}
for k in FOLDS:
    idx = [i for i, o in enumerate(eval_obs) if OOF_FOLD[o] == k]
    per_fold[k] = round(float(pq_parts(*ANN_best[idx, ci, cj].sum(0))[0]), 4)


def write_submission(params, path):
    rows, n_empty, used = [], 0, sorted(TEST)
    for obs in test_obs:
        maps = [TEST[k][obs] for k in used if obs in TEST[k]]
        if not maps:
            n_empty += 1
            continue
        size = max(m.shape[0] for m in maps)
        maps = [m if m.shape[0] == size else cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR) for m in maps]
        prob = to_full(np.round(np.mean(maps, axis=0)).astype(np.uint8))
        comps, feats, cents = components_pp(prob, params["thr"], params["merge"], params["d"], params["fill"])
        keep = keep_mask(feats, cents, GEOM[obs], params)
        k = 0
        for (m, y0, x0), kp in zip(comps, keep):
            if kp:
                full = np.zeros((H, W), np.uint8)
                full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
                k += 1
                rows.append((f"{obs}_{k}", rle_encode(full)))
        n_empty += k == 0
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    sub.to_csv(path, index=False)
    return sub, n_empty, used


sub, n_empty, used = write_submission(P, SUB_PATH)
if P != PA:
    sub_a, _, _ = write_submission(PA, SUB_PATH.with_name("submission_stageA.csv"))
json.dump(dict(params=P, oof_PQ=float(best.PQ), oof_SQ=float(best.SQ), oof_RQ=float(best.RQ),
               oof_PQ_pooled_gt=float(best.PQ_pooled_gt), stageA_params=PA, stageA_PQ=float(A.PQ),
               default_PQ=float(DEFAULT_PQ[0]), human_PQ=float(HUMAN[0]), per_fold_PQ=per_fold,
               folds=FOLDS, n_images=len(eval_obs), test_folds=used), open(OUT / "params.json", "w"), indent=1)

print("=" * 30, "RESULTS", "=" * 30)
print(f"out-of-fold maps: {len(eval_obs)} images from folds {FOLDS}; test ensemble of {len(used)} fold models {used}")
print(f"stage A best (no joining)      : PQ {A.PQ:.3f} (SQ {A.SQ:.3f}, RQ {A.RQ:.3f})  "
      f"thr {A.thr}, min_area {int(A.min_area)}, r_max {A.r_max}, min_score {A.min_score}")
print(f"stage B best                   : PQ {B.PQ:.3f} (SQ {B.SQ:.3f}, RQ {B.RQ:.3f})  "
      f"merge {B.merge} d={int(B.d)}, fill {bool(B.fill)}, thr {B.thr}, min_area {int(B.min_area)}, min_score {B.min_score}")
print(f"chosen, vs pooled annotators   : PQ {best.PQ_pooled_gt:.3f}")
print(f"default post-processing        : PQ {DEFAULT_PQ[0]:.3f}")
print(f"human vs human, same images    : PQ {HUMAN[0]:.3f}")
print("per fold (chosen settings)     : " + ", ".join(f"fold {k} {v:.3f}" for k, v in per_fold.items()))
for name, val, grid_vals in [("min_score", A.min_score, ga["min_score"]), ("min_area", A.min_area, ga["min_area"])]:
    if val == max(grid_vals):
        print(f"note: stage A's best {name} is the largest value tried ({val}); widen CFG['stage_a'] next time")
print(f"submission.csv: {len(sub):,} filaments over {len(test_obs) - n_empty} of {len(test_obs)} test images"
      + ("; submission_stageA.csv holds the stage A version" if P != PA else ""))
test_per_img = len(sub) / max(len(test_obs), 1)
print(f"filaments per image            : out-of-fold {oof_per_img:.1f} vs test ensemble {test_per_img:.1f}")
if not 0.8 <= test_per_img / max(oof_per_img, 1e-9) <= 1.25:
    print("note: the averaged test maps give a noticeably different count than single-model maps; "
          "settings tuned on single models may need a lower min_score for the ensemble")
