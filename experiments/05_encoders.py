# %% [markdown]
# # 05 · Which encoder? Three models on fold 0, scored exactly like 04
#
# **Setup:** GPU T4 x2, Internet On. Attach your `base-model` dataset (the 02 output) so its fold-0
# model can be re-scored as the reference; nothing else is needed. **Save Version -> Save & Run All.**
# Roughly 4-6 hours (3 models x 30 epochs; the larger encoders are slower per epoch).
#
# **Why crops:** ResNet34 at 1024 px already uses most of a T4's memory, so bigger encoders train on
# random 768x768 crops of the 1024x1024 images (same scale, ~45% less memory) and still predict
# whole 1024x1024 images. The ResNet34-on-crops run separates the effect of cropping from the effect
# of the encoder. If a model doesn't fit in memory, its crop shrinks by 128 px until it does.
#
# **How models are compared** (all on fold 0's 142 validation images):
# - `PQ_fixed`: 04's best post-processing, unchanged, so no tuning luck is involved.
# - `PQ_tuned`: the best of a small grid around it (joining gap, threshold, min area, confidence).
# - inference seconds per image with flip TTA, since efficiency is part of the judging.
# Each model's validation and test maps are saved, so a winner can join the ensemble directly.

# %% 0 · Config ------------------------------------------------------------------------------
import os, re, sys, json, math, time, random, subprocess, collections, warnings, gc, itertools
from pathlib import Path

CFG = dict(
    fold=0, n_folds=5, seed=42, img_size=1024,
    experiments=[
        dict(name="resnet34_crop768", encoder="resnet34", crop=768),
        dict(name="resnet50_crop768", encoder="resnet50", crop=768),
        dict(name="effb3_crop768", encoder="efficientnet-b3", crop=768),
    ],
    encoder_weights="imagenet", epochs=30, batch_size=8, lr=3e-4, weight_decay=1e-4, warmup_epochs=1,
    num_workers=4, amp=True, tta=True, channels_last=True, eval_every=10,
    pp_fixed=dict(merge="group", d=20, fill=False, thr=0.5, min_area=300, r_max=0.9, min_score=0.85),  # 04's best
    pp_grid=dict(merge=[["group", 12], ["group", 20], ["group", 30]], thr=[0.4, 0.45, 0.5, 0.55],
                 min_area=[200, 300, 400], r_max=[0.9], min_score=[0.75, 0.8, 0.85, 0.88]),
)
CFG.update(json.loads(os.environ.get("BASE_CFG", "{}")))      # overrides, used for local tests
INPUT_ROOT = Path(os.environ.get("BASE_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("BASE_OUT", "/kaggle/working/encoders"))
OUT.mkdir(parents=True, exist_ok=True)
H, W = 2048, 2048
S = CFG["img_size"]
print(json.dumps(CFG, indent=1))

# %% 1 · Imports -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _need(module, pip_name):
    try:
        return __import__(module)
    except ImportError:
        print(f"installing {pip_name} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=False)
        return __import__(module)


smp = _need("segmentation_models_pytorch", "segmentation-models-pytorch")
_need("pycocotools", "pycocotools")
from pycocotools import mask as mask_utils
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **_):
        return it

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()
USE_AMP = CFG["amp"] and DEVICE.type == "cuda"
MEMFMT = torch.channels_last if CFG["channels_last"] else torch.contiguous_format
print(f"device: {DEVICE}, GPUs: {N_GPU}, torch {torch.__version__}, smp {smp.__version__}")
if DEVICE.type == "cpu":
    print("WARNING: no GPU. Set Accelerator = GPU T4 x2 in the notebook settings.")
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

# %% 2 · Helpers (identical to 03 and 04) -----------------------------------------------------
OBS_RE = re.compile(r"(\d{14})([A-Za-z])h")
IMG_EXTS = [".fits.fz", ".fits", ".fts", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]


def ext_of(p):
    return ".fits.fz" if p.name.lower().endswith(".fits.fz") else p.suffix.lower()


def parse_obs(s):
    m = OBS_RE.search(str(s))
    if not m:
        return Path(str(s)).name.split(".")[0], pd.NaT, None
    return m.group(0), pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S", errors="coerce"), m.group(2).upper()


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
    """One annotation -> (bool crop, y0, x0), rasterised exactly like pycocotools annToMask."""
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


def paint(insts, shape):
    canvas = np.zeros(shape, bool)
    for m, y0, x0 in insts:
        canvas[y0: y0 + m.shape[0], x0: x0 + m.shape[1]] |= m
    return canvas


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
    """Intersection table between GT and predicted instances (bbox-overlapping pairs only) + areas."""
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
    """Solar disk centre (y, x) and radius: steepest brightness drop along rays from a rough centre,
    then a least-squares circle through those edge points (robust to the off-disk halo)."""
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
    """Model-resolution uint8 probability map -> 2048x2048 uint8."""
    return cv2.resize(p_u8, (W, H), interpolation=cv2.INTER_LINEAR)


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


def evaluate_maps(probs_u8, obs_list, grid):
    """PQ vs each annotator for every (joining, threshold) x (min area, limb, confidence) combination."""
    cfgs = [(m, int(d), t) for (m, d) in grid["merge"] for t in grid["thr"]]
    combos = list(itertools.product(grid["min_area"], grid["r_max"], grid["min_score"]))
    acc = np.zeros((len(cfgs), len(combos), 6))
    for p_small, obs in zip(probs_u8, obs_list):
        prob = to_full(p_small)
        for ci, (m, d, t) in enumerate(cfgs):
            comps, feats, cents = components_pp(prob, t, m, d, False)
            tables = [pq_tables(g, comps) for g in GT[obs]]
            for cj, (a, r, s) in enumerate(combos):
                keep = keep_mask(feats, cents, GEOM[obs], dict(min_area=a, r_max=r, min_score=s))
                for tb in tables:
                    acc[ci, cj] += score_tables(*tb, keep)
    rows = []
    for ci, (m, d, t) in enumerate(cfgs):
        for cj, (a, r, s) in enumerate(combos):
            pq, sq, rq = pq_parts(*acc[ci, cj])
            rows.append(dict(merge=m, d=d, thr=t, min_area=a, r_max=r, min_score=s, PQ=pq, SQ=sq, RQ=rq,
                             tp=acc[ci, cj][0], fp=acc[ci, cj][1], fn=acc[ci, cj][2],
                             splits=acc[ci, cj][4], merges=acc[ci, cj][5]))
    return pd.DataFrame(rows).sort_values("PQ", ascending=False).reset_index(drop=True)


def single_grid(p):
    return dict(merge=[[p["merge"], p["d"]]], thr=[p["thr"]], min_area=[p["min_area"]], r_max=[p["r_max"]],
                min_score=[p["min_score"]])


# %% 3 · Data, folds, cache, ground truth, disk fits ---------------------------------------------
all_files = sorted(p for p in INPUT_ROOT.rglob("*") if p.is_file())
img_files = [p for p in all_files if ext_of(p) in IMG_EXTS and OBS_RE.search(p.name)
             and not re.search(r"mask|label|annot", str(p.parent.relative_to(INPUT_ROOT)).lower())]
files = pd.DataFrame([dict(obs=parse_obs(p.name)[0], path=str(p), split=split_of(p), rank=IMG_EXTS.index(ext_of(p)))
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
        SETS[parse_obs(im.get("file_name") or im["id"])[0]].append(by_img.get(im["id"], []))
train_obs = sorted(o for o in SETS if o in best_path)
test_obs = sorted(set(files.loc[files.split == "test", "obs"]))
print(f"train images: {len(train_obs):,} | test images: {len(test_obs):,}")

fold_of, prev_folds = {}, [p for p in all_files if p.name == "folds.csv"]
if prev_folds:
    f = pd.read_csv(prev_folds[0])
    fold_of = dict(zip(f.obs.astype(str), f.fold.astype(int)))
    print(f"using folds from {prev_folds[0].relative_to(INPUT_ROOT)}")
if set(train_obs) - set(fold_of):
    from sklearn.model_selection import KFold, StratifiedKFold
    years = np.array([parse_obs(o)[1].year if pd.notna(parse_obs(o)[1]) else 0 for o in train_obs])
    sk = (StratifiedKFold(CFG["n_folds"], shuffle=True, random_state=CFG["seed"]).split(train_obs, years)
          if min(collections.Counter(years).values()) >= CFG["n_folds"]
          else KFold(CFG["n_folds"], shuffle=True, random_state=CFG["seed"]).split(train_obs))
    fold_of = {}
    for k, (_, va) in enumerate(sk):
        for i in va:
            fold_of[train_obs[i]] = k
    print("computed folds (same recipe as 02/03)")
K = CFG["fold"]
trn = [o for o in train_obs if fold_of[o] != K]
val = [o for o in train_obs if fold_of[o] == K]
print(f"fold {K}: {len(trn)} train / {len(val)} validation images")

X_all = np.zeros((len(train_obs), S, S), np.uint8)
Y_all = np.zeros((len(train_obs), S, S), np.uint8)
GT, GEOM = {}, {}
for i, obs in enumerate(tqdm(train_obs, desc="caching train")):
    img = load_gray(best_path[obs])
    X_all[i] = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
    GT[obs] = [[r for r in (decode_crop(a, *img.shape) for a in anns) if r] for anns in SETS[obs]]
    acc = np.zeros(img.shape, np.float32)
    for insts in GT[obs]:
        acc += paint(insts, img.shape)
    Y_all[i] = np.round(cv2.resize(acc / max(len(GT[obs]), 1), (S, S), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    if obs in val:
        GEOM[obs] = disk_geometry(img)
test_small = [cv2.resize(load_gray(best_path[o]), (S, S), interpolation=cv2.INTER_AREA) for o in tqdm(test_obs, desc="caching test")]
idx_of = {o: i for i, o in enumerate(train_obs)}
val_small = [X_all[idx_of[o]] for o in val]

# %% 4 · Dataset (random crops), model, training -------------------------------------------------
MEAN, STD = 0.449, 0.226
_rng = None


def rng():
    global _rng
    if _rng is None:
        _rng = np.random.default_rng(np.random.randint(2 ** 31))
    return _rng


class FilamentDS(Dataset):
    def __init__(self, idx, crop):
        self.idx, self.crop = list(idx), crop

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        x, y = X_all[self.idx[k]], Y_all[self.idx[k]]
        r = rng()
        kk = r.integers(4)
        x, y = np.rot90(x, kk), np.rot90(y, kk)
        if r.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        if r.random() < 0.5:
            M = cv2.getRotationMatrix2D((S / 2, S / 2), r.uniform(-45, 45), r.uniform(0.95, 1.05))
            M[:, 2] += r.uniform(-0.01, 0.01, 2) * S
            x = cv2.warpAffine(x, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
            y = cv2.warpAffine(y, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
        c = self.crop
        if c < S:                                                   # random crop at full model scale
            y0, x0 = r.integers(0, S - c + 1), r.integers(0, S - c + 1)
            x, y = x[y0:y0 + c, x0:x0 + c], y[y0:y0 + c, x0:x0 + c]
        a, b, g = r.uniform(0.85, 1.15), r.uniform(-15, 15), r.uniform(0.8, 1.25)
        x = cv2.LUT(np.ascontiguousarray(x), np.clip(255 * (np.clip(np.arange(256) * a + b, 0, 255) / 255) ** g, 0, 255).astype(np.uint8))
        if r.random() < 0.2:
            x = cv2.GaussianBlur(x, (0, 0), r.uniform(0.5, 1.2))
        if r.random() < 0.2:
            x = np.clip(x + r.standard_normal(x.shape, dtype=np.float32) * r.uniform(1, 4), 0, 255).astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(x)[None]), torch.from_numpy(np.ascontiguousarray(y)[None])


def to_input(x_u8):
    x = (x_u8.to(DEVICE, non_blocking=True).float() / 255 - MEAN) / STD
    return x.contiguous(memory_format=MEMFMT)


def autocast():
    return torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP)


def loss_fn(logits, y):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return nn.functional.binary_cross_entropy_with_logits(logits, y) + dice.mean()


def build_model(encoder):
    m = smp.Unet(encoder, encoder_weights=CFG["encoder_weights"], in_channels=1, classes=1)
    m = m.to(DEVICE).to(memory_format=MEMFMT)
    return nn.DataParallel(m) if N_GPU > 1 else m


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def fits_in_memory(model, crop):
    """One forward/backward pass on a full batch of crops; False if CUDA runs out of memory."""
    if DEVICE.type != "cuda":
        return True
    model.eval()                                                    # don't touch BatchNorm running stats
    try:
        x = torch.zeros(CFG["batch_size"], 1, crop, crop, device=DEVICE).contiguous(memory_format=MEMFMT)
        with autocast():
            out = model(x)
        loss_fn(out.float(), torch.zeros_like(out.float())).backward()
        ok = True
    except torch.cuda.OutOfMemoryError:
        ok = False
    model.zero_grad(set_to_none=True)
    model.train()
    gc.collect(); torch.cuda.empty_cache()
    return ok


@torch.no_grad()
def predict(model, images_u8, tta=CFG["tta"]):
    model.eval()
    out, bs = [], max(1, CFG["batch_size"] // 2)
    for i in range(0, len(images_u8), bs):
        x = to_input(torch.from_numpy(np.stack(images_u8[i:i + bs]))[:, None])
        views = [((), ())] + ([((-1,), (-1,)), ((-2,), (-2,)), ((-1, -2), (-1, -2))] if tta else [])
        p = 0
        for fwd, back in views:
            xi = x.flip(fwd) if fwd else x
            with autocast():
                pi = torch.sigmoid(model(xi).float())
            p = p + (pi.flip(back) if back else pi)
        p = (p / len(views))[:, 0].cpu().numpy()
        out += list(np.round(p * 255).astype(np.uint8))
    return out


def train_experiment(exp):
    name, enc = exp["name"], exp["encoder"]
    print(f"\n===== {name}: {enc}, crops of {exp['crop']} px, fold {K} =====")
    model = build_model(enc)
    crop = min(exp["crop"], S)
    while crop >= 512 and not fits_in_memory(model, crop):
        print(f"  {crop} px crops don't fit in GPU memory; trying {crop - 128}")
        crop -= 128
    dl = DataLoader(FilamentDS([idx_of[o] for o in trn], crop), batch_size=CFG["batch_size"], shuffle=True,
                    drop_last=len(trn) > CFG["batch_size"], num_workers=CFG["num_workers"],
                    pin_memory=DEVICE.type == "cuda", persistent_workers=CFG["num_workers"] > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    steps, warm = max(1, len(dl)) * CFG["epochs"], max(1, len(dl)) * CFG["warmup_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    log, best, t0 = [], -1.0, time.time()
    fixed = single_grid(CFG["pp_fixed"])
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        tot = n = 0
        te = time.time()
        for x, y in dl:
            x, y = to_input(x), y.to(DEVICE, non_blocking=True).float() / 255
            with autocast():
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        row = dict(epoch=epoch, train_loss=round(tot / max(n, 1), 4), epoch_min=round((time.time() - te) / 60, 2))
        if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
            res = evaluate_maps(predict(model, val_small, tta=False), val, fixed).iloc[0]
            row.update(val_PQ_fixed_noTTA=round(float(res.PQ), 4))
            if res.PQ > best:
                best = res.PQ
                torch.save(unwrap(model).state_dict(), OUT / f"{name}.pt")
                row["saved"] = True
        log.append(row)
        print(row)
    pd.DataFrame(log).to_csv(OUT / f"train_log_{name}.csv", index=False)
    unwrap(model).load_state_dict(torch.load(OUT / f"{name}.pt", map_location=DEVICE))
    val_probs = predict(model, val_small)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    ti = time.time()
    test_probs = predict(model, test_small)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    sec_per_img = (time.time() - ti) / max(len(test_small), 1)
    np.savez_compressed(OUT / f"val_probs_fold{K}_{name}.npz", obs=np.array(val), probs=np.stack(val_probs))
    if test_probs:
        np.savez_compressed(OUT / f"test_probs_fold{K}_{name}.npz", obs=np.array(test_obs), probs=np.stack(test_probs))
    info = dict(name=name, encoder=enc, crop=crop, params_M=round(sum(p.numel() for p in unwrap(model).parameters()) / 1e6, 1),
                epoch_min=float(np.median([r["epoch_min"] for r in log])), train_hours=round((time.time() - t0) / 3600, 2),
                infer_s_per_img=round(sec_per_img, 3))
    del model, opt, sched, scaler, dl
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return val_probs, info


# %% 5 · Reference: your existing fold-0 model (02), if attached ------------------------------------
results = []


def score_and_record(name, probs, info):
    fixed = evaluate_maps(probs, val, single_grid(CFG["pp_fixed"])).iloc[0]
    tuned = evaluate_maps(probs, val, CFG["pp_grid"])
    tuned.to_csv(OUT / f"grid_{name}.csv", index=False)
    t = tuned.iloc[0]
    results.append(dict(**info, PQ_fixed=round(float(fixed.PQ), 4), SQ_fixed=round(float(fixed.SQ), 4),
                        RQ_fixed=round(float(fixed.RQ), 4), splits=int(fixed.splits), merges=int(fixed.merges),
                        PQ_tuned=round(float(t.PQ), 4),
                        tuned_with=f"{t['merge']} d={int(t.d)}, thr {t.thr}, area {int(t.min_area)}, score {t.min_score}"))
    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)
    print(f"  {name}: PQ fixed {fixed.PQ:.4f} | tuned {t.PQ:.4f}")


for p in all_files:
    if p.name == f"val_probs_fold{K}.npz":
        d = np.load(p)
        o = [str(x) for x in d["obs"]]
        if sorted(o) == sorted(val):
            m = dict(zip(o, d["probs"]))
            score_and_record("reference_02_resnet34_full1024", [m[x] for x in val],
                             dict(name="reference_02_resnet34_full1024", encoder="resnet34", crop=S, params_M=24.4,
                                  epoch_min=np.nan, train_hours=np.nan, infer_s_per_img=np.nan))
        else:
            print(f"{p.relative_to(INPUT_ROOT)} holds a different fold-{K} split; not used as reference")
        break
else:
    print("no fold-0 maps from 02 attached; the table will have no reference row")

# %% 6 · Train and score each experiment -------------------------------------------------------
for exp in CFG["experiments"]:
    try:
        vp, info = train_experiment(exp)
    except Exception as e:                                          # one failing encoder shouldn't sink the run
        print(f"!! {exp['name']} failed: {type(e).__name__}: {e}")
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        continue
    score_and_record(exp["name"], vp, info)

# %% 7 · Results ------------------------------------------------------------------------------
print("=" * 30, "RESULTS", "=" * 30)
res = pd.DataFrame(results)
if len(res):
    print(res[["name", "crop", "params_M", "epoch_min", "train_hours", "infer_s_per_img", "PQ_fixed", "SQ_fixed",
               "RQ_fixed", "splits", "merges", "PQ_tuned", "tuned_with"]].to_string(index=False))
    print(f"\nfold {K}: {len(val)} validation images. PQ_fixed uses 04's settings {CFG['pp_fixed']}.")
print(f"saved to {OUT}: weights, val/test maps per model, train logs, results.csv")
