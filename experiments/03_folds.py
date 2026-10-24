# %% [markdown]
# # 03 · Train the remaining folds, tune on all out-of-fold predictions, submit a 5-model ensemble
#
# **Setup:** same as 02 (Accelerator = GPU T4 x2, Internet = On). Before running, attach the 02
# notebook's output: right-hand panel -> **Add Input** -> **Your Work** -> pick the 02 notebook.
# The script then reuses its folds and its fold-0 predictions, so the ensemble covers all 5 folds.
# Run with **Save Version -> Save & Run All**. Roughly 5-6 hours for 4 folds (Kaggle's limit is 12).
#
# **What it does**
# 1. Trains folds 1-4 with the 02 recipe, 30 epochs instead of 40 (fold 0 plateaued by ~30), with
#    lighter data loading (uint8 batches normalised on the GPU, lookup-table intensity augmentation)
#    and channels-last convolutions. Each epoch prints the share of time spent waiting for data.
# 2. Saves each fold's validation and test probability maps, plus its weights.
# 3. Tunes post-processing on all out-of-fold predictions (every training image scored by the model
#    that never saw it), 5x more images than 02 used, on a wider grid: 02's best minimum area and
#    confidence sat at the edge of its grid.
# 4. Averages the fold models' test maps, applies the tuned post-processing, writes submission.csv.

# %% 0 · Config --------------------------------------------------------------------------------
import os, re, sys, json, math, time, random, subprocess, collections, warnings, gc, itertools
from pathlib import Path

CFG = dict(
    folds=[1, 2, 3, 4],               # 02 trained fold 0. Use [0, 1, 2, 3, 4] if you can't attach its output.
    n_folds=5, seed=42,
    img_size=1024,
    encoder="resnet34", encoder_weights="imagenet",
    epochs=30, batch_size=8, lr=3e-4, weight_decay=1e-4, warmup_epochs=1,
    num_workers=4, amp=True, tta=True, channels_last=True,
    eval_every=10,
    grid=dict(thr=[0.3, 0.4, 0.5, 0.6], min_area=[200, 400, 600, 800, 1200],
              r_max=[0.85, 0.90, 0.95, 1.01], min_score=[0.5, 0.6, 0.7, 0.75, 0.8]),
)
CFG.update(json.loads(os.environ.get("BASE_CFG", "{}")))      # overrides, used for local tests
INPUT_ROOT = Path(os.environ.get("BASE_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("BASE_OUT", "/kaggle/working/folds"))
OUT.mkdir(parents=True, exist_ok=True)
SUB_PATH = Path(os.environ.get("BASE_SUB", "/kaggle/working/submission.csv"))
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
    print("WARNING: no GPU. Set Accelerator = GPU T4 x2 in the notebook settings, or this will take days.")
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

# %% 2 · Helpers: parsing, masks, PQ bookkeeping, disk fit (same as 02) --------------------------
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


def components(prob_u8, thr, min_pix=10):
    """Connected components of prob >= thr at full resolution, ignoring specks under `min_pix` px.
    Returns crops [(mask, y0, x0)], feats (n, 2) = (area px, mean probability), centroids (n, 2) = (x, y)."""
    binary = (prob_u8 >= int(round(thr * 255))).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    ids = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_pix) + 1 if n > 1 else np.zeros(0, int)
    if ids.size == 0:
        return [], np.zeros((0, 2)), np.zeros((0, 2))
    score = np.asarray(ndi.mean(prob_u8.astype(np.float32) / 255, lab, ids))
    comps = []
    for j in ids:
        x, y, w, h = stats[j, :4]
        comps.append((lab[y:y + h, x:x + w] == j, y, x))
    feats = np.column_stack([stats[ids, cv2.CC_STAT_AREA], score]).astype(float)
    return comps, feats, cents[ids]


def keep_mask(feats, cents, geom, p):
    if not len(feats):
        return np.zeros(0, bool)
    cy, cx, R = geom
    rr = np.hypot(cents[:, 1] - cy, cents[:, 0] - cx) / R
    return (feats[:, 0] >= p["min_area"]) & (rr <= p["r_max"]) & (feats[:, 1] >= p["min_score"])


def evaluate(probs_u8, obs_list, grid):
    """Score every post-processing combination on these images (PQ vs each annotator, and vs all
    annotators pooled). Probability maps are upsampled to full resolution once per image."""
    thrs = grid["thr"]
    combos = list(itertools.product(grid["min_area"], grid["r_max"], grid["min_score"]))
    acc_ann = np.zeros((len(thrs), len(combos), 6)); acc_pool = np.zeros((len(thrs), len(combos), 6))
    for p_small, obs in zip(probs_u8, obs_list):
        prob = to_full(p_small)
        for ti, thr in enumerate(thrs):
            comps, feats, cents = components(prob, thr)
            tables = [pq_tables(g, comps) for g in GT[obs]]
            pooled = (np.vstack([t[0] for t in tables]), np.concatenate([t[1] for t in tables]), tables[0][2])
            for ci, (a, r, s) in enumerate(combos):
                keep = keep_mask(feats, cents, GEOM[obs], dict(min_area=a, r_max=r, min_score=s))
                for t in tables:
                    acc_ann[ti, ci] += score_tables(*t, keep)
                acc_pool[ti, ci] += score_tables(*pooled, keep)
    rows = []
    for ti, thr in enumerate(thrs):
        for ci, (a, r, s) in enumerate(combos):
            pq, sq, rq = pq_parts(*acc_ann[ti, ci]); pqp = pq_parts(*acc_pool[ti, ci])[0]
            tp, fp, fn, _, spl, mer = acc_ann[ti, ci]
            rows.append(dict(thr=thr, min_area=a, r_max=r, min_score=s, PQ=pq, SQ=sq, RQ=rq, PQ_pooled_gt=pqp,
                             tp=tp, fp=fp, fn=fn, splits=spl, merges=mer))
    return pd.DataFrame(rows).sort_values("PQ", ascending=False).reset_index(drop=True)


def single(p):
    return {k: [v] for k, v in p.items()}


DEFAULT = dict(thr=0.5, min_area=0, r_max=1.01, min_score=0.0)

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
    print(f"annotations from {p.relative_to(INPUT_ROOT)}: {len(d['images']):,} sets, {len(d['annotations']):,} filaments")
train_obs = sorted(o for o in SETS if o in best_path)
test_obs = sorted(set(files.loc[files.split == "test", "obs"]))
print(f"train images: {len(train_obs):,} | test images: {len(test_obs):,}")
assert train_obs, "No training annotations found."

# folds: reuse 02's folds.csv when its output is attached, so fold 0's predictions stay out-of-fold
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
    print("computed folds (same recipe as 02)")
pd.DataFrame({"obs": train_obs, "fold": [fold_of[o] for o in train_obs]}).to_csv(OUT / "folds.csv", index=False)

X_all = np.zeros((len(train_obs), S, S), np.uint8)
Y_all = np.zeros((len(train_obs), S, S), np.uint8)
GT, GEOM = {}, {}
t0 = time.time()
for i, obs in enumerate(tqdm(train_obs, desc="caching train")):
    img = load_gray(best_path[obs])
    X_all[i] = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
    GT[obs] = [[r for r in (decode_crop(a, *img.shape) for a in anns) if r] for anns in SETS[obs]]
    acc = np.zeros(img.shape, np.float32)
    for insts in GT[obs]:
        acc += paint(insts, img.shape)
    Y_all[i] = np.round(cv2.resize(acc / max(len(GT[obs]), 1), (S, S), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    GEOM[obs] = disk_geometry(img)
test_small = []
for obs in tqdm(test_obs, desc="caching test"):
    img = load_gray(best_path[obs])
    test_small.append(cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA))
    GEOM[obs] = disk_geometry(img)
idx_of = {o: i for i, o in enumerate(train_obs)}
print(f"cache built in {time.time() - t0:.0f}s")

# %% 4 · Dataset, model, training loop --------------------------------------------------------
MEAN, STD = 0.449, 0.226
_rng = None


def rng():
    global _rng                      # one generator per DataLoader worker (np.random is seeded per worker)
    if _rng is None:
        _rng = np.random.default_rng(np.random.randint(2 ** 31))
    return _rng


class FilamentDS(Dataset):
    def __init__(self, idx):
        self.idx = list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        x, y = X_all[self.idx[k]], Y_all[self.idx[k]]
        r = rng()
        kk = r.integers(4)                                          # dihedral: the Sun has no "up"
        x, y = np.rot90(x, kk), np.rot90(y, kk)
        if r.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        if r.random() < 0.5:                                        # small rotation / scale / shift
            M = cv2.getRotationMatrix2D((S / 2, S / 2), r.uniform(-45, 45), r.uniform(0.95, 1.05))
            M[:, 2] += r.uniform(-0.01, 0.01, 2) * S
            x = cv2.warpAffine(x, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
            y = cv2.warpAffine(y, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
        a, b, g = r.uniform(0.85, 1.15), r.uniform(-15, 15), r.uniform(0.8, 1.25)   # contrast, brightness, gamma
        x = cv2.LUT(x, np.clip(255 * (np.clip(np.arange(256) * a + b, 0, 255) / 255) ** g, 0, 255).astype(np.uint8))
        if r.random() < 0.2:
            x = cv2.GaussianBlur(x, (0, 0), r.uniform(0.5, 1.2))   # seeing
        if r.random() < 0.2:
            x = np.clip(x + r.standard_normal(x.shape, dtype=np.float32) * r.uniform(1, 4), 0, 255).astype(np.uint8)
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])  # uint8; normalised on the GPU


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


def build_model():
    m = smp.Unet(CFG["encoder"], encoder_weights=CFG["encoder_weights"], in_channels=1, classes=1)
    m = m.to(DEVICE).to(memory_format=MEMFMT)
    return nn.DataParallel(m) if N_GPU > 1 else m


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


@torch.no_grad()
def predict(model, images_u8, tta=CFG["tta"]):
    """uint8 SxS images -> uint8 SxS probability maps (flip TTA averages 4 views)."""
    model.eval()
    out, bs = [], max(1, CFG["batch_size"])
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


def train_fold(k):
    trn = [o for o in train_obs if fold_of[o] != k]
    val = [o for o in train_obs if fold_of[o] == k]
    val_small = [X_all[idx_of[o]] for o in val]
    print(f"\n===== fold {k}: {len(trn)} train / {len(val)} validation images =====")
    dl = DataLoader(FilamentDS([idx_of[o] for o in trn]), batch_size=CFG["batch_size"], shuffle=True,
                    drop_last=len(trn) > CFG["batch_size"], num_workers=CFG["num_workers"],
                    pin_memory=DEVICE.type == "cuda", persistent_workers=CFG["num_workers"] > 0)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    steps, warm = max(1, len(dl)) * CFG["epochs"], max(1, len(dl)) * CFG["warmup_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    log, best, t_fold = [], -1.0, time.time()
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        tot = n = 0
        t_data = t_gpu = 0.0
        tic = time.time()
        for x, y in dl:
            t1 = time.time(); t_data += t1 - tic
            x, y = to_input(x), y.to(DEVICE, non_blocking=True).float() / 255
            with autocast():
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)     # .item() syncs, so timings are honest
            tic = time.time(); t_gpu += tic - t1
        row = dict(fold=k, epoch=epoch, train_loss=tot / max(n, 1), lr=opt.param_groups[0]["lr"],
                   epoch_min=(t_data + t_gpu) / 60, data_wait=t_data / max(t_data + t_gpu, 1e-9),
                   minutes=(time.time() - t_fold) / 60)
        if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
            res = evaluate(predict(model, val_small, tta=False), val, single(DEFAULT)).iloc[0]
            row.update(val_PQ_default=float(res.PQ), val_SQ=float(res.SQ), val_RQ=float(res.RQ))
            if res.PQ > best:
                best = res.PQ
                torch.save(unwrap(model).state_dict(), OUT / f"best_fold{k}.pt")
                row["saved"] = True
        log.append(row)
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()})
    pd.DataFrame(log).to_csv(OUT / f"train_log_fold{k}.csv", index=False)
    unwrap(model).load_state_dict(torch.load(OUT / f"best_fold{k}.pt", map_location=DEVICE))
    val_probs, test_probs = predict(model, val_small), predict(model, test_small)
    np.savez_compressed(OUT / f"val_probs_fold{k}.npz", obs=np.array(val), probs=np.stack(val_probs))
    if test_probs:
        np.savez_compressed(OUT / f"test_probs_fold{k}.npz", obs=np.array(test_obs), probs=np.stack(test_probs))
    del model, opt, sched, scaler, dl
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return dict(val=val, val_probs=val_probs, test_probs=test_probs)


# %% 5 · Train the folds -----------------------------------------------------------------------
RUN = {}
for k in CFG["folds"]:
    RUN[k] = train_fold(k)

# %% 6 · Gather out-of-fold and test maps (this run + any attached earlier runs) ------------------
OOF, OOF_FOLD, TEST = {}, {}, collections.defaultdict(dict)   # obs -> map; obs -> fold; fold -> {obs: map}
for k, r in RUN.items():
    for o, p in zip(r["val"], r["val_probs"]):
        OOF[o], OOF_FOLD[o] = p, k
    for o, p in zip(test_obs, r["test_probs"]):
        TEST[k][o] = p
for p in sorted(all_files):
    m = re.fullmatch(r"val_probs_fold(\d+)\.npz", p.name)
    if not m or int(m.group(1)) in RUN:
        continue
    k = int(m.group(1))
    d = np.load(p)
    obs = [str(o) for o in d["obs"]]
    wrong = sum(fold_of.get(o) != k for o in obs)
    if wrong:
        print(f"skipping {p.relative_to(INPUT_ROOT)}: {wrong} images are not in fold {k} of these folds")
        continue
    for o, pr in zip(obs, d["probs"]):
        OOF[o], OOF_FOLD[o] = pr, k
    tp = p.with_name(f"test_probs_fold{k}.npz")
    if not tp.exists() and (p.parent / "test_probs.npz").exists():
        tp = p.parent / "test_probs.npz"                       # 02 saved fold 0's test maps under this name
    if tp.exists():
        dt = np.load(tp)
        TEST[k] = {str(o): pr for o, pr in zip(dt["obs"], dt["probs"])}
    print(f"loaded fold {k} from {p.parent.relative_to(INPUT_ROOT)} ({len(obs)} validation maps"
          f"{', with test maps' if k in TEST else ''})")
folds_have = sorted(set(OOF_FOLD.values()))
print(f"out-of-fold maps for {len(OOF):,} of {len(train_obs):,} train images, folds {folds_have}; "
      f"test maps from folds {sorted(TEST)}")

# %% 7 · Tune post-processing on all out-of-fold maps -------------------------------------------------
oof_obs = [o for o in train_obs if o in OOF]
grid = evaluate([OOF[o] for o in oof_obs], oof_obs, CFG["grid"])
grid.to_csv(OUT / "oof_grid.csv", index=False)
best = grid.iloc[0]
params = dict(thr=float(best.thr), min_area=int(best.min_area), r_max=float(best.r_max), min_score=float(best.min_score))
g_ = CFG["grid"]                                    # best value on a grid edge that isn't a natural bound
edge = [k for k, hit in [("min_area", params["min_area"] == max(g_["min_area"]) > min(g_["min_area"])),
                         ("min_score", params["min_score"] == max(g_["min_score"]) > min(g_["min_score"])),
                         ("thr", len(g_["thr"]) > 1 and params["thr"] in (min(g_["thr"]), max(g_["thr"]))),
                         ("r_max", params["r_max"] == min(g_["r_max"]) < max(g_["r_max"]))] if hit]
default = evaluate([OOF[o] for o in oof_obs], oof_obs, single(DEFAULT)).iloc[0]
per_fold = {}
for k in folds_have:
    ob = [o for o in oof_obs if OOF_FOLD[o] == k]
    per_fold[k] = float(evaluate([OOF[o] for o in ob], ob, single(params)).iloc[0].PQ)
hum = np.zeros(6)
for o in oof_obs:
    for i, j in itertools.combinations(range(len(GT[o])), 2):
        hum += score_tables(*pq_tables(GT[o][i], GT[o][j]))
human = pq_parts(*hum)
json.dump(dict(params=params, oof_PQ=float(best.PQ), oof_SQ=float(best.SQ), oof_RQ=float(best.RQ),
               oof_PQ_pooled_gt=float(best.PQ_pooled_gt), oof_PQ_default=float(default.PQ), per_fold_PQ=per_fold,
               human_PQ=float(human[0]), folds=folds_have, cfg=CFG), open(OUT / "params.json", "w"), indent=1)
print("\ntop post-processing settings on all out-of-fold maps:")
print(grid.head(10).round(4).to_string(index=False))

# %% 8 · Ensemble the test maps and write the submission ------------------------------------------
rows, n_empty, used = [], 0, sorted(TEST)
for obs in tqdm(test_obs, desc="ensemble + post-process"):
    maps = [cv2.resize(TEST[k][obs], (S, S), interpolation=cv2.INTER_LINEAR) if TEST[k][obs].shape[0] != S
            else TEST[k][obs] for k in used if obs in TEST[k]]
    if not maps:
        n_empty += 1
        continue
    prob = to_full(np.round(np.mean(maps, axis=0)).astype(np.uint8))
    comps, feats, cents = components(prob, params["thr"])
    keep = keep_mask(feats, cents, GEOM[obs], params)
    k = 0
    for (m, y0, x0), kp in zip(comps, keep):
        if not kp:
            continue
        full = np.zeros((H, W), np.uint8)
        full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        k += 1
        rows.append((f"{obs}_{k}", rle_encode(full)))
    n_empty += k == 0
sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
sub.to_csv(SUB_PATH, index=False)

print("=" * 30, "RESULTS", "=" * 30)
print(f"folds trained this run: {list(RUN)}; out-of-fold maps: {len(oof_obs)} images from folds {folds_have}")
print(f"out-of-fold, vs each annotator : PQ {best.PQ:.3f} (SQ {best.SQ:.3f}, RQ {best.RQ:.3f})  with {params}")
print(f"out-of-fold, vs pooled annotators: PQ {best.PQ_pooled_gt:.3f}")
print(f"out-of-fold, default post-proc : PQ {default.PQ:.3f}")
print("per fold (tuned settings)      : " + ", ".join(f"fold {k} {v:.3f}" for k, v in per_fold.items()))
print(f"human vs human, same images    : PQ {human[0]:.3f}")
if edge:
    print(f"note: best {', '.join(edge)} sit at the edge of the grid; widen it in CFG['grid'] next time")
print(f"test: ensemble of {len(used)} fold models, {len(sub):,} filaments over {len(test_obs) - n_empty} of "
      f"{len(test_obs)} images -> {SUB_PATH}")
