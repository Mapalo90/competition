# %% [markdown]
# # 02 · Baseline: 1024² U-Net, Panoptic Quality validation, submission
#
# **Where to run it:** a Kaggle notebook created from the competition page, with
# Accelerator = **GPU T4 x2** and Internet = **On** (to install segmentation-models-pytorch and
# download ImageNet weights). Use **Save Version -> Save & Run All** so it keeps running if you
# close the tab. About 45-60 minutes with the defaults.
#
# **What it does** (each choice comes from the 01 EDA):
# 1. Training target per image = average of its annotators' masks (soft labels), at 1024x1024.
#    Half resolution keeps a perfect prediction at median IoU 0.98, so full-res tiles aren't needed.
# 2. 5 folds over images, all annotators of an image in the same fold (mirrors the official split).
# 3. U-Net with an ImageNet ResNet34 encoder, BCE + soft Dice, mixed precision, both GPUs.
# 4. Scores the validation fold with PQ two ways: against each annotator separately (comparable to
#    the 0.34 human-vs-human agreement) and against all annotators pooled into one list.
# 5. Tunes post-processing on that fold: probability threshold, minimum area, limb cut (labels
#    never go past ~0.96 R), minimum instance confidence.
# 6. Predicts the test images and writes /kaggle/working/submission.csv.
#
# Outputs in /kaggle/working/baseline/: best.pt, params.json, val_grid.csv, train_log.csv,
# val/test probability maps (to re-tune post-processing without a GPU), figures.

# %% 0 · Config --------------------------------------------------------------------------------
import os, re, sys, json, math, time, random, subprocess, collections, warnings
from pathlib import Path

CFG = dict(
    fold=0, n_folds=5, seed=42,
    img_size=1024,                                    # model input (from 2048 originals)
    encoder="resnet34", encoder_weights="imagenet",
    epochs=40, batch_size=8, lr=3e-4, weight_decay=1e-4, warmup_epochs=1,
    num_workers=4, amp=True, tta=True,
    eval_every=5,                                     # full PQ evaluation every N epochs
    grid=dict(thr=[0.3, 0.4, 0.5, 0.6], min_area=[0, 100, 200, 400],
              r_max=[0.90, 0.95, 1.01], min_score=[0.0, 0.5, 0.6, 0.7]),
)
CFG.update(json.loads(os.environ.get("BASE_CFG", "{}")))      # overrides, used for local tests
INPUT_ROOT = Path(os.environ.get("BASE_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("BASE_OUT", "/kaggle/working/baseline"))
OUT.mkdir(parents=True, exist_ok=True)
SUB_PATH = Path(os.environ.get("BASE_SUB", "/kaggle/working/submission.csv"))
H, W = 2048, 2048
print(json.dumps(CFG, indent=1))

# %% 1 · Imports -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
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
print(f"device: {DEVICE}, GPUs: {N_GPU}, torch {torch.__version__}, smp {smp.__version__}")
if DEVICE.type == "cpu":
    print("WARNING: no GPU. Set Accelerator = GPU T4 x2 in the notebook settings, or this will take hours.")
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

# %% 2 · Helpers (same parsing and PQ bookkeeping as 01_eda) -------------------------------------
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
    """Intersection table between GT and predicted instances, plus their areas.
    Only pairs whose bounding boxes overlap are intersected, so thousands of noise blobs stay cheap."""
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
    """PQ counts from an intersection table: matches need IoU > 0.5 (one-to-one)."""
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
    """Solar disk centre (y, x) and radius in full-res pixels, robust to the off-disk halo that fooled
    a plain threshold in ~14% of images: find the limb as the steepest brightness drop along rays
    from a rough centre, then least-squares fit a circle to those edge points (twice, dropping outliers)."""
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
        k = np.argmin(np.diff(prof, axis=1), axis=1)                 # steepest drop on each ray
        ey, ex = ys[np.arange(len(th_)), k], xs[np.arange(len(th_)), k]
        keep = np.ones(len(th_), bool)
        for _ in range(3):                                          # algebraic circle fit, drop outliers
            A = np.column_stack([ex[keep], ey[keep], np.ones(keep.sum())])
            b = ex[keep] ** 2 + ey[keep] ** 2
            (a1, a2, a3), *_ = np.linalg.lstsq(A, b, rcond=None)
            cx_, cy_ = a1 / 2, a2 / 2
            r_ = math.sqrt(max(a3 + cx_ ** 2 + cy_ ** 2, 1e-6))
            res = np.abs(np.hypot(ex - cx_, ey - cy_) - r_)
            keep = res < max(3.0, 2.5 * np.median(res[keep]))
        cy, cx = cy_, cx_
    return cy * f + (f - 1) / 2, cx * f + (f - 1) / 2, r_ * f


# %% 3 · Find the data ------------------------------------------------------------------------
all_files = sorted(p for p in INPUT_ROOT.rglob("*") if p.is_file())
img_files = [p for p in all_files if ext_of(p) in IMG_EXTS
             and not re.search(r"mask|label|annot", str(p.parent.relative_to(INPUT_ROOT)).lower())]
files = pd.DataFrame([dict(obs=parse_obs(p.name)[0], path=str(p), split=split_of(p), rank=IMG_EXTS.index(ext_of(p)))
                      for p in img_files], columns=["obs", "path", "split", "rank"])
best_path = files.sort_values("rank").drop_duplicates("obs").set_index("obs")["path"].to_dict()

SETS = collections.defaultdict(list)          # obs -> list of annotation sets (each a list of annotations)
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
        obs = parse_obs(im.get("file_name") or im["id"])[0]
        SETS[obs].append(by_img.get(im["id"], []))
    print(f"annotations from {p.relative_to(INPUT_ROOT)}: {len(d['images']):,} sets, {len(d['annotations']):,} filaments")

train_obs = sorted(o for o in SETS if o in best_path)
test_obs = sorted(set(files.loc[files.split == "test", "obs"]))
print(f"train images: {len(train_obs):,} ({sum(len(SETS[o]) for o in train_obs):,} annotation sets) | "
      f"test images: {len(test_obs):,}")
assert train_obs, "No training annotations found. Run 01_eda.py and check the inventory."

# %% 4 · Folds: by image, all annotators together, stratified by year --------------------------
from sklearn.model_selection import KFold, StratifiedKFold
years = np.array([parse_obs(o)[1].year if pd.notna(parse_obs(o)[1]) else 0 for o in train_obs])
counts = collections.Counter(years)
if min(counts.values()) >= CFG["n_folds"]:
    splitter = StratifiedKFold(CFG["n_folds"], shuffle=True, random_state=CFG["seed"]).split(train_obs, years)
else:
    splitter = KFold(CFG["n_folds"], shuffle=True, random_state=CFG["seed"]).split(train_obs)
fold_of = {}
for k, (_, va) in enumerate(splitter):
    for i in va:
        fold_of[train_obs[i]] = k
pd.DataFrame({"obs": train_obs, "fold": [fold_of[o] for o in train_obs]}).to_csv(OUT / "folds.csv", index=False)
trn = [o for o in train_obs if fold_of[o] != CFG["fold"]]
val = [o for o in train_obs if fold_of[o] == CFG["fold"]]
print(f"fold {CFG['fold']}: {len(trn)} train / {len(val)} validation images")

# %% 5 · Cache images and soft targets at model resolution -----------------------------------
S = CFG["img_size"]


def build_cache(obs_list, with_target=True):
    X = np.zeros((len(obs_list), S, S), np.uint8)
    Y = np.zeros((len(obs_list), S, S), np.uint8) if with_target else None
    for i, obs in enumerate(tqdm(obs_list, desc=f"caching {len(obs_list)} images")):
        img = load_gray(best_path[obs])
        X[i] = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        if with_target:
            acc = np.zeros(img.shape, np.float32)
            for anns in SETS[obs]:
                acc += paint([r for r in (decode_crop(a, *img.shape) for a in anns) if r], img.shape)
            acc /= max(len(SETS[obs]), 1)                       # fraction of annotators marking each pixel
            Y[i] = np.round(cv2.resize(acc, (S, S), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    return X, Y


t0 = time.time()
X_all, Y_all = build_cache(train_obs)
idx_of = {o: i for i, o in enumerate(train_obs)}
print(f"cache built in {time.time() - t0:.0f}s; mean target coverage {Y_all.mean() / 255:.4f} of pixels")

# GT instances at full resolution for the validation images, one list per annotator
GT = {o: [[r for r in (decode_crop(a) for a in anns) if r] for anns in SETS[o]] for o in val}

# %% 6 · Dataset and augmentation ------------------------------------------------------------
MEAN, STD = 0.449, 0.226                       # ImageNet mean/std averaged over RGB (1-channel input)


class FilamentDS(Dataset):
    def __init__(self, idx, train):
        self.idx, self.train = list(idx), train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        x = X_all[self.idx[k]].copy()
        y = Y_all[self.idx[k]].copy()
        if self.train:
            x, y = self.augment(x, y)
        x = (x.astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x[None]), torch.from_numpy(y[None].astype(np.float32) / 255)

    @staticmethod
    def augment(x, y):
        rs = np.random
        k = rs.randint(4)                                            # dihedral: the Sun has no "up"
        x, y = np.rot90(x, k), np.rot90(y, k)
        if rs.rand() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        if rs.rand() < 0.5:                                          # small rotation / scale / shift
            M = cv2.getRotationMatrix2D((S / 2, S / 2), rs.uniform(-45, 45), rs.uniform(0.95, 1.05))
            M[:, 2] += rs.uniform(-0.01, 0.01, 2) * S
            x = cv2.warpAffine(x, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
            y = cv2.warpAffine(y, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
        xf = x.astype(np.float32)
        xf = xf * rs.uniform(0.85, 1.15) + rs.uniform(-15, 15)        # contrast / brightness
        xf = 255 * (np.clip(xf, 0, 255) / 255) ** rs.uniform(0.8, 1.25)  # gamma
        if rs.rand() < 0.2:
            xf = cv2.GaussianBlur(xf, (0, 0), rs.uniform(0.5, 1.2))   # seeing
        if rs.rand() < 0.2:
            xf = xf + rs.normal(0, rs.uniform(1, 4), xf.shape)
        return np.clip(xf, 0, 255).astype(np.uint8), y


trn_idx, val_idx = [idx_of[o] for o in trn], [idx_of[o] for o in val]
dl_tr = DataLoader(FilamentDS(trn_idx, True), batch_size=CFG["batch_size"], shuffle=True, drop_last=len(trn_idx) > CFG["batch_size"],
                   num_workers=CFG["num_workers"], pin_memory=DEVICE.type == "cuda", persistent_workers=CFG["num_workers"] > 0)

# %% 7 · Model, loss, optimiser ---------------------------------------------------------------
model = smp.Unet(CFG["encoder"], encoder_weights=CFG["encoder_weights"], in_channels=1, classes=1)
model = model.to(DEVICE)
if N_GPU > 1:
    model = nn.DataParallel(model)
bce = nn.BCEWithLogitsLoss()


def loss_fn(logits, y):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return bce(logits, y) + dice.mean()


opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
steps = max(1, len(dl_tr)) * CFG["epochs"]
warm = max(1, len(dl_tr)) * CFG["warmup_epochs"]
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
try:
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


def autocast():
    return torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP)


@torch.no_grad()
def predict(images_u8, tta=CFG["tta"], bs=None):
    """uint8 SxS images -> float32 probability maps (flip TTA averages 4 views)."""
    model.eval()
    bs = bs or max(1, CFG["batch_size"])
    out = []
    for i in range(0, len(images_u8), bs):
        x = torch.from_numpy((np.stack(images_u8[i:i + bs]).astype(np.float32) / 255 - MEAN) / STD)[:, None].to(DEVICE)
        views = [(lambda t: t, lambda t: t)]
        if tta:
            views += [(lambda t: t.flip(-1), lambda t: t.flip(-1)), (lambda t: t.flip(-2), lambda t: t.flip(-2)),
                      (lambda t: t.flip(-1).flip(-2), lambda t: t.flip(-2).flip(-1))]
        p = 0
        for fwd, back in views:
            with autocast():
                p = p + back(torch.sigmoid(model(fwd(x)).float()))
        out += list((p / len(views))[:, 0].cpu().numpy())
    return out


# %% 8 · Post-processing and PQ evaluation -----------------------------------------------------
def components(prob_u8, thr, min_pix=10):
    """Connected components of prob >= thr at full resolution, ignoring specks under `min_pix` pixels
    (the smallest labelled filaments are ~100 px, so specks can never match one).
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


def evaluate(probs_full, obs_list, geoms, grid, gt=GT):
    """Score every post-processing combination. Returns a DataFrame, one row per combination."""
    combos = [(a, r, s) for a in grid["min_area"] for r in grid["r_max"] for s in grid["min_score"]]
    rows = []
    for thr in grid["thr"]:
        acc_ann = np.zeros((len(combos), 6)); acc_pool = np.zeros((len(combos), 6))
        for prob, obs in zip(probs_full, obs_list):
            comps, feats, cents = components(prob, thr)
            cy, cx, R = geoms[obs]
            rr = np.array([math.hypot(c[1] - cy, c[0] - cx) / R for c in cents]) if len(comps) else np.zeros(0)
            tables = [pq_tables(g, comps) for g in gt[obs]]
            pooled = (np.vstack([t[0] for t in tables]), np.concatenate([t[1] for t in tables]), tables[0][2]) \
                if tables else (np.zeros((0, len(comps))), np.zeros(0), np.array([m.sum() for m, _, _ in comps], float))
            for ci, (amin, rmax, smin) in enumerate(combos):
                keep = (feats[:, 0] >= amin) & (rr <= rmax) & (feats[:, 1] >= smin) if len(comps) else np.zeros(0, bool)
                for inter, ag, ap in tables:
                    acc_ann[ci] += score_tables(inter, ag, ap, keep)
                acc_pool[ci] += score_tables(*pooled, keep)
        for ci, (amin, rmax, smin) in enumerate(combos):
            pq, sq, rq = pq_parts(*acc_ann[ci]); pqp, _, _ = pq_parts(*acc_pool[ci])
            rows.append(dict(thr=thr, min_area=amin, r_max=rmax, min_score=smin, PQ=pq, SQ=sq, RQ=rq, PQ_pooled_gt=pqp,
                             tp=acc_ann[ci][0], fp=acc_ann[ci][1], fn=acc_ann[ci][2],
                             splits=acc_ann[ci][4], merges=acc_ann[ci][5]))
    return pd.DataFrame(rows).sort_values("PQ", ascending=False)


def to_full(p):
    return np.round(cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) * 255).clip(0, 255).astype(np.uint8)


val_imgs = [X_all[idx_of[o]] for o in val]
val_geom = {o: disk_geometry(load_gray(best_path[o])) for o in tqdm(val, desc="disk fits (val)")}
DEFAULT = dict(thr=[0.5], min_area=[0], r_max=[1.01], min_score=[0.0])

# human reference on the same validation images
hum = np.zeros(6)
for o in val:
    for i in range(len(GT[o])):
        for j in range(i + 1, len(GT[o])):
            hum += score_tables(*pq_tables(GT[o][i], GT[o][j]))
HUMAN_PQ = pq_parts(*hum)
print(f"human vs human on this fold: PQ {HUMAN_PQ[0]:.3f} (SQ {HUMAN_PQ[1]:.3f}, RQ {HUMAN_PQ[2]:.3f})")

# %% 9 · Train ------------------------------------------------------------------------------
log, best_pq, t0 = [], -1.0, time.time()
for epoch in range(1, CFG["epochs"] + 1):
    model.train()
    tot, n = 0.0, 0
    for x, y in tqdm(dl_tr, desc=f"epoch {epoch}", leave=False):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with autocast():
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item() * x.size(0); n += x.size(0)
    row = dict(epoch=epoch, train_loss=tot / max(n, 1), lr=opt.param_groups[0]["lr"], minutes=(time.time() - t0) / 60)
    if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
        probs = [to_full(p) for p in predict(val_imgs, tta=False)]
        res = evaluate(probs, val, val_geom, DEFAULT).iloc[0]
        row.update(val_PQ_default=res.PQ, val_SQ=res.SQ, val_RQ=res.RQ)
        if res.PQ > best_pq:
            best_pq = res.PQ
            torch.save((model.module if N_GPU > 1 else model).state_dict(), OUT / "best.pt")
            row["saved"] = True
    log.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
pd.DataFrame(log).to_csv(OUT / "train_log.csv", index=False)

# %% 10 · Tune post-processing on the validation fold ------------------------------------------
(model.module if N_GPU > 1 else model).load_state_dict(torch.load(OUT / "best.pt", map_location=DEVICE))
val_probs = predict(val_imgs)                                         # with TTA
np.savez_compressed(OUT / f"val_probs_fold{CFG['fold']}.npz", obs=np.array(val),
                    probs=np.stack([np.round(p * 255).astype(np.uint8) for p in val_probs]))
val_full = [to_full(p) for p in val_probs]
grid = evaluate(val_full, val, val_geom, CFG["grid"])
grid.to_csv(OUT / "val_grid.csv", index=False)
best = grid.iloc[0]
params = dict(thr=float(best.thr), min_area=int(best.min_area), r_max=float(best.r_max), min_score=float(best.min_score))
default = evaluate(val_full, val, val_geom, DEFAULT).iloc[0]
json.dump(dict(params=params, val_PQ=float(best.PQ), val_SQ=float(best.SQ), val_RQ=float(best.RQ),
               val_PQ_pooled_gt=float(best.PQ_pooled_gt), val_PQ_default=float(default.PQ),
               human_PQ=float(HUMAN_PQ[0]), fold=CFG["fold"], cfg=CFG), open(OUT / "params.json", "w"), indent=1)
print("\ntop post-processing settings on the validation fold:")
print(grid.head(10).round(4).to_string(index=False))

# %% 11 · Predict the test images and write the submission ---------------------------------------
rows, n_empty = [], 0
if test_obs:
    test_imgs = [cv2.resize(load_gray(best_path[o]), (S, S), interpolation=cv2.INTER_AREA) for o in tqdm(test_obs, desc="test load")]
    test_probs = predict(test_imgs)
    np.savez_compressed(OUT / "test_probs.npz", obs=np.array(test_obs),
                        probs=np.stack([np.round(p * 255).astype(np.uint8) for p in test_probs]))
    for obs, p in zip(tqdm(test_obs, desc="test post-process"), test_probs):
        prob = to_full(p)
        cy, cx, R = disk_geometry(load_gray(best_path[obs]))
        comps, feats, cents = components(prob, params["thr"])
        k = 0
        for (m, y0, x0), (area, score), c in zip(comps, feats, cents):
            if area < params["min_area"] or score < params["min_score"] or math.hypot(c[1] - cy, c[0] - cx) / R > params["r_max"]:
                continue
            full = np.zeros((H, W), np.uint8)
            full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
            k += 1
            rows.append((f"{obs}_{k}", rle_encode(full)))
        n_empty += k == 0
sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
sub.to_csv(SUB_PATH, index=False)

# %% 12 · A look at validation predictions -----------------------------------------------------
def to_rgb(img):
    v = img[img > 20]
    lo, hi = np.percentile(v if v.size else img, [0.5, 99.5])
    g = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.ascontiguousarray((np.dstack([g] * 3) * 255).astype(np.uint8))


def draw(rgb, insts, color, thickness):
    for m, y0, x0 in insts:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, [c + np.array([[[x0, y0]]], dtype=c.dtype) for c in cs], -1, color, thickness)
    return rgb


show_obs = sorted(val, key=lambda o: -sum(len(g) for g in GT[o]))[:3]
fig, ax = plt.subplots(1, len(show_obs), figsize=(7 * len(show_obs), 7), squeeze=False)
for a, obs in zip(ax[0], show_obs):
    prob = val_full[val.index(obs)]
    comps, feats, cents = components(prob, params["thr"])
    cy, cx, R = val_geom[obs]
    kept = [c for c, f, ce in zip(comps, feats, cents) if f[0] >= params["min_area"] and f[1] >= params["min_score"]
            and math.hypot(ce[1] - cy, ce[0] - cx) / R <= params["r_max"]]
    rgb = to_rgb(load_gray(best_path[obs]))
    draw(rgb, GT[obs][0], (40, 220, 80), 4)
    draw(rgb, kept, (230, 60, 230), 4)
    a.imshow(cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA))
    a.set_title(f"{obs}: green = annotator 1 ({len(GT[obs][0])}), magenta = model ({len(kept)})")
    a.axis("off")
fig.tight_layout()
fig.savefig(OUT / "val_examples.png", dpi=100, bbox_inches="tight")
plt.show()
plt.close(fig)

print("=" * 30, "RESULTS", "=" * 30)
print(f"fold {CFG['fold']} ({len(val)} validation images), {CFG['encoder']} U-Net at {S}px, {CFG['epochs']} epochs, "
      f"{(time.time() - t0) / 60:.0f} min")
print(f"model vs each annotator  : PQ {best.PQ:.3f} (SQ {best.SQ:.3f}, RQ {best.RQ:.3f})  with {params}")
print(f"model vs pooled annotators: PQ {best.PQ_pooled_gt:.3f}")
print(f"default post-processing   : PQ {default.PQ:.3f}  (thr 0.5, no filters)")
print(f"human vs human, same fold : PQ {HUMAN_PQ[0]:.3f}")
print(f"test: {len(sub):,} filaments over {len(test_obs) - n_empty} of {len(test_obs)} images -> {SUB_PATH}")
