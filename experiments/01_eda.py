# %% [markdown]
# # 01 · First-look EDA: Solar Filament Segmentation Challenge 2026
#
# **Where to run it:** a Kaggle notebook created from the competition page ("New Notebook"),
# with Accelerator = None. It's CPU-only, so it costs no GPU quota. Paste the whole file into
# one cell, or split it at the `# %%` markers. Set `MAX_OBS = 100` for a quick first pass.
#
# **What it answers** (each one feeds a modelling decision):
# 1. What is in the data folder: formats, sizes, file names, annotation schema.
# 2. How one filament is defined: pieces per instance, overlaps, gaps to neighbours.
# 3. How thin filaments are, and how much IoU a 1-pixel boundary error or a lower working
#    resolution costs (IoU > 0.5 is the Panoptic Quality match threshold).
# 4. How well human annotators agree with each other, scored with PQ. That's your ceiling.
# 5. How observations spread over time and GONG sites, and whether test observations
#    also appear in train.
# 6. Which dark structures are visible but NOT labelled (future false positives).
#
# Outputs go to /kaggle/working/eda/ (CSVs + PNGs). The SUMMARY block at the end is the part
# to read first.

# %% 0 · Setup and helper functions ------------------------------------------------------
import os, re, sys, json, math, random, itertools, subprocess, collections, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.measure import regionprops

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 160)

INPUT_ROOT = Path(os.environ.get("EDA_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("EDA_OUT", "/kaggle/working/eda"))
OUT.mkdir(parents=True, exist_ok=True)

MAX_OBS = None        # e.g. 100 for a quick pass; None = every annotated observation
N_DARK_CHECK = 40     # observations used for the "dark but unlabelled" check
NEIGHBOR_MAX = 30     # gaps between filaments are measured up to this many pixels
H, W = 2048, 2048     # image size stated on the competition page
SEED = 0
rng = random.Random(SEED)


def _import_pycocotools():
    if os.environ.get("EDA_NO_COCO"):                      # only used to test the fallback
        return None
    try:
        from pycocotools import mask
        return mask
    except ImportError:
        print("pycocotools not found; trying `pip install pycocotools` ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], capture_output=True)
        try:
            from pycocotools import mask
            return mask
        except ImportError:
            print("  install failed (internet off?). Polygons will be filled with OpenCV; RLE checks skipped.")
            return None


mask_utils = _import_pycocotools()
HAS_COCO = mask_utils is not None

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **_):
        return it

# ---- file and name parsing -------------------------------------------------------------
OBS_RE = re.compile(r"(\d{14})([A-Za-z])h")                 # GONG name, e.g. 20160920230134Lh
IMG_EXTS = [".fits.fz", ".fits", ".fts", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]  # preference order


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def ext_of(p):
    return ".fits.fz" if p.name.lower().endswith(".fits.fz") else p.suffix.lower()


def parse_obs(s):
    """'010401-20160920230134Lh.jpg' -> ('20160920230134Lh', Timestamp, 'L')."""
    m = OBS_RE.search(str(s))
    if not m:
        return Path(str(s)).name.split(".")[0], pd.NaT, None
    return m.group(0), pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S", errors="coerce"), m.group(2).upper()


def split_of(path):
    """Guess train/val/test from the folder and file names."""
    text = " ".join(t.lower() for t in Path(path).relative_to(INPUT_ROOT).parts)
    for name in ("test", "val", "train"):
        if re.search(rf"(?<![a-z]){name}", text):
            return name
    return "unknown"


def _flat(x):
    return [v for item in x for v in (_flat(item) if isinstance(item, (list, tuple)) else [item])]


def polyline_len(spine):
    pts = np.asarray(_flat(spine or []), float)
    if pts.size < 4:
        return 0.0
    pts = pts[: pts.size // 2 * 2].reshape(-1, 2)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def load_gray(path):
    path = str(path)
    if path.lower().endswith((".fits", ".fts", ".fz")):
        try:
            from astropy.io import fits                      # only needed if images are FITS
        except ImportError as e:
            raise ImportError("FITS images need astropy: run `!pip install astropy`") from e
        with fits.open(path) as hdul:
            data = next(h.data for h in hdul if h.data is not None and np.ndim(h.data) == 2)
            return np.nan_to_num(np.asarray(data, np.float32))
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        from PIL import Image
        img = np.asarray(Image.open(path))
    if img.ndim == 3:
        img = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_BGR2GRAY)
    return img


# ---- masks -----------------------------------------------------------------------------
def decode_crop(a, h=H, w=W, pad=2):
    """Rasterise one annotation into a tight crop. Returns (bool mask, y0, x0) or None.

    Polygons are shifted by an integer offset before rasterising, which gives the same
    pixels as pycocotools' full-image annToMask but is much faster and lighter."""
    seg = a.get("segmentation")
    if not seg:
        return None
    if isinstance(seg, list):                                  # polygon(s)
        polys = [np.asarray(s, float).reshape(-1, 2) for s in seg if len(s) >= 6]
        if not polys:
            return None
        allp = np.concatenate(polys)
        x0 = int(max(0, math.floor(allp[:, 0].min()) - pad))
        y0 = int(max(0, math.floor(allp[:, 1].min()) - pad))
        x1 = int(min(w, math.ceil(allp[:, 0].max()) + pad + 1))
        y1 = int(min(h, math.ceil(allp[:, 1].max()) + pad + 1))
        shifted = [(q - [x0, y0]).ravel().tolist() for q in polys]
        if HAS_COCO:
            m = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(shifted, y1 - y0, x1 - x0)))
        else:
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [np.round(np.reshape(s, (-1, 2))).astype(np.int32) for s in shifted], 1)
        m = m.astype(bool)
    else:                                                      # RLE
        if not HAS_COCO:
            return None
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
    """HxW {0,1} mask -> COCO compressed RLE counts string (the submission format)."""
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def rle_decode(counts, h=H, w=W):
    return mask_utils.decode({"size": [h, w], "counts": counts.encode("ascii")})


def paint(insts, shape):
    canvas = np.zeros(shape, bool)
    for m, y0, x0 in insts:
        canvas[y0: y0 + m.shape[0], x0: x0 + m.shape[1]] |= m
    return canvas


# ---- per-filament geometry ---------------------------------------------------------------
def resample_iou(m, f):
    """Best-case IoU if this mask had to be predicted at 1/f resolution and upsampled."""
    hc, wc = m.shape
    H2, W2 = -(-hc // f) * f, -(-wc // f) * f
    big = np.zeros((H2, W2), np.float32)
    big[:hc, :wc] = m
    small = cv2.resize(big, (W2 // f, H2 // f), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (W2, H2), interpolation=cv2.INTER_LINEAR)[:hc, :wc] > 0.5
    union = np.logical_or(up, m).sum()
    return np.logical_and(up, m).sum() / union if union else np.nan


def instance_stats(m):
    c = np.pad(m, 3)
    area = int(c.sum())
    lab, ncomp = ndi.label(c, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel())[1:]
    v = ndi.distance_transform_edt(c)[skeletonize(c)]       # half-widths along the centre line
    k3 = np.ones((3, 3), bool)
    rp = regionprops(c.astype(np.uint8))[0]
    major = rp.axis_major_length if hasattr(rp, "axis_major_length") else rp.major_axis_length
    minor = rp.axis_minor_length if hasattr(rp, "axis_minor_length") else rp.minor_axis_length
    return dict(                                   # widths: 2 x centre-line distance - 0.5, good to +-0.5 px
        area=area, n_components=int(ncomp), largest_piece_frac=sizes.max() / area,
        width_med=2 * np.median(v) - 0.5 if v.size else np.nan,          # typical width (px)
        width_p10=2 * np.percentile(v, 10) - 0.5 if v.size else np.nan,  # thinnest parts (barbs)
        skeleton_px=int(v.size),
        iou_dilate1=area / ndi.binary_dilation(c, k3).sum(),            # 1-px too fat
        iou_erode1=ndi.binary_erosion(c, k3).sum() / area,              # 1-px too thin
        iou_res_half=resample_iou(c, 2), iou_res_quarter=resample_iou(c, 4),
        elongation=major / max(minor, 1.0), solidity=rp.solidity)


def set_relations(insts, dmax=NEIGHBOR_MAX):
    """Within one annotation set: gap (px) from each filament to its nearest neighbour
    (0 = touching), neighbours within 5/10 px, and pixels shared with other filaments."""
    n = len(insts)
    gap, nb5, nb10, ov = np.full(n, np.inf), np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    boxes = [(y0, x0, y0 + m.shape[0], x0 + m.shape[1]) for m, y0, x0 in insts]
    for i, (mi, yi, xi) in enumerate(insts):
        Y0, X0, Y1, X1 = yi - dmax, xi - dmax, boxes[i][2] + dmax, boxes[i][3] + dmax
        win = np.zeros((Y1 - Y0, X1 - X0), bool)
        win[yi - Y0: yi - Y0 + mi.shape[0], xi - X0: xi - X0 + mi.shape[1]] = mi
        dt = None
        for j, (mj, yj, xj) in enumerate(insts):
            b = boxes[j]
            if j == i or b[0] >= Y1 or b[2] <= Y0 or b[1] >= X1 or b[3] <= X0:
                continue
            oy0, ox0, oy1, ox1 = max(yj, Y0), max(xj, X0), min(b[2], Y1), min(b[3], X1)
            sub = mj[oy0 - yj: oy1 - yj, ox0 - xj: ox1 - xj]
            if not sub.any():
                continue
            if dt is None:
                dt = ndi.distance_transform_edt(~win)
            g = max(dt[oy0 - Y0: oy1 - Y0, ox0 - X0: ox1 - X0][sub].min() - 1.0, 0.0)
            ov[i] += int(np.logical_and(win[oy0 - Y0: oy1 - Y0, ox0 - X0: ox1 - X0], sub).sum())
            gap[i] = min(gap[i], g)
            nb5[i] += g <= 5
            nb10[i] += g <= 10
    return gap, nb5, nb10, ov


# ---- PQ between two sets of instances (used here for annotator vs annotator) ----------------
def _inter(a, b):
    (ma, ya, xa), (mb, yb, xb) = a, b
    y0, x0 = max(ya, yb), max(xa, xb)
    y1, x1 = min(ya + ma.shape[0], yb + mb.shape[0]), min(xa + ma.shape[1], xb + mb.shape[1])
    if y0 >= y1 or x0 >= x1:
        return 0
    return int(np.logical_and(ma[y0 - ya: y1 - ya, x0 - xa: x1 - xa], mb[y0 - yb: y1 - yb, x0 - xb: x1 - xb]).sum())


def compare_sets(gt, pred):
    """Panoptic-Quality bookkeeping for two lists of (mask, y0, x0): matches need IoU > 0.5.
    'splits' = GT filaments with 2+ predictions mostly inside them (one-to-many);
    'merges' = predictions with 2+ GT filaments mostly inside them (many-to-one)."""
    ag = np.array([m.sum() for m, _, _ in gt], float)
    ap = np.array([m.sum() for m, _, _ in pred], float)
    inter = np.array([[_inter(g, p) for p in pred] for g in gt], float).reshape(len(gt), len(pred))
    union = ag[:, None] + ap[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    tp_iou, used_g, used_p = [], set(), set()
    for gi, pj in sorted(zip(*np.nonzero(iou > 0.5)), key=lambda t: -iou[t]):
        if gi not in used_g and pj not in used_p:
            used_g.add(gi)
            used_p.add(pj)
            tp_iou.append(iou[gi, pj])
    tp = len(tp_iou)
    in_gt = np.divide(inter, ap[None, :], out=np.zeros_like(inter), where=ap[None, :] > 0) >= 0.5
    in_pred = np.divide(inter, ag[:, None], out=np.zeros_like(inter), where=ag[:, None] > 0) >= 0.5
    return dict(tp=tp, fp=len(pred) - tp, fn=len(gt) - tp, sum_iou=float(np.sum(tp_iou)),
                splits=int((in_gt.sum(1) >= 2).sum()), merges=int((in_pred.sum(0) >= 2).sum()))


def pq_parts(tp, fp, fn, sum_iou):
    denom = tp + 0.5 * fp + 0.5 * fn
    return (sum_iou / denom if denom else np.nan, sum_iou / tp if tp else np.nan, tp / denom if denom else np.nan)


# ---- images ------------------------------------------------------------------------------
def disk_geometry(img, f=8):
    """Solar disk centre (y, x) and radius in pixels, from a thresholded thumbnail."""
    small = cv2.resize(img.astype(np.float32), (img.shape[1] // f, img.shape[0] // f), interpolation=cv2.INTER_AREA)
    s8 = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((th > 0).astype(np.uint8))
    if n <= 1:
        return np.nan, np.nan, np.nan
    disk = ndi.binary_fill_holes(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(disk)
    return ys.mean() * f + (f - 1) / 2, xs.mean() * f + (f - 1) / 2, math.sqrt(disk.sum() / math.pi) * f


def dark_candidates(img, cy, cx, r, sigma=25, thr=0.85, min_area=40):
    """Pixels at least 15% darker than their smoothed surroundings, inside the disk.
    A crude classical filament/sunspot detector, used only to find unlabelled dark structures."""
    f = img.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), sigma)
    yy, xx = np.ogrid[: f.shape[0], : f.shape[1]]
    dark = (f < thr * bg) & ((yy - cy) ** 2 + (xx - cx) ** 2 < (0.93 * r) ** 2)
    lab, _ = ndi.label(dark, structure=np.ones((3, 3), bool))
    keep = np.bincount(lab.ravel()) >= min_area
    keep[0] = False
    return keep[lab]


def to_rgb(img, geom=None):
    f = img.astype(np.float32)
    if geom is not None and np.isfinite(geom[2]):
        yy, xx = np.ogrid[: f.shape[0], : f.shape[1]]
        v = f[(yy - geom[0]) ** 2 + (xx - geom[1]) ** 2 < (0.95 * geom[2]) ** 2]
    else:
        v = f.ravel()
    lo, hi = np.percentile(v[:: max(1, v.size // 200_000)], [0.5, 99.5])
    g = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.ascontiguousarray((np.dstack([g] * 3) * 255).astype(np.uint8))


def draw(rgb, insts, color, thickness=2):
    for m, y0, x0 in insts:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, [c + np.array([[[x0, y0]]], dtype=c.dtype) for c in cs], -1, color, thickness)
    return rgb


PALETTE = [tuple(int(255 * c) for c in plt.cm.tab10(i)[:3]) for i in range(10)]


def show(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# %% 1 · What is in the input folder? -------------------------------------------------------
all_files = sorted(p for p in INPUT_ROOT.rglob("*") if p.is_file())
print(f"{len(all_files):,} files under {INPUT_ROOT}\n")
if not all_files:
    print("Nothing attached. In the notebook's right-hand panel: Add Input -> Competitions -> "
          "Solar Filament Segmentation Challenge 2026, then run again.")
by_dir, dir_bytes = collections.defaultdict(collections.Counter), collections.Counter()
for p in all_files:
    d = str(p.parent.relative_to(INPUT_ROOT))
    by_dir[d][ext_of(p) or "(none)"] += 1
    dir_bytes[d] += p.stat().st_size
for d in sorted(by_dir)[:50]:
    print(f"  {d:<58} {human(dir_bytes[d]):>9}   " + ", ".join(f"{e} x{n}" for e, n in by_dir[d].most_common(4)))
if len(by_dir) > 50:
    print(f"  ... and {len(by_dir) - 50} more folders")
print("\nMetadata files:")
for p in all_files:
    if ext_of(p) in {".json", ".csv", ".txt", ".md", ".yaml", ".yml", ".py", ".ipynb"}:
        print(f"  {p.relative_to(INPUT_ROOT)}  ({human(p.stat().st_size)})")

# %% 2 · Load annotations and index the images -------------------------------------------------
set_rows, ann_rows, ANN, cat_names = [], [], {}, {}      # a "set" = one image as labelled by one annotator
for p in [f for f in all_files if ext_of(f) == ".json"]:
    try:
        with open(p) as fh:
            d = json.load(fh)
    except Exception as e:
        print(f"skipping {p.name}: {e}")
        continue
    if not (isinstance(d, dict) and "images" in d):
        continue
    split = split_of(p)
    print(f"\nCOCO file: {p.relative_to(INPUT_ROOT)}  (split guess: {split})")
    print(f"  keys: {list(d)}; images: {len(d['images']):,}; annotations: {len(d.get('annotations', [])):,}")
    for c in d.get("categories", []):
        cat_names[c["id"]] = c.get("name", str(c["id"]))
    key_of = {}
    for im in d["images"]:
        sid = str(im["id"])
        obs, ts, site = parse_obs(im.get("file_name") or sid)
        key_of[im["id"]] = key = f"{p.stem}:{sid}"
        set_rows.append(dict(set_key=key, image_id=sid, file_name=im.get("file_name"), obs=obs,
                             annotator=sid.split(obs)[0].strip("-_ ") if obs in sid else "",
                             ts=ts, site=site, split=split,
                             height=int(im.get("height") or H), width=int(im.get("width") or W)))
    for a in d.get("annotations", []):
        key = key_of.get(a.get("image_id"))
        if key is None:
            continue
        uid = f"{p.stem}:{a['id']}"
        seg, bbox = a.get("segmentation"), (a.get("bbox") or [np.nan] * 4)
        ann_rows.append(dict(uid=uid, set_key=key, category=cat_names.get(a.get("category_id"), a.get("category_id")),
                             n_polygons=len(seg) if isinstance(seg, list) else 1,
                             n_vertices=sum(len(s) // 2 for s in seg) if isinstance(seg, list) else np.nan,
                             area_json=a.get("area"), bbox_w=bbox[2], bbox_h=bbox[3],
                             spine_len=polyline_len(a.get("spine")), iscrowd=a.get("iscrowd", 0)))
        ANN[uid] = a
    if d.get("annotations"):
        a0 = d["annotations"][0]
        print("  first annotation:", {k: (f"<{type(v).__name__}, len {len(v)}>" if k in ("segmentation", "spine")
                                          and hasattr(v, "__len__") else v) for k, v in a0.items()})
    print("  first image:", d["images"][0])
    if d.get("categories"):
        print("  categories:", d["categories"])

HAVE_COCO_GT = bool(ann_rows)
sample_sub = None
for p in [f for f in all_files if ext_of(f) == ".csv"]:
    try:
        head = pd.read_csv(p, nrows=5)
    except Exception as e:
        print(f"skipping {p.name}: {e}")
        continue
    print(f"\nCSV: {p.relative_to(INPUT_ROOT)}  columns: {list(head.columns)}")
    print(head.head(3).to_string(max_colwidth=50))
    id_col = next((c for c in head.columns if "id" in c.lower()), None)
    rle_col = next((c for c in head.columns if "rle" in c.lower() or "segment" in c.lower()), None)
    if re.search(r"sample|submission", p.name.lower()):
        sample_sub = (p, id_col, rle_col)
    elif not HAVE_COCO_GT and id_col and rle_col:              # no COCO file: use an RLE table as GT
        df, split = pd.read_csv(p), split_of(p)
        print(f"  -> using as ground truth: {len(df):,} RLE rows (split guess: {split})")
        for i, (fid, counts) in enumerate(zip(df[id_col], df[rle_col])):
            obs, ts, site = parse_obs(fid)
            key, uid = f"{p.stem}:{obs}", f"{p.stem}:{i}"
            ANN[uid] = {"segmentation": {"size": [H, W], "counts": str(counts)}}
            ann_rows.append(dict(uid=uid, set_key=key, category=None, n_polygons=1, n_vertices=np.nan,
                                 area_json=np.nan, bbox_w=np.nan, bbox_h=np.nan, spine_len=np.nan, iscrowd=0))
            set_rows.append(dict(set_key=key, image_id=obs, file_name=None, obs=obs, annotator="", ts=ts,
                                 site=site, split=split, height=H, width=W))

file_rows = []
for p in all_files:
    e = ext_of(p)
    if e in IMG_EXTS and not re.search(r"mask|label|annot", str(p.parent.relative_to(INPUT_ROOT)).lower()):
        file_rows.append(dict(obs=parse_obs(p.name)[0], path=str(p), ext=e, split=split_of(p), rank=IMG_EXTS.index(e)))
files = pd.DataFrame(file_rows, columns=["obs", "path", "ext", "split", "rank"])
best_path = files.sort_values("rank").drop_duplicates("obs").set_index("obs")["path"].to_dict()

sets = pd.DataFrame(set_rows, columns=["set_key", "image_id", "file_name", "obs", "annotator", "ts", "site",
                                       "split", "height", "width"]).drop_duplicates("set_key")
anns = pd.DataFrame(ann_rows, columns=["uid", "set_key", "category", "n_polygons", "n_vertices", "area_json",
                                       "bbox_w", "bbox_h", "spine_len", "iscrowd"])
sets["n_instances"] = sets.set_key.map(anns.groupby("set_key").size()).fillna(0).astype(int)
HAVE_GT = len(anns) > 0

print("\n=== Ground truth ===")
if not HAVE_GT:
    print("No annotations found. Paste the inventory above to Claude and the loader can be adapted.")
else:
    per_obs = sets[sets.n_instances > 0].groupby("obs").size()
    print(f"annotation sets (image x annotator): {len(sets):,} | unique observations: {sets.obs.nunique():,} | "
          f"filaments: {len(anns):,}")
    print(f"sets with zero filaments: {(sets.n_instances == 0).sum():,}")
    print("annotation sets per observation:", per_obs.value_counts().sort_index().to_dict())
    print(f"filaments per set: median {sets.n_instances.median():.0f}, max {sets.n_instances.max()}")
    print("categories:", anns.category.value_counts(dropna=False).to_dict())
    print("polygons per annotation:", anns.n_polygons.value_counts().sort_index().to_dict())
    print("sets by split guess:", sets.split.value_counts().to_dict())
print(f"\nimage files: {len(files):,} {files.ext.value_counts().to_dict()} | by folder: {files.split.value_counts().to_dict()}")
missing = sorted(set(sets.loc[sets.n_instances > 0, "obs"]) - set(best_path))
print(f"annotated observations with no image file: {len(missing):,}" + (f" (e.g. {missing[:3]})" if missing else ""))

# %% 3 · Check the submission format: RLE round trip ----------------------------------------
if HAS_COCO and HAVE_GT:
    k0 = anns.set_key.iloc[0]
    hh, ww = (int(v) for v in sets.set_index("set_key").loc[k0, ["height", "width"]])
    crop = next(r for r in (decode_crop(ANN[u], hh, ww) for u in anns.loc[anns.set_key == k0, "uid"]) if r)
    full = paint([crop], (hh, ww)).astype(np.uint8)
    s = rle_encode(full)
    assert np.array_equal(rle_decode(s, hh, ww), full), "RLE round trip failed"
    print(f"RLE round trip OK. Example counts string ({len(s)} chars): {s[:70]}...")
if sample_sub:
    p, id_col, rle_col = sample_sub
    ss = pd.read_csv(p)
    ss_obs = ss[id_col].map(lambda v: parse_obs(v)[0]) if id_col else pd.Series(dtype=object)
    print(f"\nsample submission {p.name}: {len(ss):,} rows, {ss_obs.nunique():,} observations")
    if HAS_COCO and rle_col:
        try:
            px = [int(rle_decode(str(c)).sum()) for c in ss[rle_col].dropna().head(10)]
            print(f"  first {len(px)} RLEs decode to {H}x{W} masks with pixel counts {px}")
        except Exception as e:
            print(f"  could not decode the sample RLEs: {e}")

# %% 4 · Per-filament geometry, neighbours, and annotator-vs-annotator comparison --------------
inst_rows, pair_rows, KEEP = [], [], {}
if HAVE_GT:
    meta = sets.set_index("set_key")
    anns_by_set = anns.groupby("set_key").uid.apply(list).to_dict()
    sets_by_obs = sets[sets.n_instances > 0].groupby("obs").set_key.apply(list).to_dict()
    obs_list = sorted(sets_by_obs)
    if MAX_OBS:
        obs_list = sorted(rng.sample(obs_list, min(MAX_OBS, len(obs_list))))
    multi = [o for o in obs_list if len(sets_by_obs[o]) > 1]
    fig_obs = set(rng.sample(obs_list, min(12, len(obs_list)))) | set(rng.sample(multi, min(2, len(multi))))
    for obs in tqdm(obs_list, desc="filaments"):
        decoded = {}
        for key in sets_by_obs[obs]:
            hh, ww = int(meta.at[key, "height"]), int(meta.at[key, "width"])
            uids, insts = [], []
            for uid in anns_by_set.get(key, []):
                r = decode_crop(ANN[uid], hh, ww)
                if r is None:
                    continue
                m, y0, x0 = r
                yy, xx = np.nonzero(m)
                st = instance_stats(m)
                st.update(uid=uid, set_key=key, obs=obs, y0=y0, x0=x0, cy=y0 + yy.mean(), cx=x0 + xx.mean())
                inst_rows.append(st)
                uids.append(uid)
                insts.append(r)
            gap, nb5, nb10, ov = set_relations(insts)
            for k in range(len(insts)):
                inst_rows[len(inst_rows) - len(insts) + k].update(
                    gap_px=gap[k], neighbors_5px=nb5[k], neighbors_10px=nb10[k], overlap_px=ov[k])
            decoded[key] = (uids, insts)
        for a, b in itertools.combinations(decoded, 2):
            r = compare_sets(decoded[a][1], decoded[b][1])
            r.update(obs=obs, set_a=a, set_b=b, n_a=len(decoded[a][1]), n_b=len(decoded[b][1]))
            pair_rows.append(r)
        if obs in fig_obs:
            KEEP[obs] = decoded
inst = pd.DataFrame(inst_rows)
pairs = pd.DataFrame(pair_rows)

# %% 5 · What does one filament look like? ------------------------------------------------------
if len(inst):
    inst = inst.merge(anns[["uid", "category", "n_polygons", "spine_len"]], on="uid", how="left")
    cols = ["area", "width_med", "width_p10", "elongation", "iou_dilate1", "iou_erode1",
            "iou_res_half", "iou_res_quarter", "gap_px"]
    print(inst[cols].replace(np.inf, np.nan).describe(percentiles=[.1, .25, .5, .75, .9]).T.round(2).to_string())
    print(f"\nfilaments with >1 connected piece: {(inst.n_components > 1).mean():.1%} "
          f"(in those, the largest piece holds {inst.loc[inst.n_components > 1, 'largest_piece_frac'].median():.0%} "
          f"of the area)")
    print(f"filaments touching/overlapping another one: {(inst.gap_px < 1).mean():.1%}; "
          f"with a neighbour within 5 px: {(inst.gap_px <= 5).mean():.1%}; within 10 px: {(inst.gap_px <= 10).mean():.1%}")
    print(f"sets containing overlapping filaments: {inst[inst.overlap_px > 0].set_key.nunique():,} "
          f"of {inst.set_key.nunique():,}")

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    ax[0, 0].hist(sets.loc[sets.n_instances >= 0, "n_instances"], bins=np.arange(0, sets.n_instances.max() + 2) - .5,
                  color="#4C72B0")
    ax[0, 0].set(title="Filaments per annotation set", xlabel="count")
    ax[0, 1].hist(inst.area, bins=np.logspace(np.log10(max(inst.area.min(), 1)), np.log10(inst.area.max() + 1), 40),
                  color="#4C72B0")
    ax[0, 1].set(xscale="log", title="Filament area", xlabel="pixels (log)")
    ax[0, 2].hist([inst.width_med.dropna(), inst.width_p10.dropna()], bins=40, label=["typical width", "thinnest 10%"])
    ax[0, 2].set(title="Filament width (from distance transform)", xlabel="pixels")
    ax[0, 2].legend()
    for c, lbl in [("iou_dilate1", "1-px dilation"), ("iou_erode1", "1-px erosion"),
                   ("iou_res_half", "best case at 1/2 res"), ("iou_res_quarter", "best case at 1/4 res")]:
        ax[1, 0].hist(inst[c].dropna(), bins=np.linspace(0, 1, 41), histtype="step", lw=2, label=lbl)
    ax[1, 0].axvline(0.5, color="k", ls="--", lw=1)
    ax[1, 0].set(title="IoU left after small errors (PQ needs > 0.5)", xlabel="IoU")
    ax[1, 0].legend(fontsize=8)
    g = inst.gap_px[np.isfinite(inst.gap_px)]
    ax[1, 1].hist(g, bins=np.arange(0, NEIGHBOR_MAX + 2), color="#4C72B0")
    ax[1, 1].set(title=f"Gap to nearest filament ({(~np.isfinite(inst.gap_px)).mean():.0%} have none within "
                       f"{NEIGHBOR_MAX}px)", xlabel="pixels (0 = touching)")
    pieces = inst.n_components.clip(upper=4).value_counts().sort_index()
    ax[1, 2].bar([str(i) if i < 4 else "4+" for i in pieces.index], pieces.values, color="#4C72B0")
    ax[1, 2].set(title="Connected pieces per filament", xlabel="pieces")
    fig.tight_layout()
    show(fig, "01_filament_geometry.png")

# %% 6 · How much do human annotators agree? (the realistic PQ ceiling) ------------------------
if len(pairs):
    pq_h, sq_h, rq_h = pq_parts(*pairs[["tp", "fp", "fn", "sum_iou"]].sum())
    pairs["pq"] = [pq_parts(*r)[0] for r in pairs[["tp", "fp", "fn", "sum_iou"]].itertuples(index=False)]
    print(f"{len(pairs):,} annotator pairs on {pairs.obs.nunique():,} observations")
    print(f"human vs human, pooled: PQ = {pq_h:.3f}  (SQ = {sq_h:.3f}, RQ = {rq_h:.3f}); "
          f"mean per-pair PQ = {pairs.pq.mean():.3f}")
    print(f"per pair: |count difference| median {(pairs.n_a - pairs.n_b).abs().median():.0f}, "
          f"splits {pairs.splits.mean():.2f}, merges {pairs.merges.mean():.2f}")
elif HAVE_GT:
    print("No observation has more than one annotation set, so annotator agreement can't be measured here.")

# %% 7 · Images: format, solar disk, brightness, where filaments sit --------------------------
img_stats, n_fail = [], 0
for obs in tqdm(sorted(best_path), desc="image stats"):
    try:
        img = load_gray(best_path[obs])
    except Exception as e:
        n_fail += 1
        print(f"could not read {best_path[obs]}: {e}")
        if n_fail >= 5:
            print("Stopping image stats after 5 failures; fix the reader above and re-run this cell.")
            break
        continue
    cy, cx, r = disk_geometry(img)
    sub = img[::4, ::4].astype(np.float32)
    yy, xx = np.mgrid[0: img.shape[0]: 4, 0: img.shape[1]: 4]
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 < (0.97 * r) ** 2 if np.isfinite(r) else np.ones(sub.shape, bool)
    v = sub[inside] if inside.any() else sub.ravel()
    top = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else np.nan
    img_stats.append(dict(obs=obs, dtype=str(img.dtype), img_h=img.shape[0], img_w=img.shape[1],
                          disk_cy=cy, disk_cx=cx, disk_r=r, disk_mean=v.mean(), disk_std=v.std(),
                          disk_p1=np.percentile(v, 1), disk_p99=np.percentile(v, 99),
                          saturated_frac=(v >= top).mean() if np.isfinite(top) else np.nan))
img_stats = pd.DataFrame(img_stats)
if len(img_stats):
    img_stats["site"] = img_stats.obs.map(lambda o: parse_obs(o)[2])
    print("image dtypes/sizes:", img_stats.groupby(["dtype", "img_h", "img_w"]).size().to_dict())
    print(f"disk radius px: median {img_stats.disk_r.median():.0f}, range {img_stats.disk_r.min():.0f}-"
          f"{img_stats.disk_r.max():.0f}; centre offset from image centre: median "
          f"{np.hypot(img_stats.disk_cy - H / 2, img_stats.disk_cx - W / 2).median():.1f} px")
    print("\non-disk brightness by GONG site (letter in the file name):")
    print(img_stats.groupby("site")[["disk_mean", "disk_std", "disk_p1", "disk_p99", "saturated_frac"]]
          .median().round(3).assign(n=img_stats.groupby("site").size()).to_string())
    if len(inst):
        inst = inst.merge(img_stats[["obs", "disk_cy", "disk_cx", "disk_r"]], on="obs", how="left")
        inst["r_norm"] = np.hypot(inst.cy - inst.disk_cy, inst.cx - inst.disk_cx) / inst.disk_r
        print(f"\nfilament centroids at r/R > 0.9 (near the limb): {(inst.r_norm > 0.9).mean():.1%}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    sites = sorted(img_stats.site.dropna().unique())
    if sites:
        ax[0].boxplot([img_stats.loc[img_stats.site == s, "disk_mean"] for s in sites])
        ax[0].set_xticks(range(1, len(sites) + 1))
        ax[0].set_xticklabels(sites)
    ax[0].set(title="On-disk mean brightness by site", xlabel="site letter")
    ax[1].hist(img_stats.disk_r.dropna(), bins=40, color="#4C72B0")
    ax[1].set(title="Solar disk radius", xlabel="pixels")
    if len(inst) and "r_norm" in inst:
        ax[2].hist(inst.r_norm.dropna(), bins=np.linspace(0, 1.05, 43), color="#4C72B0")
        ax[2].set(title="Filament position (centroid distance / disk radius)", xlabel="r / R")
    fig.tight_layout()
    show(fig, "02_images_and_positions.png")

# %% 8 · Time, sites, and train/test overlap ------------------------------------------------------
train_obs = set(sets.loc[sets.split != "test", "obs"])     # includes labelled-but-empty images
test_obs = set(sets.loc[sets.split == "test", "obs"]) | set(files.loc[files.split == "test", "obs"])
all_obs = train_obs | test_obs | set(files.obs) | set(sets.obs)
obs_tab = pd.DataFrame([dict(obs=o, ts=parse_obs(o)[1], site=parse_obs(o)[2], in_train=o in train_obs,
                             in_test=o in test_obs) for o in sorted(all_obs)])
gap_test = gap_train = np.array([])
if len(obs_tab):
    obs_tab["year"] = obs_tab.ts.dt.year
    obs_tab["role"] = np.select([obs_tab.in_train & obs_tab.in_test, obs_tab.in_test, obs_tab.in_train],
                                ["train+test", "test", "train"], "image only")
    ct_year, ct_site = pd.crosstab(obs_tab.year, obs_tab.role), pd.crosstab(obs_tab.site, obs_tab.role)
    if len(ct_year) and len(ct_site):
        print(pd.crosstab(obs_tab.year, obs_tab.role, margins=True).to_string(), "\n")
        print(pd.crosstab(obs_tab.site, obs_tab.role, margins=True).to_string())
    else:
        print("File names don't follow the GONG pattern (YYYYMMDDhhmmss + site + 'h'), so time/site checks are skipped.")

    def nearest_gap_days(query, ref, same=False):
        ref = np.sort(pd.to_datetime(pd.Series(ref)).dropna().values.astype("datetime64[s]").astype(np.int64))
        out = []
        for t in pd.to_datetime(pd.Series(query)).dropna().values.astype("datetime64[s]").astype(np.int64):
            i = np.searchsorted(ref, t)
            d = sorted(abs(int(ref[j]) - int(t)) for j in (i - 1, i, i + 1) if 0 <= j < len(ref))
            if same and d and d[0] == 0:
                d = d[1:]
            out.append(d[0] / 86400 if d else np.nan)
        return np.array(out)

    tr = obs_tab[obs_tab.in_train]
    te = obs_tab[obs_tab.in_test & ~obs_tab.in_train]
    gap_train = nearest_gap_days(tr.ts, tr.ts, same=True)
    gap_test = nearest_gap_days(te.ts, tr.ts) if len(tr) else np.array([])
    print(f"\ntest observations that are ALSO annotated in train: {len(train_obs & test_obs):,}")
    for name, g in [("train obs -> nearest other train obs", gap_train), ("test obs -> nearest train obs", gap_test)]:
        if g.size:
            print(f"{name}: within 1 day {np.mean(g <= 1):.0%}, 7 days {np.mean(g <= 7):.0%}, "
                  f"27 days {np.mean(g <= 27):.0%} (median {np.nanmedian(g):.1f} days)")
    if len(ct_year) and len(ct_site):
        fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
        ct_year.plot.bar(ax=ax[0], width=.8)
        ax[0].set(title="Observations per year", xlabel="")
        ct_site.plot.bar(ax=ax[1], width=.8)
        ax[1].set(title="Observations per GONG site", xlabel="")
        fig.tight_layout()
        show(fig, "03_time_and_sites.png")

# %% 9 · Dark structures that are NOT labelled -------------------------------------------------
dark_rows, dark_cache = [], {}
geom = img_stats.set_index("obs")[["disk_cy", "disk_cx", "disk_r"]].to_dict("index") if len(img_stats) else {}
if HAVE_GT and geom:
    cand = [o for o in obs_list if o in best_path and o in geom and np.isfinite(geom[o]["disk_r"])]
    for obs in tqdm(rng.sample(cand, min(N_DARK_CHECK, len(cand))), desc="dark check"):
        img = load_gray(best_path[obs])
        gm = geom[obs]
        key = sets_by_obs[obs][0]
        insts = [r for r in (decode_crop(ANN[u], img.shape[0], img.shape[1]) for u in anns_by_set[key]) if r]
        gt = paint(insts, img.shape)
        dark = dark_candidates(img, gm["disk_cy"], gm["disk_cx"], gm["disk_r"])
        lab, n = ndi.label(dark, structure=np.ones((3, 3), bool))
        hit = np.unique(lab[gt & dark])
        unl = dark & ~np.isin(lab, hit[hit > 0])
        dark_rows.append(dict(obs=obs, dark_px=int(dark.sum()), dark_pieces=int(n), unlabelled_pieces=int(n - (hit > 0).sum()),
                              unlabelled_px=int(unl.sum()), unlabelled_frac=unl.sum() / max(dark.sum(), 1),
                              gt_px=int(gt.sum()), gt_flagged_dark=(gt & dark).sum() / max(gt.sum(), 1)))
dark_df = pd.DataFrame(dark_rows)
if len(dark_df):
    print(f"crude dark detector on {len(dark_df)} images (pixels >=15% darker than their surroundings):")
    print(f"  share of dark area NOT touching any labelled filament: median {dark_df.unlabelled_frac.median():.0%}")
    print(f"  unlabelled dark pieces per image: median {dark_df.unlabelled_pieces.median():.0f}")
    print(f"  share of labelled filament pixels the detector flags as dark: median {dark_df.gt_flagged_dark.median():.0%}")
    top = dark_df.sort_values("unlabelled_px", ascending=False).head(3)
    fig, ax = plt.subplots(1, len(top), figsize=(6 * len(top), 6), squeeze=False)
    for a, obs in zip(ax[0], top.obs):
        img = load_gray(best_path[obs])
        gm = geom[obs]
        key = sets_by_obs[obs][0]
        insts = [r for r in (decode_crop(ANN[u], img.shape[0], img.shape[1]) for u in anns_by_set[key]) if r]
        dark = dark_candidates(img, gm["disk_cy"], gm["disk_cx"], gm["disk_r"])
        lab, _ = ndi.label(dark, structure=np.ones((3, 3), bool))
        hit = np.unique(lab[paint(insts, img.shape) & dark])
        unl = dark & ~np.isin(lab, hit[hit > 0])
        rgb = to_rgb(img, (gm["disk_cy"], gm["disk_cx"], gm["disk_r"]))
        rgb[unl] = (0.35 * rgb[unl] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
        draw(rgb, insts, (40, 220, 80), 4)
        a.imshow(cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA))
        a.set(title=f"{obs}: green = labelled, red = dark but unlabelled")
        a.axis("off")
    fig.tight_layout()
    show(fig, "04_dark_but_unlabelled.png")

# %% 10 · Look at the labels: full disks, close-ups, and two annotators side by side -------------
readable = set(img_stats.obs) if len(img_stats) else set()
KEEP = {o: v for o, v in KEEP.items() if o in readable}
if KEEP:
    img_cache = {}

    def get_img(obs):
        if obs not in img_cache:
            img_cache[obs] = load_gray(best_path[obs])
        return img_cache[obs]

    def geom_of(obs):
        g = geom.get(obs)
        return (g["disk_cy"], g["disk_cx"], g["disk_r"]) if g else None

    shown = [o for o in sorted(KEEP, key=lambda o: -len(KEEP[o][next(iter(KEEP[o]))][1])) if o in best_path][:4]
    if shown:
        fig, ax = plt.subplots(1, len(shown), figsize=(6 * len(shown), 6), squeeze=False)
        for a, obs in zip(ax[0], shown):
            key = next(iter(KEEP[obs]))
            rgb = to_rgb(get_img(obs), geom_of(obs))
            for i, ins in enumerate(KEEP[obs][key][1]):
                draw(rgb, [ins], PALETTE[i % 10], 4)
            a.imshow(cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA))
            a.set(title=f"{obs} ({len(KEEP[obs][key][1])} filaments)")
            a.axis("off")
        fig.tight_layout()
        show(fig, "05_full_disk_labels.png")

    # close-ups: two filaments from each area quartile, raw next to outlined
    pool = [(o, k, i, ins[0].sum()) for o in KEEP if o in best_path for k in KEEP[o]
            for i, ins in enumerate(KEEP[o][k][1])]
    if pool:
        qs = np.quantile([p[3] for p in pool], [.25, .5, .75])
        picks = []
        for q in range(4):
            bucket = [p for p in pool if np.searchsorted(qs, p[3]) == q]
            picks += rng.sample(bucket, min(2, len(bucket)))
        fig, ax = plt.subplots(len(picks) // 2 + len(picks) % 2, 4, figsize=(16, 4.1 * (len(picks) // 2 + len(picks) % 2)),
                               squeeze=False)
        for n, (obs, key, i, area) in enumerate(picks):
            m, y0, x0 = KEEP[obs][key][1][i]
            img = get_img(obs)
            size = int(min(700, max(160, 1.3 * max(m.shape))))
            cy, cx = y0 + m.shape[0] // 2, x0 + m.shape[1] // 2
            Y0, X0 = max(0, cy - size // 2), max(0, cx - size // 2)
            Y1, X1 = min(img.shape[0], Y0 + size), min(img.shape[1], X0 + size)
            raw = to_rgb(img[Y0:Y1, X0:X1])
            outl = draw(raw.copy(), [(m, y0 - Y0, x0 - X0)], (255, 60, 60), 1)
            a1, a2 = ax[n // 2, (n % 2) * 2], ax[n // 2, (n % 2) * 2 + 1]
            a1.imshow(raw)
            a2.imshow(outl)
            a1.set_title(f"{obs} · {area:,} px", fontsize=9)
            a2.set_title("label outline", fontsize=9)
            a1.axis("off")
            a2.axis("off")
        for a in ax.ravel()[len(picks) * 2:]:
            a.axis("off")
        fig.tight_layout()
        show(fig, "06_closeups.png")

    multi_shown = [o for o in KEEP if len(KEEP[o]) > 1 and o in best_path][:2]
    if multi_shown:
        fig, ax = plt.subplots(len(multi_shown), 2, figsize=(13, 6.5 * len(multi_shown)), squeeze=False)
        for row, obs in zip(ax, multi_shown):
            k_a, k_b = list(KEEP[obs])[:2]
            ins_a, ins_b = KEEP[obs][k_a][1], KEEP[obs][k_b][1]
            rgb = to_rgb(get_img(obs), geom_of(obs))
            draw(rgb, ins_a, (40, 220, 80), 4)
            draw(rgb, ins_b, (230, 60, 230), 4)
            row[0].imshow(cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA))
            row[0].set(title=f"{obs}: green = {meta.at[k_a, 'annotator'] or 'set A'} ({len(ins_a)}), "
                             f"magenta = {meta.at[k_b, 'annotator'] or 'set B'} ({len(ins_b)})")
            m, y0, x0 = max(ins_a, key=lambda t: t[0].sum())
            pad = 80
            Y0, X0 = max(0, y0 - pad), max(0, x0 - pad)
            Y1, X1 = min(rgb.shape[0], y0 + m.shape[0] + pad), min(rgb.shape[1], x0 + m.shape[1] + pad)
            zoom = to_rgb(get_img(obs)[Y0:Y1, X0:X1])
            draw(zoom, [(mm, yy - Y0, xx - X0) for mm, yy, xx in ins_a], (40, 220, 80), 1)
            draw(zoom, [(mm, yy - Y0, xx - X0) for mm, yy, xx in ins_b], (230, 60, 230), 1)
            row[1].imshow(zoom)
            row[1].set(title="zoom on the largest filament")
            for a in row:
                a.axis("off")
        fig.tight_layout()
        show(fig, "07_two_annotators.png")

# %% 11 · SUMMARY (read this first) + save tables ---------------------------------------------
lines = []
if HAVE_GT:
    lines.append(f"Data: {len(anns):,} filaments in {int((sets.n_instances > 0).sum()):,} annotation sets over "
                 f"{sets.obs.nunique():,} observations; {len(test_obs):,} test observations; "
                 f"{int((sets.groupby('obs').size() > 1).sum()):,} observations labelled by more than one annotator.")
if len(inst):
    lines += [
        f"Width: typical {inst.width_med.median():.1f} px, thinnest parts ~{inst.width_p10.median():.1f} px (+-0.5); "
        f"median area {inst.area.median():,.0f} px ({(inst.area < 300).mean():.0%} under 300 px).",
        f"1-px errors: a uniform 1-px dilation leaves median IoU {inst.iou_dilate1.median():.2f} "
        f"({(inst.iou_dilate1 < .5).mean():.0%} of filaments drop below the 0.5 match line); "
        f"1-px erosion leaves {inst.iou_erode1.median():.2f} ({(inst.iou_erode1 < .5).mean():.0%} below).",
        f"Resolution: a perfect prediction at 1/2 res caps median IoU at {inst.iou_res_half.median():.2f} "
        f"({(inst.iou_res_half < .5).mean():.0%} below 0.5); at 1/4 res {inst.iou_res_quarter.median():.2f} "
        f"({(inst.iou_res_quarter < .5).mean():.0%} below 0.5).",
        f"Instance definition: {(inst.n_polygons > 1).mean():.1%} of annotations use >1 polygon, "
        f"{(inst.n_components > 1).mean():.1%} rasterise to >1 connected piece.",
        f"Neighbours: {(inst.gap_px < 1).mean():.1%} of filaments touch/overlap another; "
        f"{(inst.gap_px <= 10).mean():.1%} have one within 10 px."]
    if "r_norm" in inst:
        lines.append(f"Position: {(inst.r_norm > 0.9).mean():.1%} of filament centroids lie beyond 0.9 R (limb).")
if len(pairs):
    lines.append(f"Human vs human: PQ {pq_h:.3f} (SQ {sq_h:.3f}, RQ {rq_h:.3f}) over {len(pairs):,} annotator pairs; "
                 f"{pairs.splits.mean():.2f} splits and {pairs.merges.mean():.2f} merges per pair.")
if gap_train.size:
    lines.append(f"Time: {np.mean(gap_train <= 1):.0%} of train observations have another train observation within "
                 f"1 day, {np.mean(gap_train <= 7):.0%} within 7 days.")
if gap_test.size or train_obs & test_obs:
    lines.append(f"Train/test: {len(train_obs & test_obs):,} test observations are also annotated in train; "
                 + (f"{np.mean(gap_test <= 1):.0%} of the rest have a train observation within 1 day." if gap_test.size else ""))
if len(dark_df):
    lines.append(f"Unlabelled dark structures: median {dark_df.unlabelled_frac.median():.0%} of crude-detector dark area "
                 f"touches no label ({dark_df.unlabelled_pieces.median():.0f} pieces per image).")

print("=" * 30, "SUMMARY", "=" * 30)
print("\n".join("- " + s for s in lines))
print("\nRules of thumb:")
if len(inst):
    if (inst.iou_res_half < .5).mean() > .05:
        print("- Half resolution already pushes many filaments under the 0.5 match line: train and predict at full "
              "resolution (tiles), not on downsampled disks.")
    if (inst.n_components > 1).mean() < .02:
        print("- Each label is one connected piece, so connected components of a good binary mask map onto "
              "instances. Open question: were visually broken filaments bridged into one label or split into "
              "several? Check the close-ups and the two-annotator figure.")
    else:
        print("- Many labels have several pieces: plain connected components will over-split. Plan a fragment-merging step.")
    if (inst.gap_px <= 10).mean() > .2:
        print("- Many filaments sit close to a neighbour: aggressive dilation-based merging will over-merge.")
if len(pairs):
    print(f"- Treat ~{pq_h:.2f} PQ (human agreement) as the ceiling when judging your own CV score.")
if HAVE_GT and (sets.groupby("obs").size() > 1).any():
    print("- Several annotation sets share an image: build CV folds by observation name (better: by month), "
          "never by image id, or the same image lands in train and validation.")
if train_obs & test_obs:
    print("- Some test images also have train labels. Don't feed train labels into test predictions; the rules "
          "allow only test images at inference, and judges reproduce from code.")

for name, df in [("annotation_sets", sets), ("annotations", anns), ("filaments", inst), ("annotator_pairs", pairs),
                 ("image_stats", img_stats), ("observations", obs_tab), ("dark_check", dark_df)]:
    if len(df):
        df.to_csv(OUT / f"{name}.csv", index=False)
print(f"\nTables and figures saved to {OUT}")
