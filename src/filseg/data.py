"""Finding and reading the competition data: images, annotations, folds and training targets.

Terminology used throughout:
  obs             observation name from the GONG file name, e.g. "20140609195854Bh"
  annotation set  the filaments one annotator drew on one image; MAGFiLO labels 42% of training images
                  two or three times, so an image can have several sets
  instance        one filament as (bool mask crop, y0, x0) in full-resolution 2048 x 2048 coordinates
"""
from __future__ import annotations

import collections
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

H = W = 2048
OBS_RE = re.compile(r"(\d{14})([A-Za-z])h")
IMG_EXTS = (".fits.fz", ".fits", ".fts", ".jpg", ".jpeg", ".png", ".tif", ".tiff")   # preference order

Instance = tuple  # (np.ndarray[bool], int, int)


def obs_id(name: str) -> str | None:
    """'010401-20140609195854Bh.jpeg' -> '20140609195854Bh' (None if no GONG name inside)."""
    m = OBS_RE.search(str(name))
    return m.group(0) if m else None


def obs_time(obs: str) -> datetime | None:
    m = OBS_RE.search(obs)
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None


def ext_of(p: Path) -> str:
    return ".fits.fz" if p.name.lower().endswith(".fits.fz") else p.suffix.lower()


def split_of(path: Path, root: Path) -> str:
    """'train', 'val', 'test' or 'unknown', from the folder and file names below `root`."""
    text = " ".join(t.lower() for t in Path(path).relative_to(root).parts)
    for name in ("test", "val", "train"):
        if re.search(rf"(?<![a-z]){name}", text):
            return name
    return "unknown"


def load_gray(path: str | Path) -> np.ndarray:
    """Read an H-alpha image as uint8 grayscale (JPEG/PNG/TIFF, or FITS rescaled to 0-255)."""
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


@dataclass
class CompetitionData:
    root: Path
    image_path: dict = field(default_factory=dict)        # obs -> preferred image file
    annotation_sets: dict = field(default_factory=dict)   # obs -> [ [COCO annotation, ...] per annotator ]
    train_obs: list = field(default_factory=list)         # annotated observations that have an image
    test_obs: list = field(default_factory=list)


def discover(root: str | Path = "/kaggle/input") -> CompetitionData:
    """Index every image and COCO annotation file under `root` (the order of sets follows the JSON)."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    data = CompetitionData(root=root)
    best_rank = {}
    for p in files:
        e, o = ext_of(p), obs_id(p.name)
        if e in IMG_EXTS and o and not re.search(r"mask|label|annot", str(p.parent.relative_to(root)).lower()):
            rank = IMG_EXTS.index(e)
            if rank < best_rank.get(o, 99):
                best_rank[o], data.image_path[o] = rank, p
            if split_of(p, root) == "test" and o not in data.test_obs:
                data.test_obs.append(o)
    sets = collections.defaultdict(list)
    for p in files:
        if ext_of(p) != ".json":
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not (isinstance(d, dict) and d.get("annotations") and split_of(p, root) != "test"):
            continue
        by_img = collections.defaultdict(list)
        for a in d["annotations"]:
            by_img[a["image_id"]].append(a)
        for im in d["images"]:
            o = obs_id(im.get("file_name") or im["id"])
            if o:
                sets[o].append(by_img.get(im["id"], []))
    data.annotation_sets = dict(sets)
    data.train_obs = sorted(o for o in sets if o in data.image_path)
    data.test_obs = sorted(data.test_obs)
    return data


def decode_annotation(a: dict, h: int = H, w: int = W, pad: int = 2) -> Instance | None:
    """One COCO annotation (polygon or RLE) -> tight (bool crop, y0, x0); identical pixels to annToMask."""
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


def instances_of(annotation_sets: list, h: int = H, w: int = W) -> list:
    """Annotation sets of one image -> one list of instances per annotator."""
    return [[r for r in (decode_annotation(a, h, w) for a in anns) if r] for anns in annotation_sets]


def paint(insts: list, shape: tuple) -> np.ndarray:
    canvas = np.zeros(shape, bool)
    for m, y0, x0 in insts:
        canvas[y0: y0 + m.shape[0], x0: x0 + m.shape[1]] |= m
    return canvas


def soft_target(gt_sets: list, shape: tuple = (H, W)) -> np.ndarray:
    """Training target: fraction of annotators who marked each pixel (float32 in [0, 1])."""
    acc = np.zeros(shape, np.float32)
    for insts in gt_sets:
        acc += paint(insts, shape)
    return acc / max(len(gt_sets), 1)


def make_folds(train_obs: list, n_folds: int = 5, seed: int = 42) -> dict:
    """Folds over images (all annotators of an image together), stratified by year when every year
    has at least n_folds images; the same recipe produced the folds used for the submitted models."""
    from sklearn.model_selection import KFold, StratifiedKFold
    train_obs = sorted(train_obs)
    years = np.array([obs_time(o).year if obs_time(o) else 0 for o in train_obs])
    if min(collections.Counter(years).values()) >= n_folds:
        split = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(train_obs, years)
    else:
        split = KFold(n_folds, shuffle=True, random_state=seed).split(train_obs)
    fold_of = {}
    for k, (_, va) in enumerate(split):
        for i in va:
            fold_of[train_obs[i]] = k
    return fold_of


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def resize_target(target: np.ndarray, size: int) -> np.ndarray:
    """Soft target -> uint8 (0-255) at model resolution."""
    return np.round(cv2.resize(target, (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
