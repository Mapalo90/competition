"""Submission file: one row per predicted filament, COCO run-length encoding of its 2048 x 2048 mask."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .data import H, W


def rle_encode(mask: np.ndarray) -> str:
    """H x W {0,1} mask -> compressed COCO RLE counts string (no quotes, size implied 2048 x 2048)."""
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def rle_decode(counts: str, h: int = H, w: int = W) -> np.ndarray:
    return mask_utils.decode({"size": [h, w], "counts": counts.encode("ascii")})


def write_submission(pred: dict, path: str | Path, obs_order: list | None = None) -> pd.DataFrame:
    """pred: obs -> list of instances (mask, y0, x0). Rows are '<obs>_<k>' with k = 1, 2, ...
    Instances never overlap (they are distinct connected regions of one thresholded map)."""
    rows = []
    for obs in obs_order or sorted(pred):
        for k, (m, y0, x0) in enumerate(pred.get(obs, []), start=1):
            full = np.zeros((H, W), np.uint8)
            full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
            rows.append((f"{obs}_{k}", rle_encode(full)))
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    sub.to_csv(path, index=False)
    return sub


def read_submission(path: str | Path) -> dict:
    """Inverse of write_submission: obs -> list of full-resolution instances."""
    sub = pd.read_csv(path)
    out = {}
    for fid, counts in zip(sub.filament_id, sub.segmentation_rle):
        obs = str(fid).rsplit("_", 1)[0]
        m = rle_decode(str(counts)).astype(bool)
        ys, xs = np.nonzero(m)
        if ys.size:
            out.setdefault(obs, []).append((m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], int(ys.min()), int(xs.min())))
    return out
