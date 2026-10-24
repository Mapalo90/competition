"""Probability map -> filament instances.

Steps: upsample to 2048, threshold, join fragments closer than `d` pixels into one filament, then keep
only filaments that are large enough, confident enough (mean probability) and not near the limb."""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi

from .config import PostConfig
from .data import H, W


def upsample(prob_u8: np.ndarray, size: tuple = (H, W)) -> np.ndarray:
    """Model-resolution uint8 probability map -> full resolution (bilinear)."""
    return cv2.resize(prob_u8, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def average_maps(maps: list) -> np.ndarray:
    """Average uint8 probability maps of several models (resized to the largest if needed)."""
    size = max(m.shape[0] for m in maps)
    maps = [m if m.shape[0] == size else cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR) for m in maps]
    return np.round(np.mean(maps, axis=0)).astype(np.uint8)


def components(prob_u8: np.ndarray, thr: float, merge: str = "none", d: int = 0, fill: bool = False,
               min_pix: int = 10) -> tuple:
    """Candidate filaments from a full-resolution uint8 probability map.

    merge="group": pieces whose gap is at most d px count as one filament (their pixels are unchanged,
                   so a filament may consist of several pieces, as annotators often draw them).
    merge="close": morphological closing with a d-px disk first, so the gaps themselves are painted in.
    Specks under `min_pix` pixels are ignored.
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
    for j in ids:
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


def keep_mask(feats: np.ndarray, cents: np.ndarray, geom: tuple, p) -> np.ndarray:
    """Which candidates pass the area, limb and confidence filters. `p` is a PostConfig or dict."""
    get = (lambda k: p[k]) if isinstance(p, dict) else (lambda k: getattr(p, k))
    if not len(feats):
        return np.zeros(0, bool)
    cy, cx, R = geom
    rr = np.hypot(cents[:, 1] - cy, cents[:, 0] - cx) / R
    return (feats[:, 0] >= get("min_area")) & (rr <= get("r_max")) & (feats[:, 1] >= get("min_score"))


def instances_from_map(prob_small: np.ndarray, geom: tuple, p: PostConfig) -> list:
    """Full post-processing for one image: model-resolution map -> kept instances (mask, y0, x0)."""
    comps, feats, cents = components(upsample(prob_small), p.thr, p.merge, p.d, p.fill)
    return [c for c, k in zip(comps, keep_mask(feats, cents, geom, p)) if k]
