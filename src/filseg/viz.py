"""Figures: overlays on H-alpha images and the distributions the judges look at
(IoU / Dice of overlapping pairs, one-to-many and many-to-one relations)."""
from __future__ import annotations

from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .metrics import overlap_table

COLORS = {"gt": (40, 220, 80), "gt2": (60, 160, 255), "pred": (230, 60, 230)}


def to_rgb(img: np.ndarray, geom: tuple | None = None) -> np.ndarray:
    """Contrast-stretch the solar disk for display."""
    f = img.astype(np.float32)
    if geom is not None:
        yy, xx = np.ogrid[: f.shape[0], : f.shape[1]]
        v = f[(yy - geom[0]) ** 2 + (xx - geom[1]) ** 2 < (0.95 * geom[2]) ** 2]
    else:
        v = f[f > 20]
    lo, hi = np.percentile(v if v.size else f, [0.5, 99.5])
    g = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.ascontiguousarray((np.dstack([g] * 3) * 255).astype(np.uint8))


def draw(rgb: np.ndarray, insts: list, color: tuple, thickness: int = 2) -> np.ndarray:
    for m, y0, x0 in insts:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, [c + np.array([[[x0, y0]]], dtype=c.dtype) for c in cs], -1, color, thickness)
    return rgb


def overlay(img: np.ndarray, gt_sets: list, pred: list, geom: tuple | None = None, ax=None, title: str = "",
            zoom: tuple | None = None):
    """Annotator 1 green, annotator 2 blue, prediction magenta. `zoom` = (y0, x0, y1, x1) to crop."""
    rgb = to_rgb(img, geom)
    full = zoom is None
    t = 4 if full else 1
    for i, g in enumerate(gt_sets[:2]):
        draw(rgb, g, COLORS["gt"] if i == 0 else COLORS["gt2"], t)
    draw(rgb, pred, COLORS["pred"], t)
    if zoom is not None:
        y0, x0, y1, x1 = zoom
        rgb = rgb[y0:y1, x0:x1]
    elif rgb.shape[0] > 1024:
        rgb = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA)
    ax = ax or plt.gca()
    ax.imshow(rgb)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return ax


def pair_stats(gt: dict, pred: dict) -> dict:
    """Per annotator set: IoU and Dice of every overlapping (GT, prediction) pair, and for each GT filament
    the number of predictions it overlaps (1:n) / for each prediction the number of GT filaments (n:1)."""
    ious, dices, gt_deg, pred_deg = [], [], [], []
    for obs, sets in gt.items():
        p = pred.get(obs, [])
        for g in sets:
            inter, ag, ap = overlap_table(g, p)
            if len(ag) and len(ap):
                union = ag[:, None] + ap[None, :] - inter
                hit = inter > 0
                ious += list((inter / union)[hit])
                dices += list((2 * inter / (ag[:, None] + ap[None, :]))[hit])
                gt_deg += list(hit.sum(1)); pred_deg += list(hit.sum(0))
            else:
                gt_deg += [0] * len(ag); pred_deg += [0] * len(ap)
    return dict(iou=np.array(ious), dice=np.array(dices), gt_degree=Counter(gt_deg), pred_degree=Counter(pred_deg))


def distribution_figure(stats: dict, title: str = ""):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a, key, name in ((ax[0], "iou", "IoU"), (ax[1], "dice", "Dice")):
        v = stats[key]
        a.hist(v, bins=np.linspace(0, 1, 51), color="navy", alpha=0.75, edgecolor="black")
        if v.size:
            a.axvline(v.mean(), color="crimson", ls="--", label=f"mean {v.mean():.3f}")
            a.legend(fontsize=8)
        a.set(title=f"{name} of overlapping pairs", xlabel=name)
    g, p = stats["gt_degree"], stats["pred_degree"]
    deg = range(0, max(list(g) + list(p) + [1]) + 1)
    ax[2].bar([d + 0.2 for d in deg], [g.get(d, 0) for d in deg], width=0.4, label="GT filament: predictions overlapping (1:n)")
    ax[2].bar([d - 0.2 for d in deg], [p.get(d, 0) for d in deg], width=0.4, label="prediction: GT filaments overlapping (n:1)")
    ax[2].set(title="One-to-many and many-to-one", xlabel="overlapping partners", xticks=list(deg))
    ax[2].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
