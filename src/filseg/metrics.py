"""Panoptic Quality (Kirillov et al., CVPR 2019) for filament instances.

    PQ = sum(IoU of matched pairs) / (TP + FP/2 + FN/2) = SQ x RQ,  a match needs IoU > 0.5

Images annotated by several people can be scored two ways, both reported throughout:
  "annotator": against each annotator's set separately, counts summed over all sets
               (the organizers' self-evaluation notebook gives the same result for non-overlapping masks)
  "pooled":    against all annotators' filaments of an image in one list, one-to-one matching
               (this is the variant whose level matches the public leaderboard)
"""
from __future__ import annotations

import numpy as np


def _inter(a: tuple, b: tuple) -> int:
    (ma, ya, xa), (mb, yb, xb) = a, b
    y0, x0 = max(ya, yb), max(xa, xb)
    y1, x1 = min(ya + ma.shape[0], yb + mb.shape[0]), min(xa + ma.shape[1], xb + mb.shape[1])
    if y0 >= y1 or x0 >= x1:
        return 0
    return int(np.logical_and(ma[y0 - ya: y1 - ya, x0 - xa: x1 - xa], mb[y0 - yb: y1 - yb, x0 - xb: x1 - xb]).sum())


def overlap_table(gt: list, pred: list) -> tuple:
    """Intersections (n_gt, n_pred) and areas; only pairs whose bounding boxes overlap are intersected."""
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


def match_counts(inter: np.ndarray, ag: np.ndarray, ap: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """One-to-one matching at IoU > 0.5 (greedy by IoU, equivalent to optimal at this threshold).
    Returns [TP, FP, FN, sum of matched IoU, splits, merges]: a split is a GT filament with 2+ predictions
    mostly inside it (one-to-many), a merge a prediction with 2+ GT filaments mostly inside it."""
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


def pq_parts(tp: float, fp: float, fn: float, sum_iou: float, *_) -> tuple:
    """(PQ, SQ, RQ) from counts."""
    den = tp + 0.5 * fp + 0.5 * fn
    return (sum_iou / den if den else np.nan, sum_iou / tp if tp else np.nan, tp / den if den else np.nan)


def bootstrap_interval(counts: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple:
    """Percentile bootstrap interval of PQ over images. counts: one row per image, [TP, FP, FN, sum of IoU, ...]."""
    counts = np.asarray(counts, float)[:, :4]
    idx = np.random.default_rng(seed).integers(0, len(counts), (n_boot, len(counts)))
    s = counts[idx].sum(1)
    den = s[:, 0] + 0.5 * s[:, 1] + 0.5 * s[:, 2]
    pq = np.divide(s[:, 3], den, out=np.full(n_boot, np.nan), where=den > 0)
    tail = 50 * (1 - level)
    return tuple(float(v) for v in np.nanpercentile(pq, [tail, 100 - tail]))


def image_counts(gt_sets: list, pred: list, keep: np.ndarray | None = None) -> dict:
    """Counts for one image under both rules: {'annotator': [...6], 'pooled': [...6]}."""
    tables = [overlap_table(g, pred) for g in gt_sets]
    ann = np.zeros(6)
    for t in tables:
        ann += match_counts(*t, keep)
    if tables:
        pooled = (np.vstack([t[0] for t in tables]), np.concatenate([t[1] for t in tables]), tables[0][2])
    else:
        pooled = (np.zeros((0, len(pred))), np.zeros(0), np.array([m.sum() for m, _, _ in pred], float))
    return {"annotator": ann, "pooled": match_counts(*pooled, keep)}


def panoptic_quality(gt: dict, pred: dict) -> dict:
    """gt: obs -> list of annotator sets of instances; pred: obs -> list of instances.
    Returns {rule: {'PQ', 'SQ', 'RQ', 'TP', 'FP', 'FN', 'splits', 'merges'}} for both rules."""
    tot = {"annotator": np.zeros(6), "pooled": np.zeros(6)}
    for obs, gt_sets in gt.items():
        c = image_counts(gt_sets, pred.get(obs, []))
        for k in tot:
            tot[k] += c[k]
    out = {}
    for k, c in tot.items():
        pq, sq, rq = pq_parts(*c)
        out[k] = dict(PQ=pq, SQ=sq, RQ=rq, TP=int(c[0]), FP=int(c[1]), FN=int(c[2]), splits=int(c[4]), merges=int(c[5]))
    return out
