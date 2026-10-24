"""Post-processing search on out-of-fold maps (every training image scored by the model that never saw it)."""
from __future__ import annotations

import itertools
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from .metrics import match_counts, overlap_table, pq_parts
from .postprocess import components, keep_mask, upsample

DEFAULT_GRID = dict(merge=[["none", 0], ["group", 8], ["group", 12], ["group", 20], ["group", 30], ["group", 40]],
                    thr=[0.35, 0.4, 0.45, 0.5], min_area=[150, 200, 250, 300, 400], r_max=[0.9, 1.01],
                    min_score=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85])

_JOB = {}                      # filled before forking so worker processes inherit the data


def _image(i: int) -> tuple:
    j = _JOB
    obs = j["obs"][i]
    prob = upsample(j["maps"][obs])
    ann = np.zeros((len(j["cfgs"]), len(j["combos"]), 6))
    pool = np.zeros_like(ann)
    for ci, (merge, d, thr) in enumerate(j["cfgs"]):
        comps, feats, cents = components(prob, thr, merge, d, False)
        tables = [overlap_table(g, comps) for g in j["gt"][obs]]
        pooled = (np.vstack([t[0] for t in tables]), np.concatenate([t[1] for t in tables]), tables[0][2])
        for cj, (a, r, s) in enumerate(j["combos"]):
            keep = keep_mask(feats, cents, j["geom"][obs], dict(min_area=a, r_max=r, min_score=s))
            for t in tables:
                ann[ci, cj] += match_counts(*t, keep)
            pool[ci, cj] += match_counts(*pooled, keep)
    return i, ann, pool


def search(maps: dict, gt: dict, geom: dict, grid: dict = DEFAULT_GRID, workers: int = 4, folds: dict | None = None,
           log=print) -> pd.DataFrame:
    """Score every combination of (joining, threshold) x (min area, limb cut, min confidence).
    maps: obs -> model-resolution uint8 map; gt: obs -> annotator sets; geom: obs -> (cy, cx, R).
    Returns one row per combination with PQ/SQ/RQ under both rules (and per fold if `folds` is given)."""
    obs = [o for o in sorted(maps) if o in gt]
    cfgs = [(m, int(d), t) for (m, d) in grid["merge"] for t in grid["thr"]]
    combos = list(itertools.product(grid["min_area"], grid["r_max"], grid["min_score"]))
    _JOB.update(obs=obs, maps=maps, gt=gt, geom=geom, cfgs=cfgs, combos=combos)
    ANN = np.zeros((len(obs), len(cfgs), len(combos), 6)); POOL = np.zeros_like(ANN)
    t0 = time.time()
    try:
        if workers <= 1:
            raise RuntimeError("single process requested")
        with mp.get_context("fork").Pool(workers) as pool:
            for i, a, p in pool.imap_unordered(_image, range(len(obs)), chunksize=2):
                ANN[i], POOL[i] = a, p
    except Exception as e:
        log(f"  ({e}; running in one process)")
        for i in range(len(obs)):
            _, ANN[i], POOL[i] = _image(i)
    log(f"  search: {len(cfgs)} x {len(combos)} settings on {len(obs)} images in {time.time() - t0:.0f}s")
    rows = []
    fold_idx = {}
    if folds:
        for i, o in enumerate(obs):
            fold_idx.setdefault(folds[o], []).append(i)
    ta, tp_ = ANN.sum(0), POOL.sum(0)
    for ci, (merge, d, thr) in enumerate(cfgs):
        for cj, (a, r, s) in enumerate(combos):
            pq, sq, rq = pq_parts(*ta[ci, cj])
            pqp, sqp, rqp = pq_parts(*tp_[ci, cj])
            row = dict(merge=merge, d=d, thr=thr, min_area=a, r_max=r, min_score=s,
                       PQ_annotator=pq, SQ_annotator=sq, RQ_annotator=rq, PQ_pooled=pqp, SQ_pooled=sqp, RQ_pooled=rqp,
                       TP=ta[ci, cj][0], FP=ta[ci, cj][1], FN=ta[ci, cj][2], splits=ta[ci, cj][4], merges=ta[ci, cj][5],
                       preds_per_image=(tp_[ci, cj][0] + tp_[ci, cj][1]) / max(len(obs), 1))
            for k, idx in sorted(fold_idx.items()):
                row[f"PQ_annotator_fold{k}"] = pq_parts(*ANN[idx, ci, cj].sum(0))[0]
                row[f"PQ_pooled_fold{k}"] = pq_parts(*POOL[idx, ci, cj].sum(0))[0]
            rows.append(row)
    return pd.DataFrame(rows)


def best_params(table: pd.DataFrame, objective: str = "annotator") -> dict:
    """Best row for objective 'annotator' or 'pooled' as a PostConfig-style dict."""
    b = table.sort_values(f"PQ_{objective}", ascending=False).iloc[0]
    return dict(merge=str(b["merge"]), d=int(b.d), thr=float(b.thr), min_area=int(b.min_area), r_max=float(b.r_max),
                min_score=float(b.min_score))


def nested_estimate(table: pd.DataFrame, objective: str = "annotator") -> tuple:
    """Unbiased estimate of the tuned score: for each fold, choose settings on the other folds (mean of their
    per-fold PQ) and score them on the held-out fold. Needs the per-fold columns written by `search(folds=...)`.
    Returns (mean held-out PQ, {fold: (held-out PQ, chosen settings)})."""
    cols = sorted(c for c in table.columns if c.startswith(f"PQ_{objective}_fold"))
    if len(cols) < 2:
        raise ValueError("per-fold columns missing: run search(..., folds=...) on at least two folds")
    out = {}
    for c in cols:
        others = [o for o in cols if o != c]
        row = table.loc[table[others].mean(axis=1).idxmax()]
        k = int(c.rsplit("fold", 1)[1])
        out[k] = (float(row[c]), dict(merge=str(row["merge"]), d=int(row.d), thr=float(row.thr),
                                      min_area=int(row.min_area), r_max=float(row.r_max), min_score=float(row.min_score)))
    return float(np.mean([v[0] for v in out.values()])), out
