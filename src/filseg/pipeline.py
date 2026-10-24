"""High-level steps shared by the command line and the notebook."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, PostConfig
from .data import CompetitionData, discover, instances_of, load_gray, make_folds, resize_image, resize_target, soft_target
from .geometry import disk_geometry
from .postprocess import average_maps, instances_from_map
from .submission import write_submission


def prepare(data: CompetitionData, size: int, folds: dict | None = None, n_folds: int = 5, seed: int = 42,
            with_targets: bool = True, progress=lambda it, **_: it) -> dict:
    """Cache everything training and evaluation need: model-resolution images and soft targets, GT instances,
    disk geometry for every image, and the folds."""
    folds = folds or make_folds(data.train_obs, n_folds, seed)
    X = np.zeros((len(data.train_obs), size, size), np.uint8)
    Y = np.zeros_like(X)
    gt, geom = {}, {}
    for i, o in enumerate(progress(data.train_obs, desc="train images")):
        img = load_gray(data.image_path[o])
        X[i] = resize_image(img, size)
        gt[o] = instances_of(data.annotation_sets[o], *img.shape)
        if with_targets:
            Y[i] = resize_target(soft_target(gt[o], img.shape), size)
        geom[o] = disk_geometry(img)
    test_small = []
    for o in progress(data.test_obs, desc="test images"):
        img = load_gray(data.image_path[o])
        test_small.append(resize_image(img, size))
        geom[o] = disk_geometry(img)
    return dict(X=X, Y=Y, gt=gt, geom=geom, folds=folds, test_small=test_small,
                idx={o: i for i, o in enumerate(data.train_obs)})


def load_maps(dirs: list) -> tuple:
    """Collect saved maps: val_probs_fold{k}.npz (out-of-fold) and test_probs_fold{k}.npz / test_probs.npz.
    Returns (oof: obs -> map, oof_fold: obs -> k, test: k -> {obs: map}). Earlier copies of a fold win."""
    oof, oof_fold, test = {}, {}, {}
    for d in dirs:
        for p in sorted(Path(d).rglob("val_probs_fold*.npz")):
            m = re.fullmatch(r"val_probs_fold(\d+)\.npz", p.name)
            if not m or int(m.group(1)) in set(oof_fold.values()):
                continue
            k = int(m.group(1))
            z = np.load(p)
            for o, pr in zip(z["obs"], z["probs"]):
                oof[str(o)], oof_fold[str(o)] = pr, k
            tp = p.with_name(f"test_probs_fold{k}.npz")
            if not tp.exists() and (p.parent / "test_probs.npz").exists():
                tp = p.parent / "test_probs.npz"
            if tp.exists():
                zt = np.load(tp)
                test[k] = {str(o): pr for o, pr in zip(zt["obs"], zt["probs"])}
    return oof, oof_fold, test


def predict_test(test: dict, test_obs: list, geom: dict, post: PostConfig) -> dict:
    """Average the fold models' maps per test image, then post-process."""
    out = {}
    for o in test_obs:
        maps = [test[k][o] for k in sorted(test) if o in test[k]]
        out[o] = instances_from_map(average_maps(maps), geom[o], post) if maps else []
    return out


def make_submission(test: dict, test_obs: list, geom: dict, post: PostConfig, path: str | Path) -> pd.DataFrame:
    return write_submission(predict_test(test, test_obs, geom, post), path, test_obs)
