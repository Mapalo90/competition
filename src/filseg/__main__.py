"""Command line: python -m filseg <command> [options]

  train     train fold models (weights, validation and test maps go to --out)
  oof       out-of-fold and test maps from trained fold models, by default the ones shipped in weights/
            (each model predicts the fold it never saw)
  tune      post-processing search on out-of-fold maps -> best settings JSON
  evaluate  PQ of out-of-fold maps under the configured post-processing
  submit    average the fold models' test maps, post-process, write submission.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _common(p):
    p.add_argument("--data", default="/kaggle/input", help="folder containing the competition data")
    p.add_argument("--config", default=None, help="JSON config (defaults reproduce the submission)")
    p.add_argument("--out", default="outputs", help="output folder")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="filseg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "oof", "tune", "evaluate", "submit"):
        p = sub.add_parser(name)
        _common(p)
        p.add_argument("--maps", nargs="*", default=None, help="folders with saved maps (default: --out)")
        p.add_argument("--weights", nargs="*", default=None,
                       help="folders with fold models (default: --out, then the repository's weights/ folder)")
        p.add_argument("--folds", nargs="*", type=int, default=None)
        p.add_argument("--objective", choices=["annotator", "pooled"], default="annotator")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--submission", default="submission.csv")
    a = ap.parse_args(argv)

    import torch
    from .config import Config, PostConfig
    from .data import discover
    from .metrics import panoptic_quality
    from .model import build_model, load_weights, predict_maps
    from .pipeline import load_maps, make_submission, prepare
    from .postprocess import instances_from_map
    from .tune import best_params, search
    from .weights import REPO_WEIGHTS, find_weights

    cfg = Config.from_json(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = discover(a.data)
    print(f"train images {len(data.train_obs)}, test images {len(data.test_obs)}")
    need_targets = a.cmd == "train"
    cache = prepare(data, cfg.train.img_size, n_folds=cfg.train.n_folds, seed=cfg.train.seed, with_targets=need_targets)
    folds = a.folds if a.folds is not None else list(range(cfg.train.n_folds))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.cmd == "train":
        from .train import train_fold
        for k in folds:
            val_obs = [o for o in data.train_obs if cache["folds"][o] == k]
            trn = [cache["idx"][o] for o in data.train_obs if cache["folds"][o] != k]
            train_fold(cfg.train, k, cache["X"], cache["Y"], trn, [cache["idx"][o] for o in val_obs], val_obs,
                       cache["gt"], cache["geom"], out, device, cache["test_small"], data.test_obs)
        return

    if a.cmd == "oof":
        weights = find_weights(a.weights or [out, REPO_WEIGHTS], cfg.train.n_folds)
        for k in folds:
            if k not in weights:
                print(f"fold {k}: no weights found"); continue
            model = build_model(cfg.train.encoder, None, device, cfg.train.channels_last)
            load_weights(model, weights[k], device)
            val_obs = [o for o in data.train_obs if cache["folds"][o] == k]
            vp = predict_maps(model, [cache["X"][cache["idx"][o]] for o in val_obs], device, tta=cfg.post.tta)
            tp = predict_maps(model, cache["test_small"], device, tta=cfg.post.tta)
            np.savez_compressed(out / f"val_probs_fold{k}.npz", obs=np.array(val_obs), probs=np.stack(vp))
            np.savez_compressed(out / f"test_probs_fold{k}.npz", obs=np.array(data.test_obs), probs=np.stack(tp))
            print(f"fold {k}: {len(vp)} validation and {len(tp)} test maps from {weights[k]}")
        return

    oof, oof_fold, test = load_maps(a.maps or [out])
    gt = {o: cache["gt"][o] for o in oof if o in cache["gt"]}
    print(f"out-of-fold maps for {len(gt)} images (folds {sorted(set(oof_fold.values()))}); test maps from folds {sorted(test)}")

    if a.cmd == "tune":
        tab = search(oof, gt, cache["geom"], workers=a.workers, folds=oof_fold)
        tab.to_csv(out / "tuning_grid.csv", index=False)
        best = best_params(tab, a.objective)
        (out / "best_post.json").write_text(json.dumps(best, indent=1))
        print(f"best for the {a.objective} objective: {best}")
        from .tune import nested_estimate
        if len(set(oof_fold.values())) > 1:
            est, _ = nested_estimate(tab, a.objective)
            print(f"nested estimate (settings chosen on the other folds): PQ {est:.4f}")
    elif a.cmd == "evaluate":
        pred = {o: instances_from_map(oof[o], cache["geom"][o], cfg.post) for o in gt}
        for rule, r in panoptic_quality(gt, pred).items():
            print(f"{rule:9s} PQ {r['PQ']:.4f}  SQ {r['SQ']:.4f}  RQ {r['RQ']:.4f}  TP {r['TP']}  FP {r['FP']}  FN {r['FN']}")
    elif a.cmd == "submit":
        sub = make_submission(test, data.test_obs, cache["geom"], cfg.post, out / a.submission)
        print(f"{len(sub)} filaments over {sub.filament_id.str.rsplit('_', n=1).str[0].nunique()} of "
              f"{len(data.test_obs)} test images -> {out / a.submission}")


if __name__ == "__main__":
    sys.exit(main())
