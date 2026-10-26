# %% [markdown]
# # Solar filament segmentation: the full pipeline
#
# This notebook reproduces our submission to the Solar Filament Segmentation Challenge 2026 from the
# competition data, using the `filseg` package in this repository.
#
# **Approach in one paragraph.** Each 2048 x 2048 GONG H-alpha image is downsampled to 1024 x 1024 and
# segmented as a whole disk by a U-Net (ResNet34 encoder, ImageNet weights). The training target is the
# *average* of the image's annotators (42% of training images were labelled by two or three people),
# so the network learns how likely each pixel is to be labelled. Five models, one per fold, are averaged
# with flip test-time augmentation. Instances are then formed at full resolution: threshold, join
# fragments closer than 20 px into one filament, and keep filaments that are at least 300 px, have mean
# probability at least 0.85 and lie within 0.9 solar radii. These settings were tuned for Panoptic
# Quality on out-of-fold predictions of all 707 training images.
#
# **Run modes** (set `MODE` below). Only the competition data is needed: the five trained models are in this
# repository's `weights/` folder.
# | MODE | time on Kaggle | what it does |
# |---|---|---|
# | `"weights"` (default) | ~30 min, GPU T4 x2 | load the five fold models, recompute all probability maps, evaluate, write the submission |
# | `"train"` | ~7 h, GPU T4 x2 | train the five fold models from scratch, then as above |
# | `"maps"` | ~10 min, CPU | start from probability maps saved by an earlier run (attached), then as above |
#
# On Kaggle: create the notebook from the competition page (the data is attached automatically), turn
# Internet on (to fetch this repository and segmentation-models-pytorch) and run all cells. The last
# section checks the new submission against the file we submitted.

# %% 0 · Settings -------------------------------------------------------------------------------
import os
MODE = os.environ.get("FILSEG_MODE", "weights")     # "weights", "train" or "maps"
REPO_URL = "https://github.com/Mapalo90/competition.git"
WEIGHTS_DIR = os.environ.get("FILSEG_WEIGHTS")      # None: the repository's weights/ folder
DATA_DIR = os.environ.get("FILSEG_DATA", "/kaggle/input")
OUT_DIR = os.environ.get("FILSEG_OUT", "/kaggle/working/outputs")
TUNE = os.environ.get("FILSEG_TUNE", "0") == "1"   # re-run the post-processing search (~10 min)
FIGURES = True

# %% 1 · Install and import ------------------------------------------------------------------------
import sys, subprocess, time, json
from pathlib import Path

def _have_filseg():
    try:
        import filseg  # noqa: F401
        return True
    except ImportError:
        return False

_here = [str(Path(__file__).resolve().parents[1] / "src")] if "__file__" in globals() else []
# the code can also come from an attached Kaggle dataset holding this repository (while it is private)
_attached = [str(p.parent.parent) for p in Path(DATA_DIR).glob("**/src/filseg/__init__.py")] if Path(DATA_DIR).exists() else []
for candidate in ["../src", "src", "/kaggle/working/filament-segmentation/src"] + _here + _attached:
    if Path(candidate, "filseg").exists() and candidate not in sys.path:
        sys.path.insert(0, str(Path(candidate).resolve()))
if not _have_filseg():
    subprocess.run(["git", "clone", "-q", REPO_URL, "/kaggle/working/filament-segmentation"], check=True)
    sys.path.insert(0, "/kaggle/working/filament-segmentation/src")
for pkg, pip_name in (("segmentation_models_pytorch", "segmentation-models-pytorch==0.5.0"), ("pycocotools", "pycocotools")):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from filseg import __version__
from filseg.config import Config
from filseg.data import discover, load_gray
from filseg.pipeline import prepare, load_maps, predict_test, make_submission
from filseg.postprocess import instances_from_map
from filseg.metrics import panoptic_quality
from filseg.model import build_model, load_weights, predict_maps
from filseg.weights import REPO_WEIGHTS, find_weights
from filseg.tune import search, best_params
from filseg import viz

cfg = Config.from_json(os.environ.get("FILSEG_CONFIG") or None,            # defaults reproduce the submission
                       overrides=json.loads(os.environ.get("FILSEG_OVERRIDES", "{}")))
out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"filseg {__version__} | torch {torch.__version__} | device {device} x {max(torch.cuda.device_count(), 1)} | mode {MODE}")
print(json.dumps({"train": vars(cfg.train), "post": vars(cfg.post)}, indent=1, default=str))

# %% [markdown]
# ## 1. Data
# Every image and annotation file under `DATA_DIR` is indexed. Folds are formed over images, keeping all
# annotators of an image together and stratifying by year; this recipe yields exactly the folds the
# submitted models were trained on. Images are cached at 1024 px with their soft targets, and the solar
# disk is located on each full-resolution image for the limb filter.

# %% 2 · Discover and cache ----------------------------------------------------------------------
t0 = time.time()
data = discover(DATA_DIR)
cache = prepare(data, cfg.train.img_size, n_folds=cfg.train.n_folds, seed=cfg.train.seed, with_targets=(MODE == "train"))
folds = cache["folds"]
n_sets = sum(len(data.annotation_sets[o]) for o in data.train_obs)
print(f"{len(data.train_obs)} training images ({n_sets} annotation sets, "
      f"{sum(len(a) for o in data.train_obs for a in data.annotation_sets[o]):,} filaments), "
      f"{len(data.test_obs)} test images; cached in {time.time() - t0:.0f}s")
print("images per fold:", pd.Series(folds).value_counts().sort_index().to_dict())

# %% 3 · Training, or finding the trained models ----------------------------------------------------
fold_dir = out / "folds"
if MODE == "train":
    from filseg.train import train_fold
    for k in range(cfg.train.n_folds):
        val_obs = [o for o in data.train_obs if folds[o] == k]
        trn = [cache["idx"][o] for o in data.train_obs if folds[o] != k]
        train_fold(cfg.train, k, cache["X"], cache["Y"], trn, [cache["idx"][o] for o in val_obs], val_obs,
                   cache["gt"], cache["geom"], fold_dir, device, cache["test_small"], data.test_obs)
elif MODE == "weights":
    weights = find_weights([WEIGHTS_DIR] if WEIGHTS_DIR else None, cfg.train.n_folds)
    print("weights:", {k: str(v) for k, v in weights.items()})
    assert len(weights) == cfg.train.n_folds, f"expected {cfg.train.n_folds} fold models in {WEIGHTS_DIR or REPO_WEIGHTS}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    t_inf, n_inf = 0.0, 0
    for k, w in weights.items():
        model = build_model(cfg.train.encoder, None, device, cfg.train.channels_last)
        load_weights(model, w, device)
        val_obs = [o for o in data.train_obs if folds[o] == k]
        vp = predict_maps(model, [cache["X"][cache["idx"][o]] for o in val_obs], device, tta=cfg.post.tta)
        t1 = time.time()
        tp = predict_maps(model, cache["test_small"], device, tta=cfg.post.tta)
        t_inf += time.time() - t1; n_inf += len(tp)
        np.savez_compressed(fold_dir / f"val_probs_fold{k}.npz", obs=np.array(val_obs), probs=np.stack(vp))
        np.savez_compressed(fold_dir / f"test_probs_fold{k}.npz", obs=np.array(data.test_obs), probs=np.stack(tp))
        print(f"fold {k}: maps for {len(vp)} validation and {len(tp)} test images")
    print(f"network inference: {t_inf / max(n_inf, 1) * cfg.train.n_folds:.2f} s per test image for the 5-model ensemble with TTA")

maps_dirs = [fold_dir] if MODE in ("train", "weights") else [DATA_DIR]
oof, oof_fold, test = load_maps(maps_dirs)
gt = {o: cache["gt"][o] for o in oof if o in cache["gt"]}
print(f"out-of-fold maps for {len(gt)} of {len(data.train_obs)} training images (folds {sorted(set(oof_fold.values()))}); "
      f"test maps from {len(test)} fold models")

# %% [markdown]
# ## 2. Out-of-fold evaluation
# Every training image is scored with the model that never saw it. PQ is reported against each annotator
# separately (the organizers' self-evaluation notebook) and against all annotators pooled with one-to-one
# matching (whose level matches the public leaderboard), with 95% bootstrap intervals over images. Human
# agreement on the same images, one annotator scored against another, is the reference point. Per-image
# counts and a summary go to `per_image_counts.csv` and `results.json`.

# %% 4 · Evaluate -----------------------------------------------------------------------------
from filseg.metrics import overlap_table, match_counts, pq_parts, image_counts, bootstrap_interval
pred_oof = {o: instances_from_map(oof[o], cache["geom"][o], cfg.post) for o in gt}
res = panoptic_quality(gt, pred_oof)
per_image = {o: image_counts(gt[o], pred_oof[o]) for o in gt}
for rule, r in res.items():
    r["PQ_95"] = bootstrap_interval([per_image[o][rule] for o in gt])
    extra = f"  splits {r['splits']}  merges {r['merges']}" if rule == "annotator" else ""
    print(f"{rule:9s} PQ {r['PQ']:.4f} (95% interval over images {r['PQ_95'][0]:.4f}-{r['PQ_95'][1]:.4f})  "
          f"SQ {r['SQ']:.4f}  RQ {r['RQ']:.4f}  TP {r['TP']}  FP {r['FP']}  FN {r['FN']}{extra}")
per_fold = {k: panoptic_quality({o: gt[o] for o in gt if oof_fold[o] == k}, pred_oof) for k in sorted(set(oof_fold.values()))}
fold_pq = np.array([v["annotator"]["PQ"] for v in per_fold.values()])
print("per fold (annotator rule):", {k: round(float(v["annotator"]["PQ"]), 4) for k, v in per_fold.items()},
      f"mean {fold_pq.mean():.4f}, s.d. {fold_pq.std(ddof=1):.4f}")
human = {}                                   # every pair of annotators of an image, one scored against the other
for o in gt:
    pairs = [match_counts(*overlap_table(gt[o][i], gt[o][j])) for i in range(len(gt[o])) for j in range(i + 1, len(gt[o]))]
    if pairs:
        human[o] = np.sum(pairs, axis=0)
hum_pq, hum_ci = pq_parts(*np.sum(list(human.values()), axis=0))[0], bootstrap_interval(list(human.values()))
print(f"human vs human on the same images: PQ {hum_pq:.4f} (95% interval {hum_ci[0]:.4f}-{hum_ci[1]:.4f}, {len(human)} images)")
names = ["TP", "FP", "FN", "sumIoU", "splits", "merges"]
pd.DataFrame([dict(obs=o, fold=oof_fold[o], annotators=len(gt[o]),
                   **{f"{rule}_{n}": per_image[o][rule][i] for rule in ("annotator", "pooled") for i, n in enumerate(names)},
                   **({f"human_{n}": human[o][i] for i, n in enumerate(names)} if o in human else {})) for o in gt]
             ).to_csv(out / "per_image_counts.csv", index=False)
json.dump({"oof": res, "per_fold": {k: v["annotator"]["PQ"] for k, v in per_fold.items()},
           "human": {"PQ": hum_pq, "PQ_95": hum_ci, "images": len(human)}, "images": len(gt), "post": vars(cfg.post)},
          open(out / "results.json", "w"), indent=1, default=float)

# %% 5 · Optional: post-processing search -----------------------------------------------------------
if TUNE:
    tab = search(oof, gt, cache["geom"], workers=os.cpu_count() or 2, folds=oof_fold)
    tab.to_csv(out / "tuning_grid.csv", index=False)
    print(tab.sort_values("PQ_annotator", ascending=False).head(5).round(4).to_string(index=False))
    print("best (annotator rule):", best_params(tab, "annotator"))
    print("best (pooled rule):   ", best_params(tab, "pooled"))
    from filseg.tune import nested_estimate
    for rule in ("annotator", "pooled"):
        est, per = nested_estimate(tab, rule)
        print(f"nested estimate ({rule}): settings chosen on four folds, scored on the fifth -> PQ {est:.4f}")

# %% [markdown]
# ## 3. Test predictions and submission
# The five fold models' test maps are averaged, post-processed with the configured settings and written
# as one row per filament with its COCO run-length encoding.

# %% 6 · Submission -----------------------------------------------------------------------------
t1 = time.time()
sub = make_submission(test, data.test_obs, cache["geom"], cfg.post, out / "submission.csv")
dt = (time.time() - t1) / max(len(data.test_obs), 1)
print(f"{len(sub)} filaments over {sub.filament_id.str.rsplit('_', n=1).str[0].nunique()} of {len(data.test_obs)} "
      f"test images -> {out / 'submission.csv'}; post-processing {dt:.2f} s per image")
if Path("/kaggle/working").exists():
    sub.to_csv("/kaggle/working/submission.csv", index=False)

# %% [markdown]
# ## 4. Check against the submitted file
# `submissions/submission_kaggle.csv` is the file we submitted (public leaderboard PQ 0.36). The new submission
# is scored against it with the competition metric, so PQ 1.0 would mean the same filaments with the same
# pixels. Mixed-precision GPU inference differs slightly between GPUs and library versions, and the models
# are stored in float16, so a value just below 1 is expected.

# %% 7 · Reproduction check -------------------------------------------------------------------------
submitted = REPO_WEIGHTS.parent / "submissions" / "submission_kaggle.csv"
if submitted.exists():
    from filseg.submission import read_submission
    ref, new = read_submission(submitted), read_submission(out / "submission.csv")
    c = sum(match_counts(*overlap_table(ref.get(o, []), new.get(o, []))) for o in data.test_obs)
    pq, sq, _ = pq_parts(*c)
    print(f"new vs submitted file: {int(c[0])} filaments matched, {int(c[1])} only in the new file, "
          f"{int(c[2])} only in the submitted file; PQ {pq:.4f}, mean IoU of matched filaments {sq:.4f}")

# %% [markdown]
# ## 5. Figures
# Overlays of out-of-fold predictions on validation images (annotator 1 green, annotator 2 blue,
# prediction magenta), and the distributions the evaluation rubric names: IoU and Dice of overlapping
# GT/prediction pairs, and one-to-many / many-to-one relations.

# %% 8 · Figures --------------------------------------------------------------------------------
if FIGURES:
    multi = sorted((o for o in gt if len(gt[o]) >= 2), key=lambda o: -sum(len(s) for s in gt[o]))[:3]
    fig, axes = plt.subplots(1, len(multi), figsize=(6 * len(multi), 6), squeeze=False)
    for ax, o in zip(axes[0], multi):
        viz.overlay(load_gray(data.image_path[o]), gt[o], pred_oof[o], cache["geom"][o], ax,
                    f"{o}: {len(gt[o])} annotators, {len(pred_oof[o])} predicted")
    fig.tight_layout(); fig.savefig(out / "fig_overlays.png", dpi=110); plt.show()
    stats = viz.pair_stats(gt, pred_oof)
    np.savez_compressed(out / "pair_stats.npz", iou=stats["iou"], dice=stats["dice"],
                        gt_degree=np.array(sorted(stats["gt_degree"].items())),
                        pred_degree=np.array(sorted(stats["pred_degree"].items())))
    fig = viz.distribution_figure(stats, "Out-of-fold predictions, all training images")
    fig.savefig(out / "fig_distributions.png", dpi=110); fig.savefig(out / "fig_distributions.pdf"); plt.show()

# %% 9 · Package versions (for requirements.txt) -----------------------------------------------------
from importlib.metadata import version, PackageNotFoundError
pins = []
for name in ("torch", "torchvision", "segmentation-models-pytorch", "timm", "huggingface-hub", "numpy", "scipy",
             "pandas", "scikit-learn", "opencv-python", "opencv-python-headless", "pycocotools", "matplotlib", "tqdm"):
    try:
        pins.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        pass
Path(out / "requirements_used.txt").write_text("\n".join(pins) + "\n")
print("\n".join(pins))
