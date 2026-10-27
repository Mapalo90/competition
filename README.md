# Solar filament segmentation (Solar Filament Segmentation Challenge 2026)

Repository: https://github.com/Mapalo90/competition

Instance segmentation of solar filaments in full-disk GONG H-alpha images, trained on the MAGFiLO
annotations. A whole-disk U-Net predicts how likely each pixel is to be labelled as filament; a tuned
post-processing step turns the averaged prediction of five fold models into individual filaments.

| | Panoptic Quality | 95% interval |
|---|---|---|
| Out-of-fold, all 707 training images, scored against each annotator | **0.408** (SQ 0.668, RQ 0.611) | 0.399-0.418 |
| Same, all annotators of an image pooled, one-to-one matching | 0.353 (SQ 0.681, RQ 0.519) | 0.344-0.363 |
| Human agreement (one annotator scored against another, same images) | 0.340 | 0.324-0.356 |
| Public leaderboard | 0.36 | |

Per fold (each annotator): 0.427 / 0.405 / 0.391 / 0.409 / 0.409. Intervals are percentile bootstraps over images.

**Reproduced from this repository.** Run on Kaggle from the models in `weights/` and the competition data
alone, the notebook gives the numbers above and rebuilds the submitted file: 1,150 of its 1,152 filaments,
mean IoU 0.9998 (PQ 0.999 against the submitted file); the other two sit exactly at the 0.85
mean-probability threshold.

## Method

1. **Input.** Each 2048 x 2048 JPEG is downsampled to 1024 x 1024. A perfect prediction at this
   resolution, upsampled back, still has a median IoU of 0.98 with its label, so the whole disk fits in
   one pass without tiling.
2. **Target.** 42% of training images were labelled by two or three annotators. The target is the
   fraction of annotators who marked each pixel, so the network learns labelling probability rather than
   one annotator's style.
3. **Model.** U-Net with an ImageNet ResNet34 encoder (segmentation-models-pytorch), 1-channel input,
   BCE + soft Dice loss, AdamW with cosine schedule, mixed precision, random dihedral flips, rotation,
   scale, brightness / contrast / gamma, blur and noise. 30 epochs per fold (40 for fold 0).
4. **Validation.** Five folds over images, all annotators of an image in the same fold, stratified by
   year. Checkpoints are chosen by Panoptic Quality, not loss.
5. **Ensemble.** The five fold models' probability maps are averaged, each with four-flip test-time
   augmentation.
6. **Instances.** At full resolution: threshold 0.5; pieces less than 20 px apart are joined into one
   filament (annotators often draw a broken filament as one); filaments under 300 px, with mean
   probability under 0.85, or centred beyond 0.9 solar radii are dropped. The solar disk is located by a
   circle fit to limb points, which is robust to the off-disk halo present in ~14% of images. All five
   settings were chosen by a two-stage search (1,236 combinations) on the out-of-fold predictions.

## What we learned (details in the report)

- **The labels set the ceiling.** Two annotators agree at PQ 0.34; the model, at 0.41 against each
  annotator, already exceeds that.
- **A bigger encoder did not help.** On fold 0, ResNet50 (768 px crops) matched ResNet34 (0.426 vs 0.427;
  difference -0.001, 95% interval -0.010 to +0.007) at 1.7x the inference cost.
- **Scoring rule.** The public leaderboard's level is consistent with PQ against all annotators of an
  image pooled with one-to-one matching (0.353 out-of-fold vs 0.35-0.36), rather than against each annotator
  (0.408). A filament drawn by three annotators then appears three times in the answer key, and one
  prediction can only match one copy.
- **Fine structures.** Barbs and threads (parts under 5 px wide) are 2.4% of the labelled area; the
  model captures 45% of them within detected filaments. Most of the remaining IoU loss is a 1-2 px
  disagreement along the edge of the main body, where annotators also disagree with each other. An
  image-guided refinement (re-deciding a thin band around each filament by local darkness and growing
  through connected dark pixels) raised the share of labelled thin parts covered from 35.5% to at most 40%
  without improving PQ, so it is not part of the pipeline.

## Reproduce

The five trained fold models are in [`weights/`](weights) (float16, 49 MB each), so reproducing the
submission needs only this repository and the competition data; nothing is retrained.

**On Kaggle** (recommended): create a notebook from the competition page, turn Internet on, import
[`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb) and run all. Its last section scores the new
submission against the file we submitted, [`submissions/submission_kaggle.csv`](submissions/submission_kaggle.csv).

| `MODE` | time | |
|---|---|---|
| `"weights"` (default) | ~30 min, GPU T4 x2 | load the models from `weights/`, recompute all probability maps, evaluate, write the submission |
| `"train"` | ~7 h, GPU T4 x2 | train the five fold models from scratch, then as above |
| `"maps"` | ~10 min, CPU | start from probability maps saved by an earlier run (attached), then as above |

**Locally or from a shell:**

```bash
pip install -r requirements.txt && pip install -e .
python -m filseg oof      --data /path/to/competition --out runs     # maps from the models in weights/
python -m filseg evaluate --data /path/to/competition --out runs     # out-of-fold PQ, both rules
python -m filseg submit   --data /path/to/competition --out runs     # runs/submission.csv
python -m filseg tune     --data /path/to/competition --out runs     # optional: post-processing search
pytest -q                                                            # unit tests
```

`python -m filseg train` trains new fold models into `--out` instead (`oof` then uses those).
All settings live in [`src/filseg/config.py`](src/filseg/config.py); the defaults reproduce the submission.

## Efficiency

Training: about 2.3 min per epoch on two T4 GPUs (about 70 min per fold). Inference: about 0.8 s per
image per model with test-time augmentation on two T4s, so about 4 s per image for the five-model
ensemble; post-processing about 0.2 s per image on one CPU core.

## Repository

```
src/filseg/
  config.py        all settings (defaults = submitted solution)
  data.py          finding images and annotations, decoding masks, soft targets, folds
  geometry.py      solar disk centre and radius (limb filter)
  model.py         U-Net, test-time augmentation, weight files
  train.py         augmentation, loss, training loop with PQ checkpointing
  postprocess.py   probability map -> filament instances
  metrics.py       Panoptic Quality under both scoring rules, split / merge counts
  tune.py          post-processing search on out-of-fold maps
  pipeline.py      shared high-level steps
  submission.py    run-length encoding and the submission file
  viz.py           overlays and the IoU / Dice / one-to-many figures
  __main__.py      command line
notebooks/pipeline.ipynb   end-to-end notebook (pipeline.py: the same notebook as a script)
weights/                   the five trained fold models, with checksums and provenance
submissions/               the file submitted to Kaggle (public leaderboard 0.36)
configs/final.json         the submitted settings, as a file
experiments/               the Kaggle notebooks as run during the competition
report/                    technical report in the official template (LaTeX, figures, PDF)
tests/                     unit tests (synthetic data, run in seconds)
```

## Data and acknowledgements

MAGFiLO: Ahmadzadeh et al., *A dataset of manually annotated filaments from H-alpha observations*,
Scientific Data 11, 1031 (2024), doi:10.1038/s41597-024-03876-y. This work uses GONG data obtained by the
NSO Integrated Synoptic Program, managed by the National Solar Observatory, which is operated by AURA
under a cooperative agreement with the National Science Foundation and with contribution from NOAA.

Released under the MIT License (see `LICENSE`).
