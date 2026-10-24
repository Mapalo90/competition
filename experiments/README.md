# Experiment history

The scripts in this folder are the Kaggle notebooks exactly as they were run during the competition, in
order. They are kept for traceability of the numbers in the report. The maintained pipeline is the
`filseg` package (`src/filseg`) and [`notebooks/pipeline.ipynb`](../notebooks/pipeline.ipynb), which
reproduces the same folds and scores.

| Notebook | Hardware | What it did | Main result |
|---|---|---|---|
| `01_eda.py` | CPU | Data audit: annotation structure, filament geometry, resolution and boundary sensitivity, limb distribution, time/site split, annotator agreement | Human vs human PQ 0.34; half resolution keeps median IoU 0.98; 99% of filament centres within 0.9 R |
| `02_baseline.py` | 2 x T4, 1.75 h | ResNet34 U-Net at 1024 px on soft (annotator-average) targets, fold 0, 40 epochs; post-processing search on fold 0 | Fold 0 PQ 0.406 (each annotator) with tuned filters |
| `03_folds.py` | 2 x T4, 4.7 h | Folds 1-4 with the same recipe (30 epochs, faster data pipeline); search on out-of-fold maps | Out-of-fold PQ 0.400 (565 images); folds 0.39-0.41 |
| `04_ensemble_pp.py` | CPU | All five folds' out-of-fold maps (707 images); two-stage search incl. fragment joining; 5-model test ensemble | Out-of-fold PQ 0.408 (each annotator), 0.353 (pooled); public leaderboard 0.36 |
| `05_encoders.py` | 2 x T4, 2.1 h | Encoder comparison on fold 0: ResNet34 vs ResNet50 on 768 px crops | ResNet50 0.426 vs ResNet34 0.427 (full images) at 1.7x cost; not adopted |
| `06_final_pp.py` | CPU | Re-tuning for the pooled scoring rule with ensemble calibration | Fold-0 estimate +0.003 (pooled); public leaderboard 0.35 vs 0.36, within noise; not adopted |

The submitted solution uses the models from 02 (fold 0) and 03 (folds 1-4) with the post-processing
settings from 04, which are the defaults in `src/filseg/config.py`.
