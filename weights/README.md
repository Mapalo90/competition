# Trained models

The five fold models of the submitted solution. Each is a U-Net with a ResNet34 encoder
(segmentation-models-pytorch 0.5.0, one input and one output channel, 24.4 million parameters), stored as a
float16 state dict. `filseg.model.load_weights` loads a file into the float32 model built by
`filseg.model.build_model`; the notebook and `python -m filseg oof` use this folder by default.

| File | Validation fold | Trained by | Epochs | Out-of-fold PQ* | Size | SHA-256 |
|---|---|---|---|---|---|---|
| `fold0.pt` | 0 (142 images) | `experiments/02_baseline.py` | 40 | 0.427 | 49.0 MB | `e60815190177c2c676cd4d80b766141838979c915a96ab0d4926af680c242331` |
| `fold1.pt` | 1 (142 images) | `experiments/03_folds.py` | 30 | 0.405 | 49.0 MB | `2c91c9691c6308089c0f08c194f4fb817a3341b1aedbd2092e4214d0d38cd9e7` |
| `fold2.pt` | 2 (141 images) | `experiments/03_folds.py` | 30 | 0.391 | 49.0 MB | `d30a585c14c65ea538f3c9507b29eba8edc73998c41bb667a441e917a9fcd994` |
| `fold3.pt` | 3 (141 images) | `experiments/03_folds.py` | 30 | 0.409 | 49.0 MB | `de5d3ce085f3f767b5bfd942c48c1544981c24ac7be3ea647fada786f7b4118c` |
| `fold4.pt` | 4 (141 images) | `experiments/03_folds.py` | 30 | 0.409 | 49.0 MB | `7a95fd7175a54adddd42370ffbf29ac7334560353d50a24bf17a3e610ac0ff0e` |

\*PQ against each annotator on the fold's own images, with the final post-processing.

Each model was trained on the other four folds. The checkpoint kept is the one with the best validation Panoptic
Quality, checked every 5 epochs for fold 0 (epoch 35 was kept) and every 10 epochs for folds 1-4. The files were
converted from those float32 checkpoints with `filseg.model.save_weights`. On validation images the float16 copy
changes the thresholded mask by at most a few pixels per million, about the same as the difference between CPU and
GPU inference.
