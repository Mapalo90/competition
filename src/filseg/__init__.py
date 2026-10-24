"""filseg: solar filament instance segmentation for the Solar Filament Segmentation Challenge 2026.

Pipeline: whole-disk U-Net at 1024 px trained on the average of each image's annotators, 5-fold
ensemble with flip test-time augmentation, then instance post-processing (fragment joining, area /
confidence / limb filters) tuned for Panoptic Quality on out-of-fold predictions.
"""

__version__ = "1.0.0"
