"""All settings in one place. Defaults reproduce the submitted solution."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """Model and training recipe (one U-Net per fold)."""
    img_size: int = 1024                  # model input; originals are 2048 x 2048
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    epochs: int = 30
    batch_size: int = 8                   # across both T4 GPUs
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    num_workers: int = 4
    amp: bool = True                      # float16 mixed precision
    channels_last: bool = True
    eval_every: int = 10                  # PQ check (and checkpoint) every N epochs
    n_folds: int = 5
    seed: int = 42
    # fold 0 was trained first, for 40 epochs with a PQ check every 5; folds 1-4 used the defaults
    fold_epochs: dict = field(default_factory=lambda: {0: 40})
    fold_eval_every: dict = field(default_factory=lambda: {0: 5})

    def epochs_for(self, fold: int) -> int:
        return int(self.fold_epochs.get(fold, self.fold_epochs.get(str(fold), self.epochs)))

    def eval_every_for(self, fold: int) -> int:
        return int(self.fold_eval_every.get(fold, self.fold_eval_every.get(str(fold), self.eval_every)))


@dataclass
class PostConfig:
    """Probability map -> filament instances. Values tuned on all 707 out-of-fold maps."""
    thr: float = 0.5                      # binarisation threshold on the averaged probability map
    merge: str = "group"                  # "group": pieces closer than d px form one filament; "close"; "none"
    d: int = 20                           # joining distance in full-resolution pixels
    fill: bool = False                    # fill holes inside each filament
    min_area: int = 300                   # drop filaments smaller than this (px at 2048)
    r_max: float = 0.9                    # drop filaments whose centre lies beyond 0.9 solar radii
    min_score: float = 0.85               # drop filaments whose mean probability is below this
    tta: bool = True                      # average the four flips at inference


@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    post: PostConfig = field(default_factory=PostConfig)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def from_json(cls, path: str | Path | None = None, overrides: dict | None = None) -> "Config":
        raw = json.loads(Path(path).read_text()) if path else {}
        for section, values in (overrides or {}).items():
            raw.setdefault(section, {}).update(values)
        return cls(train=TrainConfig(**raw.get("train", {})), post=PostConfig(**raw.get("post", {})))
