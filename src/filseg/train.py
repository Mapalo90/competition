"""Training one fold: augmentation, loss, schedule, PQ-based checkpointing."""
from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import PostConfig, TrainConfig
from .metrics import image_counts, pq_parts
from .model import autocast, build_model, load_weights, predict_maps, save_weights, to_input
from .postprocess import components, keep_mask, upsample

DEFAULT_POST = PostConfig(thr=0.5, merge="none", d=0, min_area=0, r_max=1.01, min_score=0.0)   # used for checkpointing


class FilamentDataset(Dataset):
    """Cached uint8 images X and soft targets Y (N, S, S); returns augmented uint8 tensors."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, idx: list):
        self.X, self.Y, self.idx = X, Y, list(idx)
        self._rng = None

    def __len__(self):
        return len(self.idx)

    def rng(self):
        if self._rng is None:                  # one generator per DataLoader worker
            self._rng = np.random.default_rng(np.random.randint(2 ** 31))
        return self._rng

    def __getitem__(self, k):
        x, y = self.X[self.idx[k]], self.Y[self.idx[k]]
        S, r = x.shape[0], self.rng()
        kk = r.integers(4)                                          # dihedral: the Sun has no "up"
        x, y = np.rot90(x, kk), np.rot90(y, kk)
        if r.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        if r.random() < 0.5:                                        # small rotation / scale / shift
            M = cv2.getRotationMatrix2D((S / 2, S / 2), r.uniform(-45, 45), r.uniform(0.95, 1.05))
            M[:, 2] += r.uniform(-0.01, 0.01, 2) * S
            x = cv2.warpAffine(x, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
            y = cv2.warpAffine(y, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
        a, b, g = r.uniform(0.85, 1.15), r.uniform(-15, 15), r.uniform(0.8, 1.25)   # contrast, brightness, gamma
        x = cv2.LUT(x, np.clip(255 * (np.clip(np.arange(256) * a + b, 0, 255) / 255) ** g, 0, 255).astype(np.uint8))
        if r.random() < 0.2:
            x = cv2.GaussianBlur(x, (0, 0), r.uniform(0.5, 1.2))   # seeing
        if r.random() < 0.2:
            x = np.clip(x + r.standard_normal(x.shape, dtype=np.float32) * r.uniform(1, 4), 0, 255).astype(np.uint8)
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy + soft Dice against the soft (annotator-average) target."""
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return nn.functional.binary_cross_entropy_with_logits(logits, y) + dice.mean()


def quick_pq(maps: list, obs: list, gt: dict, geom: dict, p: PostConfig = DEFAULT_POST) -> tuple:
    """PQ (annotator rule) of model-resolution maps under post-processing `p`."""
    tot = np.zeros(6)
    for m, o in zip(maps, obs):
        comps, feats, cents = components(upsample(m), p.thr, p.merge, p.d, p.fill)
        tot += image_counts(gt[o], comps, keep_mask(feats, cents, geom[o], p))["annotator"]
    return pq_parts(*tot)


def train_fold(cfg: TrainConfig, fold: int, X: np.ndarray, Y: np.ndarray, train_idx: list, val_idx: list,
               val_obs: list, gt: dict, geom: dict, out_dir: str | Path, device=None, test_images: list | None = None,
               test_obs: list | None = None, log=print) -> dict:
    """Train one fold; keeps the checkpoint with the best validation PQ. Returns validation/test maps."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and torch.device(device).type == "cuda"
    epochs, eval_every = cfg.epochs_for(fold), cfg.eval_every_for(fold)
    dl = DataLoader(FilamentDataset(X, Y, train_idx), batch_size=cfg.batch_size, shuffle=True,
                    drop_last=len(train_idx) > cfg.batch_size, num_workers=cfg.num_workers,
                    pin_memory=torch.device(device).type == "cuda", persistent_workers=cfg.num_workers > 0)
    model = build_model(cfg.encoder, cfg.encoder_weights, device, cfg.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps, warm = max(1, len(dl)) * epochs, max(1, len(dl)) * cfg.warmup_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    val_small = [X[i] for i in val_idx]
    ckpt = out_dir / f"fold{fold}.pt"
    rows, best, t0 = [], -1.0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tot = n = 0
        te = time.time()
        for x, y in dl:
            x, y = to_input(x, device, cfg.channels_last), y.to(device, non_blocking=True).float() / 255
            with autocast(device, use_amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        row = dict(fold=fold, epoch=epoch, train_loss=round(tot / max(n, 1), 4), lr=opt.param_groups[0]["lr"],
                   epoch_min=round((time.time() - te) / 60, 2))
        if epoch % eval_every == 0 or epoch == epochs:
            pq, sq, rq = quick_pq(predict_maps(model, val_small, device, tta=False, amp=use_amp,
                                               channels_last=cfg.channels_last), val_obs, gt, geom)
            row.update(val_PQ=round(float(pq), 4), val_SQ=round(float(sq), 4), val_RQ=round(float(rq), 4))
            if pq > best:
                best = pq
                save_weights(model, ckpt)
                row["saved"] = True
        rows.append(row)
        log(row)
    pd.DataFrame(rows).to_csv(out_dir / f"train_log_fold{fold}.csv", index=False)
    load_weights(model, ckpt, device)
    val_probs = predict_maps(model, val_small, device, amp=use_amp, channels_last=cfg.channels_last)
    np.savez_compressed(out_dir / f"val_probs_fold{fold}.npz", obs=np.array(val_obs), probs=np.stack(val_probs))
    test_probs = []
    if test_images:
        test_probs = predict_maps(model, test_images, device, amp=use_amp, channels_last=cfg.channels_last)
        names = test_obs if test_obs is not None else [str(i) for i in range(len(test_images))]
        np.savez_compressed(out_dir / f"test_probs_fold{fold}.npz", obs=np.array(names), probs=np.stack(test_probs))
    del model, opt, sched, scaler, dl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"fold {fold}: best validation PQ {best:.4f} (default post-processing, no TTA) in {(time.time() - t0) / 60:.0f} min")
    return dict(val_obs=val_obs, val_probs=val_probs, test_probs=test_probs, log=rows)
