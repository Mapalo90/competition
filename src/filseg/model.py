"""U-Net construction, inference with flip test-time augmentation, and weight files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MEAN, STD = 0.449, 0.226          # ImageNet mean/std averaged over RGB, for 1-channel input


def build_model(encoder: str = "resnet34", encoder_weights: str | None = "imagenet", device: str | torch.device = "cuda",
                channels_last: bool = True, data_parallel: bool = True) -> nn.Module:
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=1, classes=1)
    m = m.to(device)
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    if data_parallel and torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
    return m


def unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m


def to_input(x_u8: torch.Tensor, device, channels_last: bool = True) -> torch.Tensor:
    """uint8 (N, 1, S, S) -> normalised float on the device (normalising on the GPU keeps loading cheap)."""
    x = (x_u8.to(device, non_blocking=True).float() / 255 - MEAN) / STD
    return x.contiguous(memory_format=torch.channels_last if channels_last else torch.contiguous_format)


def autocast(device, enabled: bool = True):
    dev = torch.device(device)
    return torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=enabled and dev.type == "cuda")


@torch.no_grad()
def predict_maps(model: nn.Module, images_u8: list, device, tta: bool = True, batch_size: int = 4, amp: bool = True,
                 channels_last: bool = True) -> list:
    """uint8 S x S images -> uint8 S x S probability maps; with `tta`, the mean over the four flips."""
    model.eval()
    views = [((), ())] + ([((-1,), (-1,)), ((-2,), (-2,)), ((-1, -2), (-1, -2))] if tta else [])
    out = []
    for i in range(0, len(images_u8), batch_size):
        x = to_input(torch.from_numpy(np.stack(images_u8[i:i + batch_size]))[:, None], device, channels_last)
        p = 0
        for fwd, back in views:
            xi = x.flip(fwd) if fwd else x
            with autocast(device, amp):
                pi = torch.sigmoid(model(xi).float())
            p = p + (pi.flip(back) if back else pi)
        p = (p / len(views))[:, 0].cpu().numpy()
        out += list(np.round(p * 255).astype(np.uint8))
    return out


def save_weights(model: nn.Module, path: str | Path, half: bool = True) -> None:
    """State dict, by default in float16 (~49 MB for ResNet34 U-Net, under GitHub's 50 MB soft limit)."""
    sd = unwrap(model).state_dict()
    if half:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save(sd, path)


def load_weights(model: nn.Module, path: str | Path, device="cpu") -> nn.Module:
    sd = torch.load(path, map_location=device)
    target = unwrap(model)
    ref = target.state_dict()
    target.load_state_dict({k: v.to(ref[k].dtype) for k, v in sd.items()})
    return model
