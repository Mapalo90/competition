"""The trained fold models: the float16 copies shipped in this repository (weights/fold{k}.pt), or the files a
training run writes (fold{k}.pt; best_fold{k}.pt and baseline/best.pt in the original Kaggle runs)."""
from __future__ import annotations

import re
from pathlib import Path

REPO_WEIGHTS = Path(__file__).resolve().parents[2] / "weights"     # <repository>/weights, beside src/


def find_weights(search_dirs: list | None = None, n_folds: int = 5) -> dict:
    """fold -> weight file. Folders are searched in order and the first file found for a fold is used;
    by default only this repository's weights/ folder is searched."""
    found = {}
    for d in [REPO_WEIGHTS] if search_dirs is None else search_dirs:
        if not Path(d).is_dir():
            continue
        for p in sorted(Path(d).rglob("*.pt")):
            m = re.fullmatch(r"(?:best_)?fold(\d+)\.pt", p.name)
            k = int(m.group(1)) if m else (0 if p.name == "best.pt" and p.parent.name == "baseline" else None)
            if k is not None and k < n_folds:
                found.setdefault(k, p)
    return dict(sorted(found.items()))
