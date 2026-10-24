"""Locating trained fold models: this repository's names (fold{k}.pt) and the original run outputs."""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path


def find_weights(search_dirs: list, n_folds: int = 5) -> dict:
    """fold -> weight file. Accepts fold{k}.pt, best_fold{k}.pt, and baseline/best.pt (fold 0 of the first run)."""
    found = {}
    for d in search_dirs:
        for p in sorted(Path(d).rglob("*.pt")):
            m = re.fullmatch(r"(?:best_)?fold(\d+)\.pt", p.name)
            if m:
                found.setdefault(int(m.group(1)), p)
            elif p.name == "best.pt" and p.parent.name == "baseline":
                found.setdefault(0, p)
    return {k: found[k] for k in sorted(found) if k < n_folds}


def download_weights(url_template: str, out_dir: str | Path, n_folds: int = 5) -> dict:
    """Download fold{k}.pt files, e.g. from a GitHub release: '.../releases/download/v1.0/fold{k}.pt'."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k in range(n_folds):
        dst = out_dir / f"fold{k}.pt"
        if not dst.exists():
            urllib.request.urlretrieve(url_template.format(k=k), dst)
        paths[k] = dst
    return paths
