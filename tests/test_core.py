"""Fast unit tests on synthetic data (no competition files needed). Run: pytest -q"""
import math

import numpy as np
import pytest

from filseg.config import Config
from filseg.geometry import disk_geometry
from filseg.metrics import bootstrap_interval, image_counts, match_counts, overlap_table, panoptic_quality, pq_parts
from filseg.postprocess import components, keep_mask
from filseg.submission import read_submission, rle_decode, rle_encode, write_submission
from filseg.weights import find_weights


def square(y, x, h, w):
    return (np.ones((h, w), bool), y, x)


def test_pq_hand_computed():
    gt = [square(0, 0, 20, 20), square(100, 100, 20, 20)]
    pred = [square(0, 0, 20, 20), square(100, 100, 20, 10), square(300, 300, 5, 5)]   # exact, IoU 0.5 (no match), stray
    tp, fp, fn, s, _, _ = match_counts(*overlap_table(gt, pred))
    assert (tp, fp, fn) == (1, 2, 1)
    assert pq_parts(tp, fp, fn, s)[0] == pytest.approx(0.4)


def test_split_and_merge_counts():
    split = match_counts(*overlap_table([square(100, 100, 20, 20)], [square(100, 100, 20, 10), square(100, 110, 20, 10)]))
    assert split[4] == 1 and split[0] == 0
    merge = match_counts(*overlap_table([square(0, 0, 20, 10), square(0, 10, 20, 10)], [square(0, 0, 20, 20)]))
    assert merge[5] == 1


def test_pooled_vs_annotator():
    # two annotators drew the same filament; one prediction can match only one pooled copy
    gt_sets = [[square(0, 0, 20, 20)], [square(0, 0, 20, 20)]]
    c = image_counts(gt_sets, [square(0, 0, 20, 20)])
    assert pq_parts(*c["annotator"])[0] == pytest.approx(1.0)
    assert tuple(c["pooled"][:3]) == (1, 0, 1)
    out = panoptic_quality({"img": gt_sets}, {"img": [square(0, 0, 20, 20)]})
    assert out["pooled"]["PQ"] == pytest.approx(1 / 1.5)


def test_rle_round_trip(tmp_path):
    m = np.zeros((2048, 2048), np.uint8)
    m[100:140, 200:260] = 1
    assert np.array_equal(rle_decode(rle_encode(m)), m)
    pred = {"20140101000000Bh": [(np.ones((40, 60), bool), 100, 200)]}
    sub = write_submission(pred, tmp_path / "s.csv")
    assert list(sub.filament_id) == ["20140101000000Bh_1"]
    back = read_submission(tmp_path / "s.csv")
    (bm, by, bx), = back["20140101000000Bh"]
    assert (by, bx, bm.shape) == (100, 200, (40, 60))
    assert bm.base is None or bm.base.size == bm.size          # a compact copy, not a view into a full mask


def test_group_joining_keeps_pixels_and_joins_close_pieces():
    prob = np.zeros((2048, 2048), np.uint8)
    prob[1000:1010, 1000:1100] = 255        # piece 1
    prob[1000:1010, 1110:1200] = 255        # piece 2, 10 px gap
    prob[1500:1510, 1000:1100] = 255        # far away
    comps, feats, cents = components(prob, 0.5, "none", 0)
    assert len(comps) == 3
    comps, feats, cents = components(prob, 0.5, "group", 20)
    assert len(comps) == 2
    assert sorted(feats[:, 0].tolist()) == [1000.0, 1900.0]       # gap pixels are not added
    keep = keep_mask(feats, cents, (1024, 1024, 909), dict(min_area=1500, r_max=0.9, min_score=0.5))
    assert keep.tolist() == [True, False] or keep.tolist() == [False, True]


@pytest.mark.parametrize("halo", [0, 110])
def test_disk_geometry_ignores_halo(halo):
    yy, xx = np.indices((2048, 2048))
    cy, cx, R = 1030, 1015, 905
    rr = np.hypot(yy - cy, xx - cx) / R
    img = np.where(rr < 1, 130 + 30 * np.sqrt(np.clip(1 - rr ** 2, 0, 1)), 0) + np.where(rr > 1, halo * np.exp(-(rr - 1) / 0.15), 0)
    gy, gx, gr = disk_geometry(np.clip(img, 0, 255).astype(np.uint8))
    assert math.hypot(gy - cy, gx - cx) < 2 and abs(gr - R) < 10


def test_config_round_trip(tmp_path):
    c = Config()
    c.post.min_score = 0.8
    c.to_json(tmp_path / "c.json")
    c2 = Config.from_json(tmp_path / "c.json", overrides={"train": {"epochs": 3}})
    assert c2.post.min_score == 0.8 and c2.train.epochs == 3 and c2.train.epochs_for(0) == 40


def test_find_weights_names_and_order(tmp_path):
    for name in ["a/fold1.pt", "a/best_fold2.pt", "a/baseline/best.pt", "a/other.pt", "a/fold7.pt", "b/fold1.pt"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    w = find_weights([tmp_path / "b", tmp_path / "a", tmp_path / "missing"], n_folds=5)
    assert sorted(w) == [0, 1, 2]
    assert w[0].name == "best.pt" and w[1].parent.name == "b"          # earlier folders win


def test_shipped_weights_load():
    """The five fold models in weights/ load into the configured architecture (all keys and shapes match)."""
    pytest.importorskip("segmentation_models_pytorch")
    from filseg.model import build_model, load_weights
    w = find_weights()
    assert sorted(w) == list(range(Config().train.n_folds))
    model = build_model(Config().train.encoder, None, "cpu", channels_last=False, data_parallel=False)
    for path in w.values():
        load_weights(model, path)


def test_bootstrap_interval():
    same = np.tile([4, 2, 2, 3.0], (50, 1))                 # identical images -> no spread
    lo, hi = bootstrap_interval(same)
    assert lo == pytest.approx(hi) == pytest.approx(pq_parts(4, 2, 2, 3.0)[0])
    mixed = np.vstack([np.tile([5, 0, 0, 5.0], (50, 1)), np.tile([0, 5, 5, 0.0], (50, 1))])
    lo, hi = bootstrap_interval(mixed)
    assert lo < 0.5 < hi and hi - lo < 0.3
