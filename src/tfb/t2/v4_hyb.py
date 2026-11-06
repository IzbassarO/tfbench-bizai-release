"""V4 onset retrained on hybrid T+30 labels, same rows/features/params.

Rows stay V4's sample (eligible T+30 cells of V4's training windows); only train_y is replaced by a caller-supplied label
grid. Models: v4/models/<fold>/<group>/seed<s>.txt. Released-window feature matrices are built once with the deployment
path (v4_predict) and cached in v4/released/, so any retrained group predicts val/private without re-running feature builders.

  python -m tfb.t2.v4_hyb --stage released     # cache released features; checks the V4 models reproduce probs_v4_onset
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .io import history_arrays, panel_names, read_network, read_panel_arrays, read_window_assets
from .run import PROJECT, log
from .v2_experiment import load_model
from .v4_model import fit_model, load_pool
from ..io import RELEASE as RELEASE_ROOT   # TFB_RELEASE (run.py --data), else data/kaggle_public

OUT = PROJECT / "outputs/t2"
ROOT = RELEASE_ROOT
REL = OUT / "v4/released"


def _args(**kw):
    a = SimpleNamespace(root=ROOT, output=OUT, rounds=350, threads=4, feature_policy="discharge")
    a.panels = panel_names(ROOT); a.__dict__.update(kw)
    return a


def train_rows(fold: str, panel: str) -> pd.DataFrame:
    """One row per V4 training sample of a panel (load_pool order): T+30 slot index and link id."""
    a = _args(); net = read_network(ROOT, panel)
    tw = pd.read_csv(OUT / "v4/data" / fold / f"{panel}_train_windows.csv")
    with np.load(OUT / "v4/data" / fold / f"{panel}.npz") as d:
        li, y = d["train_link_index"], d["train_y"]
    t30 = np.repeat(tw.origin_idx.to_numpy() + 6, tw.train_row_stop - tw.train_row_start)
    assert len(t30) == len(li) == len(y) == tw.train_row_stop.iloc[-1]
    return pd.DataFrame({"t30": t30, "link_id": np.asarray(net.link_ids)[li], "y": y})


def relabel(fold: str, label_fn) -> tuple[np.ndarray, pd.DataFrame]:
    """label_fn(panel, t30_idx, link_ids) -> bool labels (family A arrays' time index). Returns new y in
    load_pool order + per-panel change counts."""
    ys, stats = [], []
    for panel in _args().panels:
        r = train_rows(fold, panel); y = np.asarray(label_fn(panel, r.t30.to_numpy(), r.link_id.to_numpy()), bool)
        ys.append(y)
        stats.append(dict(fold=fold, panel=panel, rows=len(y), pos_old=int(r.y.sum()), pos_new=int(y.sum()),
                          flip_to_pos=int((y & ~r.y).sum()), flip_to_neg=int((~y & r.y).sum())))
    return np.concatenate(ys), pd.DataFrame(stats)


def fit(fold: str, group: str, y=None, seeds=(42, 137, 271), threads=4):
    a = _args(threads=threads)
    x, y0, w, _, _, names = load_pool(a, fold)
    y = y0 if y is None else y; assert len(y) == len(y0)
    return [fit_model(a, x, y, w, names, OUT / "v4/models" / fold / group / f"seed{s}.txt", seed=s) for s in seeds], names


def released_features():
    """Cache [W, L, 392] onset features for every released val/private onset window (deployment builder)."""
    from .v4_predict import build_window_features
    REL.mkdir(parents=True, exist_ok=True); base = OUT / "datasets/final_enhanced_release"
    for panel_id, panel in enumerate(_args().panels):
        net = read_network(ROOT, panel)
        prior, event_prior = np.load(base / f"{panel}_prior.npy"), np.load(base / f"{panel}_onset_prior.npy")
        with np.load(OUT / "v4/data/final" / f"{panel}_capacity.npz") as d:
            capacity = {k: d[k] for k in d.files}
        with np.load(OUT / "v4/data/final" / f"{panel}.npz") as d:
            frozen = d["feature_names"].tolist()
        for split in ("validation", "private"):
            dest = REL / f"{panel}_{split}.npz"
            if dest.exists():
                continue
            index, history, _ = read_window_assets(ROOT, panel, split)
            days = set(index.date.astype(str)); days |= {str(np.datetime64(d) - np.timedelta64(1, "D")) for d in days}
            current = read_panel_arrays(ROOT, net, split, OUT / "cache", masked=True, dates=days)
            xs, wids = [], []
            for w in index[index.condition.eq("queue_onset")].itertuples():
                local = history_arrays(history[history.window_id.eq(w.window_id)], w.forecast_origin, net, current)
                oi = int((local.timestamps[-1] - current.timestamps[0]) / np.timedelta64(5, "m"))
                x, names = build_window_features(local, current, net, location_prior=prior, onset_prior=event_prior,
                                                 capacity=capacity, panel_id=panel_id, origin_idx=oi)
                assert names == frozen
                xs.append(x.astype(np.float32)); wids.append(str(w.window_id))
            np.savez(dest, x=np.stack(xs), window_id=np.array(wids), link_id=np.asarray(net.link_ids, str))
            log(f"v4_hyb released features {panel}/{split}: {len(xs)} windows")


def predict_released(models) -> pd.DataFrame:
    """Long frame window_id, link_id, p (T+30) for all released onset windows."""
    parts = []
    for panel in _args().panels:
        for split in ("validation", "private"):
            with np.load(REL / f"{panel}_{split}.npz") as d:
                x, wid, lid = d["x"], d["window_id"], d["link_id"]
            p = np.mean([m.predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2]) for m in models], 0)
            parts.append(pd.DataFrame({"window_id": np.repeat(wid, len(lid)), "link_id": np.tile(lid, len(wid)), "p": p.ravel()}))
    return pd.concat(parts, ignore_index=True)


def frozen_models(fold="final", group="normal", seeds=(42, 137, 271)):
    return [load_model(OUT / "v4/models" / fold / group / f"seed{s}.txt", 4) for s in seeds]


def check_released():
    """The final V4 models on cached features must reproduce probs_v4_onset.parquet (T+30 rows)."""
    p = predict_released(frozen_models())
    ref = pd.read_parquet(OUT / "probs_v4_onset.parquet")
    ref = ref[ref.window_id.isin(set(p.window_id))]
    last = ref.groupby("window_id").timestamp.transform("max") == ref.timestamp
    m = p.merge(ref[last][["window_id", "link_id", "queue_prob"]], on=["window_id", "link_id"], validate="1:1")
    assert len(m) == len(p), (len(m), len(p))
    err = float(np.abs(m.p - m.queue_prob).max()); log(f"v4_hyb released check: {len(m):,} cells, max |dp| {err:.2e}")
    assert err < 1e-9
    return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="released"); a = ap.parse_args()
    if a.stage == "released":
        released_features(); check_released()
