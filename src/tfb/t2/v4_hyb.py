"""V4 onset retrained on alternative (e.g. hybrid) T+30 labels, same rows/features/params; extra seeds.

Rows stay V4's frozen sample (eligible T+30 cells of V4's training windows); only train_y is replaced
by a caller-supplied label grid. Models: v4/models/<fold>/<group>/seed<s>.txt (fit_model cache+signature).
Released-window feature matrices are built once with the frozen deployment path (v4_predict) and cached
in v4/released/, so any retrained group predicts val/private without re-running feature builders.

Hybrid re-selection (A_hyb2): the release selector (select_rolling_windows on ffill selector labels) is re-run
with the label speed (e.g. hybrid) in place of raw speed; new windows get V4 features from the frozen
deployment builder on the training overlay path (v4_data.causal_history_and_day). Rows = eligible T+30 cells,
y = raw truth there (hybrid == truth on eligible cells). Cache: v4/data_hyb/<tag>/<panel>.npz (all folds).

  python -m tfb.t2.v4_hyb --stage released     # cache released features; checks frozen models reproduce probs_v4_onset
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .io import history_arrays, panel_names, read_network, read_panel_arrays, read_window_assets
from .run import PROJECT, log
from .v2_experiment import load_model
from .v4_model import fit_model, load_pool

OUT = PROJECT / "outputs/t2"
ROOT = PROJECT / "data/kaggle_public"
REL = OUT / "v4/released"
HSEL = OUT / "v4/data_hyb"
CUT = {"janfeb": "2031-01-01", "octnov": "2030-10-01", "final": "2031-03-01"}


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


def predict_cv(fold: str, models) -> dict:
    """{panel: (prob [W, L] at T+30 in family A link order, window_id)} on V4's eval windows."""
    out = {}
    for panel in _args().panels:
        with np.load(OUT / "v4/data" / fold / f"{panel}.npz") as d:
            ex = d["eval_x"]; flat = ex.reshape(-1, ex.shape[-1])
        meta = pd.read_csv(OUT / "v4/data" / fold / f"{panel}_eval.csv")
        out[panel] = (np.mean([m.predict(flat).reshape(ex.shape[:2]) for m in models], 0), meta.window_id.to_numpy(str))
    return out


def released_features():
    """Cache [W, L, 392] onset features for every released val/private onset window (frozen deployment builder)."""
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
    """Frozen final models on cached features must reproduce probs_v4_onset.parquet (T+30 rows)."""
    p = predict_released(frozen_models())
    ref = pd.read_parquet(OUT / "probs_v4_onset.parquet")
    ref = ref[ref.window_id.isin(set(p.window_id))]
    last = ref.groupby("window_id").timestamp.transform("max") == ref.timestamp
    m = p.merge(ref[last][["window_id", "link_id", "queue_prob"]], on=["window_id", "link_id"], validate="1:1")
    assert len(m) == len(p), (len(m), len(p))
    err = float(np.abs(m.p - m.queue_prob).max()); log(f"v4_hyb released check: {len(m):,} cells, max |dp| {err:.2e}")
    assert err < 1e-9
    return err


def select_onset(panel, speed=None) -> pd.DataFrame:
    """Release-policy onset windows of a train panel; speed [T, L] (family A link order) replaces raw speed."""
    from .labels import make_selector_labels
    from .windows import select_rolling_windows
    net = read_network(ROOT, panel); arr = read_panel_arrays(ROOT, net, "train", OUT / "cache")
    lab = make_selector_labels(arr.speed if speed is None else speed, net.vf, method="ffill", timestamps=arr.timestamps)
    w = select_rolling_windows(lab, arr.timestamps, arr.eligible, block_days=30, stride_days=7, seed=42)
    return w[w.condition.eq("queue_onset")].reset_index(drop=True)


def window_features(panel, origins):
    """{origin_idx: [L, 392]} V4 features of train windows (training overlay + frozen deployment builder)."""
    from .v4_data import _load_masked_day, causal_history_and_day
    from .v4_predict import build_window_features
    net = read_network(ROOT, panel); arr = read_panel_arrays(ROOT, net, "train", OUT / "cache")
    base = OUT / "datasets/final_enhanced_release"  # prior columns are masked out of V4 (onset_feature_mask)
    prior, event_prior = np.load(base / f"{panel}_prior.npy"), np.load(base / f"{panel}_onset_prior.npy")
    pid = panel_names(ROOT).index(panel); days, out = {}, {}
    caps = {}
    for f in CUT:
        with np.load(OUT / "v4/data" / f / f"{panel}_capacity.npz") as d:
            caps[f] = {k: d[k] for k in d.files}
    for o in origins:
        date = str(arr.timestamps[o].astype("datetime64[D]"))
        if date not in days:
            days = {date: _load_masked_day(ROOT, net, arr, date)}
        hist, day, slot = causal_history_and_day(arr, days[date], int(o))
        out[int(o)] = {f: build_window_features(hist, day, net, location_prior=prior, onset_prior=event_prior,
                                                capacity=c, panel_id=pid, origin_idx=slot)[0].astype(np.float32)
                       for f, c in caps.items()}
    return out


def build_hybsel(tag, speed_fn, panels=None):
    """speed_fn(panel, net) -> [T, L] selector speed. Writes v4/data_hyb/<tag>/<panel>.npz + windows csv."""
    from .labels import make_labels
    dst = HSEL / tag; dst.mkdir(parents=True, exist_ok=True); rows = []
    for panel in panels or _args().panels:
        net = read_network(ROOT, panel); arr = read_panel_arrays(ROOT, net, "train", OUT / "cache")
        old = pd.read_csv(OUT / "datasets/final_enhanced_release" / f"{panel}_windows.csv")
        old = set(old[old.condition.eq("queue_onset")].origin_idx)
        w = select_onset(panel, speed_fn(panel, net)); w.to_csv(dst / f"{panel}_windows.csv", index=False)
        X = window_features(panel, w.origin_idx.to_numpy())
        pack = {}
        for f, cut in CUT.items():
            xs, ys, oo = [], [], []
            for o in w.origin_idx.to_numpy(int):
                if arr.timestamps[o + 6] >= np.datetime64(cut, "m"):
                    continue
                v = arr.eligible[o + 6]; xs.append(X[o][f][v])
                ys.append(make_labels(arr.speed[o + 6:o + 7], net.vf)[0][v]); oo.append(np.full(v.sum(), o, np.int32))
            pack.update({f"{f}_x": np.concatenate(xs), f"{f}_y": np.concatenate(ys), f"{f}_origin": np.concatenate(oo)})
        np.savez(dst / f"{panel}.npz", **pack)
        new = set(w.origin_idx)
        rows.append(dict(panel=panel, windows_old=len(old), windows_new=len(new), kept=len(old & new),
                         dropped=len(old - new), added=len(new - old),
                         **{f"{f}_rows": len(pack[f"{f}_y"]) for f in CUT}, **{f"{f}_pos": int(pack[f"{f}_y"].sum()) for f in CUT}))
        log(f"v4_hyb hybsel {tag} {panel}: {rows[-1]}")
        del arr, X, pack
    return pd.DataFrame(rows)


def fit_hybsel(fold, tag, seeds=(42, 137, 271), threads=4):
    a = _args(threads=threads)
    with np.load(OUT / "v4/data/final" / f"{a.panels[0]}.npz") as d:
        names = d["feature_names"].tolist()
    xs, ys = [], []
    for panel in a.panels:
        with np.load(HSEL / tag / f"{panel}.npz") as d:
            xs.append(d[f"{fold}_x"]); ys.append(d[f"{fold}_y"])
    x, y = np.concatenate(xs), np.concatenate(ys); del xs
    return [fit_model(a, x, y, np.ones(len(y)), names, OUT / "v4/models" / fold / f"hybsel_{tag}" / f"seed{s}.txt", seed=s)
            for s in seeds]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="released"); a = ap.parse_args()
    if a.stage == "released":
        released_features(); check_released()
