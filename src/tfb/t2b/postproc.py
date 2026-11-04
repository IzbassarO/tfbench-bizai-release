"""Physical post-processing of decoded queue grids along the corridor (Task 2), stage `postproc` (RUNTIME=cpu).

A queue grows upstream from a bottleneck as one contiguous run of links. Per window and horizon, along the links in
downstream order (tfb.t2b.data.load):
  gap    fill runs of <= gap unqueued links between two queued links (holes inside a queue)
  up     extend every run `up` links upstream of its tail (the queue's upstream end; the bottleneck is the head)
  run    drop runs shorter than `run` links (isolated cells); a window that would become empty keeps its input
  tgap   ongoing only: fill a horizon queued at both h-1 and h+1 (single-step temporal holes)
Onset predicts T+30 only, so tgap is ignored there. Parameters are chosen on the train CV folds only (family A
official-like windows, raw truth on eligible cells), separately for onset (A_hyb) and ongoing (E25). Public LB is never used.

allin queue.yaml: component onset|ongoing, runtime cpu, producer: tfb.t2b.postproc:produce, config.pp = {"gap": 1}
grid / standalone: python -m tfb.t2b.postproc --results DIR [--params '{"min_gain": 0.004}'] [--force]
  -> DIR/postproc/{cv.csv, result.json}; if a condition passes the gate (min-fold gain >= min_gain on both folds):
     outputs/t2/queue_onset_Ahyb_E25_pp.csv and queue_onset_Ahyb_6s_E25_pp.csv (checked with run.verify)
"""
from __future__ import annotations
import argparse, itertools, json, os, pickle, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
from .. import io
from ..t2.decode import expected_iou_decode
from . import codex, evaluate as ev, hybrid as H, run as R
from .probavg import _fam
from .report import _grids, load_fold

STAGE = "postproc"
DEFAULTS = dict(gap=(0, 1, 2), up=(0, 1), run=(1, 2, 3), tgap=(0, 1), min_gain=0.004, e25_w=0.25,
                sources=("queue_onset_Ahyb_E25.csv", "queue_onset_Ahyb_6s_E25.csv"))
IDENTITY = dict(gap=0, up=0, run=1, tgap=0)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str)); os.replace(tmp, path)


def _runs(D):
    """D [..., L] bool -> (run id per cell, 0 = not queued; run length per cell)."""
    start = D & ~np.concatenate([np.zeros(D.shape[:-1] + (1,), bool), D[..., :-1]], -1)
    flat = D.reshape(-1, D.shape[-1]); sid = np.cumsum(start.reshape(-1), 0).reshape(flat.shape) * flat
    length = np.bincount(sid.ravel())
    length[0] = 0
    return sid.reshape(D.shape), length[sid].reshape(D.shape)


def pp(D: np.ndarray, gap=0, up=0, run=1, tgap=0) -> np.ndarray:
    """D [n, 6, L] bool (links downstream-ordered) -> post-processed copy (see module doc)."""
    X = D.copy(); L = X.shape[-1]; idx = np.arange(L)
    if gap:
        last = np.maximum.accumulate(np.where(X, idx, -1), -1)                       # last queued at or upstream of l
        nxt = np.flip(np.minimum.accumulate(np.flip(np.where(X, idx, L), -1), -1), -1)  # next queued at or downstream
        X |= (last >= 0) & (nxt < L) & (nxt - last - 1 <= gap)
    for k in range(1, up + 1):
        X[..., :-k] |= X[..., k:]
    if run > 1:
        X &= _runs(X)[1] >= run
    if tgap and X.shape[1] > 2:
        X[:, 1:-1] |= X[:, :-2] & X[:, 2:]
    empty = ~X.any((1, 2)) & D.any((1, 2))
    X[empty] = D[empty]
    return X


def _grid(cond, p):
    keys = ("gap", "up", "run") + (("tgap",) if cond == "queue_ongoing" else ())
    return [dict(IDENTITY, **dict(zip(keys, v))) for v in itertools.product(*(p[k] for k in keys))]


def cv_inputs(fold: str, e25_w=0.25, conds=("queue_onset", "queue_ongoing")) -> dict:
    """{cond: [(panel, D [n,6,L], Y, E, window_id)]}: onset = A_hyb (cv_<fold>_pure.pkl, 0.5 v4_hyb + 0.5 noloc_hyb, decode30),
    ongoing = E25 = expected-IoU decode of (1 - w) v2 raw + w family B noloc (as tfb.t2b.ladder.cv_ongoing)."""
    out = {c: [] for c in conds}
    Z = pickle.load(open(H.OUT / f"cv_{fold}_pure.pkl", "rb"))
    for p in Z["v4h"] if "queue_onset" in conds else ():
        B = codex.windows(fold, p); mk = (B["meta"].condition == "queue_onset").to_numpy()
        out["queue_onset"].append((p, H.dec30(0.5 * Z["v4h"][p] + 0.5 * Z["nlh"][p]), B["Y"][mk], B["E"][mk],
                                   B["meta"].window_id.to_numpy()[mk]))
    if "queue_ongoing" not in conds:
        return out
    probs, cb = load_fold(fold, "hybrid")
    for (pk, cond, _), (Pv, wpos) in _grids(probs).items():
        if "@" in pk and cond == "queue_ongoing":
            b = cb[pk]; P = e25_w * Pv["noloc"] + (1 - e25_w) * b["codex"]["probability_v2"][wpos]
            D = np.stack([expected_iou_decode(x, force_nonempty=True) for x in P]).astype(bool)
            out["queue_ongoing"].append((pk.split("@")[0], D, b["Y"][wpos], b["E"][wpos], b["meta"].window_id.to_numpy()[wpos]))
    return out


def cv(params: dict, log=print) -> pd.DataFrame:
    rows = []
    for fold in ("janfeb", "octnov"):
        t0 = time.time(); inp = cv_inputs(fold, params["e25_w"])
        for cond, items in inp.items():
            for cfg in _grid(cond, params):
                df = pd.concat([pd.DataFrame({"panel": p, "iou": ev.window_iou(pp(D, **cfg), Y, E)}) for p, D, Y, E, _ in items])
                rows.append(dict(fold=fold, cond=cond, **cfg, score=_fam(df)))
        log(f"[postproc] {fold}: {len(rows)} rows, {time.time() - t0:.0f} s")
    t = pd.DataFrame(rows); k = list(IDENTITY)
    base = t[(t[k] == pd.Series(IDENTITY)).all(1)].set_index(["fold", "cond"]).score
    t["gain"] = t.score - base.reindex(pd.MultiIndex.from_frame(t[["fold", "cond"]])).to_numpy()
    return t


def select(t: pd.DataFrame, min_gain: float) -> dict:
    """Per condition: config with the best min-fold gain; accepted only if that gain >= min_gain (both folds)."""
    k = list(IDENTITY); out = {}
    for cond, g in t.groupby("cond"):
        w = g.groupby(k).gain.agg(["min", "mean"]).reset_index().sort_values(["min", "mean"], ascending=False).iloc[0]
        cfg = {c: int(w[c]) for c in k}
        out[cond] = dict(config=cfg, min_fold_gain=float(w["min"]), mean_gain=float(w["mean"]), accepted=bool(w["min"] >= min_gain))
    return out


def apply(src: str, choice: dict, dst: str) -> dict:
    """outputs/t2/<src> -> outputs/t2/<dst> with the accepted configs (onset / ongoing rows separately)."""
    sub = codex.submission(io.OUTPUTS / "t2" / src); parts, changed = [], 0
    for p in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str); on = (wi.condition == "queue_onset").to_numpy()
            G = codex.grid(sub, p, s, wi); D = G.copy()
            for cond, m in (("queue_onset", on), ("queue_ongoing", ~on)):
                if choice[cond]["accepted"] and m.any():
                    D[m] = pp(G[m], **choice[cond]["config"])
            changed += int((D != G).sum()); parts.append(R.to_rows(D, p, s, wi))
    pd.concat(parts, ignore_index=True).to_csv(io.OUTPUTS / "t2" / dst, index=False)
    return dict(src=src, dst=dst, cells_changed=changed, verify=R.verify([f"../t2/{dst}"]))


def write_cv(d: Path, comp: str, fold: str, panel: str, P, Y, E, ref, wid) -> str:
    """One tfb.allin CV file (README 'Queue adapters'): P float [W,6,L], Y/E/reference binary, train split."""
    f = Path(d) / "cv" / f"{comp}_{fold}_{panel}.npz"; f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f.with_suffix(".tmp.npz"), P=np.asarray(P, np.float32), Y=Y.astype(np.uint8), E=E.astype(np.uint8),
             reference=ref.astype(np.uint8), window_id=np.asarray(wid).astype(str), panel=panel, fold=fold, split="train")
    f.with_suffix(".tmp.npz").replace(f)
    return str(f)


def produce(config, ctx):
    """tfb.allin producer (component onset|ongoing, runtime cpu) for ONE fixed config (config["pp"], e.g. {"gap": 1}):
    P = pp(production decode) as 0/1, reference = the production decode (A_hyb onset / E25 ongoing); decode=threshold.
    Choose candidate configs with the grid (run / CLI) first; a job per config keeps the gate free of in-job selection."""
    comp = ctx.task["component"]; cond = f"queue_{comp}"; cfg = dict(IDENTITY, **config["pp"]); files = []
    for fold in ("janfeb", "octnov"):
        for panel, D, Y, E, wid in cv_inputs(fold, config.get("e25_w", 0.25), (cond,))[cond]:
            files.append(write_cv(ctx.checkpoint_dir, comp, fold, panel, pp(D, **cfg), Y, E, D, wid))
        ctx.heartbeat(dict(fold=fold))
    choice = {c: dict(accepted=c == cond, config=cfg) for c in ("queue_onset", "queue_ongoing")}
    dst = f"queue_{comp}_pp_" + "_".join(f"{k}{v}" for k, v in cfg.items()) + ".csv"
    apply(config.get("source", "queue_onset_Ahyb_E25.csv"), choice, dst)
    final = Path(ctx.checkpoint_dir) / dst; shutil.copy2(io.OUTPUTS / "t2" / dst, final)
    return dict(cv_files=files, test_predictions=str(final), decode="threshold", threshold=0.5)


def run(results: Path, params: dict | None = None, force=False, log=print) -> dict:
    params = dict(DEFAULTS, **(params or {})); out = Path(results) / STAGE; res = out / "result.json"
    if res.exists() and not force:
        return json.loads(res.read_text())
    t0 = time.time(); t = cv(params, log); out.mkdir(parents=True, exist_ok=True); t.to_csv(out / "cv.csv", index=False)
    choice = select(t, params["min_gain"]); files = []
    if any(c["accepted"] for c in choice.values()):
        files = [apply(s, choice, s.replace(".csv", "_pp.csv")) for s in params["sources"]]
    r = dict(stage=STAGE, runtime="cpu", status="candidate" if files else "rejected", choice=choice, files=files,
             params=params, seconds=round(time.time() - t0))
    atomic_json(res, r); log(json.dumps(choice, indent=1))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", type=Path, default=io.OUTPUTS / "allin")
    ap.add_argument("--params", default="{}"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); run(a.results, json.loads(a.params), a.force)
