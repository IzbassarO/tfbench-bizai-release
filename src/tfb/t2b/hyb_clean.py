"""hyb_clean: A_hyb rebuilt on hybrid labels from the clean imputer (no validation/private rows in any fit).

Rule (organizers, 2026-10-29): a window with origin T may use any released data stamped <= T, but no data after T,
also not indirectly through fitting. The recipe's A_hyb labels come from the v4t1 imputer, which was fitted with
transductive rows over the whole validation/private months; this stage removes that path:
  imputer  tfb.t1.imputer_clean: same LightGBM, train rows only (speed)            -> <dir>/imputer/lgbm_speed.txt
  labels   tfb.t1.hybrid layout at n_trees (recipe: 4000)                           -> <dir>/train_hybrid_speed_clean/
  cv       tfb.t2b.hyb_relabel.cv_files, label mode "mix_clean" (v4_hyb + t2b noloc_hyb relabelled), both folds;
           reference = the current A_hyb (mode pure)                                 -> <dir>/cv/onset_*.npz
  final    v4_hyb + noloc_hyb final fits, A_hyb_mix_clean spliced with E25           -> outputs/t2/queue_onset_Ahyb_mix_clean_E25.csv
E25 (ongoing) needs no rebuild: family A v2 and the family B noloc ongoing member are trained on raw labels (no imputer).
This onset is mandatory under the rule, so the CSV is written whatever the CV says; the CV is reported, not gated.

allin queue.yaml: component onset, runtime cpu, producer tfb.t2b.hyb_clean:produce_onset
standalone: PYTHONPATH=src python -m tfb.t2b.hyb_clean --results DIR [--params JSON]
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import lightgbm as lgb
from .. import cache, io
from ..t1 import hybrid as T1H, imputer_clean as IC
from . import hybrid as H, hyb_relabel as HR
from .postproc import atomic_json

STAGE = "hyb_clean"
MODE = "mix_clean"
CSV = f"queue_onset_Ahyb_{MODE}_E25.csv"   # = tfb.allin.policy.CLEAN_QUEUE
DEFAULTS = dict(n_trees=4000, rounds=20000, seed=42, e25_w=0.25, min_gain=0.004)


def write_labels(f: Path, panel: str, pn, imp, pred):
    """One panel of the tfb.t1.hybrid parquet layout (panel, timestamp, link_id, speed_hybrid, is_imputed)."""
    import pyarrow as pa, pyarrow.parquet as pq
    T, L = imp.shape
    ts = (np.repeat(pn.dates, io.STEPS_PER_DAY).astype(object) + "T"
          + np.tile([f"{m // 60:02d}:{m % 60:02d}:00Z" for m in range(0, 1440, 5)], len(pn.dates)).astype(object))
    t = pa.table({"panel": pa.array([panel] * (T * L)).dictionary_encode(),
                  "timestamp": pa.array(np.repeat(ts, L)).dictionary_encode(),
                  "link_id": pa.array(np.tile(np.asarray(pn.links, object), T)).dictionary_encode(),
                  "speed_hybrid": np.where(imp, pred, pn.speed).astype(np.float32).ravel(), "is_imputed": imp.ravel()})
    pq.write_table(t, f.with_suffix(".tmp")); f.with_suffix(".tmp").replace(f)


def labels(model: Path, n_trees: int, out: Path, beat, log=print, panels=None) -> str:
    """Train hybrid speed: released truth on eligible cells, clean-imputer speed on every other cell (as tfb.t1.hybrid.build)."""
    out.mkdir(parents=True, exist_ok=True); booster = lgb.Booster(model_file=str(model))
    for panel in panels or T1H.PANELS:
        f = out / f"{panel}.parquet"
        if f.exists():
            continue
        pn = cache.load(panel, "train"); imp = ~pn.elig
        pred = T1H.impute([booster], T1H.hidden_layer(pn, imp), pn, None, imp, (n_trees,))[n_trees]
        write_labels(f, panel, pn, imp, pred)
        vf = np.broadcast_to(pn.v_f, imp.shape)
        log(f"[hyb_clean] labels {panel}: imputed {int(imp.sum())}, queued share imputed {(pred[imp] <= 0.6 * vf[imp]).mean():.4f}")
        beat(dict(step="labels", panel=panel)); del pn
    return str(out)


def pipeline(d: Path, params=None, beat=lambda *a, **k: None, force=False, log=print) -> dict:
    p = dict(DEFAULTS, **(params or {})); d = Path(d); d.mkdir(parents=True, exist_ok=True)
    H.HYB_MIX = d / "train_hybrid_speed_clean"   # mode "mix_clean" reads its labels here (as hyb_relabel's "mix")

    def step(name, fn):
        f = d / f"{name}.json"
        if f.exists() and not force:
            return json.loads(f.read_text())
        v = fn(); atomic_json(f, v); beat(dict(step=name, done=True)); return v
    r = dict(stage=STAGE, runtime="cpu", params=p, mode=MODE)
    r["imputer"] = step("imputer", lambda: dict(path=str(IC.fit(d / "imputer", rounds=p["rounds"], seed=p["seed"]))))
    r["labels"] = step("labels", lambda: dict(path=labels(Path(r["imputer"]["path"]), p["n_trees"], H.HYB_MIX, beat, log)))
    r["cv"] = step("cv", lambda: HR.cv_files(p, d, beat, log, mode=MODE, conds=("onset",)))
    r["final"] = step("final_onset", lambda: HR.final("onset", p, d, beat, log, mode=MODE))
    keep(d)
    atomic_json(d / "result.json", r)
    return r


def pickles():
    return [H.OUT / f"cv_{f}_{MODE}.pkl" for f in ("janfeb", "octnov")] + [H.OUT / f"final_{MODE}.pkl"]


def keep(d: Path):
    """Durable copies of the VM-local pickles that march_refit / asm read (outputs/ is VM disk on Colab)."""
    (d / "pkl").mkdir(exist_ok=True)
    for f in pickles():
        shutil.copy2(f, d / "pkl" / f.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", type=Path, default=io.OUTPUTS / "allin")
    ap.add_argument("--params", default="{}"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); print(json.dumps(pipeline(a.results / STAGE, json.loads(a.params), force=a.force), indent=1, default=str))
