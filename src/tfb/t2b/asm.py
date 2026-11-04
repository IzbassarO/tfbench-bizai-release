"""ASM propagation features (Treiber & Helbing adaptive smoothing) for the family B noloc members, plus their producers.

Causal by construction: only the 13 released slots T-60..T of a window (B["S"]). For a target (x_j, T + 5h min) and
a wave speed c, every past slot k contributes the sensors near the characteristic's foot x_j - c (t* - t_k):
  V_c(j) = sum_ik phi_ik v_ik / sum_ik phi_ik,  phi_ik = exp(-|x_i - x_j + c dt_k| / SIGMA) exp(-age_k / TAU)
c = C_FREE (+80 km/h, information moves downstream) and C_CONG (-15 km/h, upstream). Treiber's congestion weight
  w = 0.5 (1 + tanh((V_crit - min(V_free, V_cong)) / dV)),  V_crit = 0.6 v_f, dV = 0.15 v_f
gives V = w V_cong + (1 - w) V_free. Features per h in (0, 3, 6) slots: asm_v, asm_w, asm_cg, asm_fr (ratios to v_f;
NaN where no sensor lies near the foot). Switched on with tfb.t2b.features.ASM["on"] (off: recipe features unchanged).

Producers (component onset|ongoing, runtime cpu; after hyb_clean): one factor vs the clean recipe.
  onset   0.5 v4_hyb(clean) + 0.5 noloc_hyb(clean, ASM)  vs  0.5 v4_hyb(clean) + 0.5 noloc_hyb(clean)
  ongoing 0.75 v2 + 0.25 noloc(raw, ASM)                  vs  E25
"""
from __future__ import annotations
import argparse, json, pickle
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from .. import io
from ..t2.decode import expected_iou_decode
from . import codex, cv, evaluate as ev, features as FT, hybrid as H, hyb_clean as HC, hyb_relabel as HR, run as R
from .postproc import atomic_json, write_cv

SIGMA, TAU = 0.6, 15.0           # km (Treiber), minutes of age preference for older slots
C_FREE, C_CONG = 80.0, -15.0     # km/h
HORIZONS = (0, 3, 6)             # slots after T
STAGE = "asm"
OG = "queue_ongoing"
_K = {}


def kernel(x: np.ndarray, c: float, h: int) -> np.ndarray:
    """[13 * L, L] weights from (slot k, sensor i) to target link j (k-major rows)."""
    key = (x.tobytes(), c, h)
    if key not in _K:
        age = 5.0 * (12 - np.arange(13))                        # minutes before T
        dt = (5.0 * h + age) / 60.0                             # hours from slot k to the target time
        d = x[None, :, None] - x[None, None, :] + c * dt[:, None, None]   # [k, i, j]: x_i - (x_j - c dt)
        _K.clear(); _K[key] = (np.exp(-np.abs(d) / SIGMA) * np.exp(-age / TAU)[:, None, None]).reshape(-1, len(x))
    return _K[key]


def features(B: dict, net: dict, st: dict) -> dict:
    S = np.asarray(B["S"], np.float64); n, K, L = S.shape; assert K == 13
    obs = np.isfinite(S).reshape(n, -1).astype(np.float64); val = np.nan_to_num(S).reshape(n, -1)
    vf = net["v_f"]; f = {}
    for h in HORIZONS:
        V = {}
        for name, c in (("fr", C_FREE), ("cg", C_CONG)):
            Kc = kernel(st["x"], c, h); den = obs @ Kc
            with np.errstate(invalid="ignore", divide="ignore"):
                V[name] = np.where(den > 1e-3, (val @ Kc) / den, np.nan)
        lo = np.fmin(V["fr"], V["cg"])
        w = 0.5 * (1 + np.tanh((0.6 * vf - lo) / (0.15 * vf)))
        both = np.where(np.isnan(V["cg"]), V["fr"], np.where(np.isnan(V["fr"]), V["cg"], w * V["cg"] + (1 - w) * V["fr"]))
        f[f"asm_v{h}"] = both / vf; f[f"asm_w{h}"] = np.where(np.isnan(lo), np.nan, w)
        f[f"asm_cg{h}"] = V["cg"] / vf; f[f"asm_fr{h}"] = V["fr"] / vf
    return {k: v.astype(np.float32) for k, v in f.items()}


@contextmanager
def enabled():
    FT.ASM["on"] = True
    try:
        yield
    finally:
        FT.ASM["on"] = False


def _dec(P):
    return np.stack([expected_iou_decode(x, force_nonempty=True) for x in P]).astype(bool)


def cv_files(p: dict, d: Path, beat, log=print) -> dict:
    """Both folds, both conditions: ASM noloc vs the same noloc without ASM, family A members fixed."""
    from .probavg import _fam
    files, gains, rows = {"onset": [], "ongoing": []}, {"onset": {}, "ongoing": {}}, []
    for fold, (fit_m, _) in cv.FOLDS.items():
        Zc = pickle.load(open(H.OUT / f"cv_{fold}_{HC.MODE}.pkl", "rb"))
        with enabled():
            nla = H.noloc_codex(fold, H.noloc_fit(fit_m, f"cache_hyb_{HC.MODE}", log=log))
            oga = H.noloc_codex(fold, H.noloc_fit(fit_m, "cache", log=log, cond=OG, extra=False), cond=OG, h=None)
        og0 = H.noloc_codex(fold, H.noloc_fit(fit_m, "cache", log=log, cond=OG, extra=False), cond=OG, h=None)
        w = p["e25_w"]
        for q in io.T2_PANELS:
            B = codex.windows(fold, q); wid = B["meta"].window_id.to_numpy()
            mk = (B["meta"].condition == "queue_onset").to_numpy(); P = np.zeros(B["Y"][mk].shape)
            P[:, 5] = 0.5 * Zc["v4h"][q] + 0.5 * nla[q]; ref = H.dec30(0.5 * Zc["v4h"][q] + 0.5 * Zc["nlh"][q])
            files["onset"].append(write_cv(d, "onset", fold, q, P, B["Y"][mk], B["E"][mk], ref, wid[mk]))
            rows += [dict(comp="onset", fold=fold, panel=q, model=k, iou=x) for k, X in (("asm", H.dec30(P[:, 5])), ("ref", ref))
                     for x in ev.window_iou(X, B["Y"][mk], B["E"][mk])]
            mk = (B["meta"].condition == OG).to_numpy(); v2 = B["codex"]["probability_v2"][mk]
            P = (1 - w) * v2 + w * oga[q]; ref = _dec((1 - w) * v2 + w * og0[q])
            files["ongoing"].append(write_cv(d, "ongoing", fold, q, P, B["Y"][mk], B["E"][mk], ref, wid[mk]))
            rows += [dict(comp="ongoing", fold=fold, panel=q, model=k, iou=x) for k, X in (("asm", _dec(P)), ("ref", ref))
                     for x in ev.window_iou(X, B["Y"][mk], B["E"][mk])]
        beat(dict(step="cv", fold=fold), check_budget=False)
    r = pd.DataFrame(rows)
    for (comp, fold), g in r.groupby(["comp", "fold"]):
        gains[comp][fold] = float(_fam(g[g.model == "asm"]) - _fam(g[g.model == "ref"]))
    log(f"[asm] cv gains {gains}")
    return dict(files=files, gains=gains)


def final(comp: str, p: dict, d: Path, hc: Path, beat, log=print) -> str:
    """Released CSV: hyb_clean's CSV with this condition's noloc member refitted with ASM."""
    src = HC.CSV   # on VM disk via HC.restore (from the hyb_clean_onset record's Drive copy)
    if comp == "onset":
        with enabled():
            nl = H.released_noloc(H.noloc_fit(R.ALL_MONTHS, f"cache_hyb_{HC.MODE}", log=log))
        F = pickle.load(open(H.OUT / f"final_{HC.MODE}.pkl", "rb")); on = {}
        for q in io.T2_PANELS:
            for s in ("validation", "private"):
                wi = pd.read_csv(io.t2_dir(q, s) / "window_index.csv", dtype=str); wo = wi[wi.condition == "queue_onset"]
                on[(q, s)] = 0.5 * H.long_to_grid(F["v4h"], q, s, wo) + 0.5 * nl[(q, s)]
        H.write(on, "../t2/queue_onset_A_hyb_clean_asm.csv")
        out = "queue_onset_Ahyb_clean_asm_E25.csv"; H.splice("queue_onset_A_hyb_clean_asm.csv", src, out)
    else:
        with enabled():
            nl = H.released_noloc(H.noloc_fit(R.ALL_MONTHS, "cache", log=log, cond=OG, extra=False), cond=OG, h=None)
        out = HR.write_ongoing(nl, src, "queue_onset_Ahyb_clean_E25asm.csv", p["e25_w"])
    dst = d / out; dst.write_bytes((io.OUTPUTS / "t2" / out).read_bytes()); beat(dict(step=f"final_{comp}"))
    return str(dst)


def pipeline(d: Path, params=None, beat=lambda *a, **k: None, force=False, log=print, comps=("onset", "ongoing")) -> dict:
    p = {"e25_w": 0.25, **(params or {})}; d = Path(d); d.mkdir(parents=True, exist_ok=True)
    hc = d.parent / HC.STAGE
    if not (hc / "result.json").exists():
        raise FileNotFoundError(f"asm needs tfb.t2b.hyb_clean first ({hc})")
    HC.restore(hc, log)

    def step(name, fn):
        f = d / f"{name}.json"
        if f.exists() and not force:
            return json.loads(f.read_text())
        v = fn(); atomic_json(f, v); beat(dict(step=name, done=True)); return v
    r = dict(stage=STAGE, runtime="cpu", params=p, sigma=SIGMA, tau=TAU, c_free=C_FREE, c_cong=C_CONG)
    r["cv"] = step("cv", lambda: cv_files(p, d, beat, log))
    r["final"] = {c: step(f"final_{c}", lambda c=c: final(c, p, d, hc, beat, log)) for c in comps}
    atomic_json(d / "result.json", r)
    return r


def _produce(comp, config, ctx):
    r = pipeline(Path(ctx.store.root) / STAGE, config.get("params"), ctx.heartbeat, comps=(comp,))
    return dict(cv_files=r["cv"]["files"][comp], test_predictions=r["final"][comp], asm_cv=r["cv"]["gains"][comp])


def produce_onset(config, ctx):
    return _produce("onset", config, ctx)


def produce_ongoing(config, ctx):
    return _produce("ongoing", config, ctx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", type=Path, default=io.OUTPUTS / "allin")
    ap.add_argument("--params", default="{}"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); print(json.dumps(pipeline(a.results / STAGE, json.loads(a.params), force=a.force), indent=1, default=str))
