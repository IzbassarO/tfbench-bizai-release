"""Exactly the gate's 0.5 LGBM + 0.5 MLP, blend -> pp2 -> J.

Uses the existing scorer primitives, with precomputed ramp validity. Only template
target cells are stored in prediction files; scoring context is never a model input.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from tfb.eval.score_t1 import score_arrays
from tfb.t3.post import alpha_cells, fd_soft
from tfb.t3.proxy import fd_score, lwr_score


def context(bundle, hold, panel):
    """One panel's score context (truth, target cells, LGBM, network), alignment-checked."""
    with np.load(Path(bundle) / hold / "score" / f"{panel}.npz") as z:
        pn = SimpleNamespace(**{k: z[k] for k in z.files})
    pn.panel = panel
    rr,cc=np.nonzero(pn.target)
    if (not np.array_equal(rr,pn.target_row) or not np.array_equal(cc,pn.target_col)
            or len(pn.rows)!=len(pn.target) or np.any(np.diff(pn.rows)<=0)):
        raise ValueError(f'Score context target/row alignment mismatch: {panel}')
    return pn


def panel_score(pn, sp, fl, a):
    """pp2 with alphas a = {cong, free} on the full arrays, then (S_state mean, physics) of one panel."""
    al = alpha_cells(sp, pn, a, pn.elig & ~pn.target)
    sp, fl = fd_soft(sp, fl, pn.target, pn, al, "v")
    tab = score_arrays(pn, sp, fl)
    state = float(tab[tab.regime == "mean"].S_state.iloc[0])
    reg = np.repeat(pn.regime, 288); ph = []
    for k in range(3):
        rows = reg == k
        if not rows.any():
            continue
        sub = pn.target & rows[:, None]
        # proxy.evaluate restores truth outside each regime's targets.
        s, f = np.where(sub, sp, pn.speed), np.where(sub, fl, pn.flow)
        fd, lt50 = fd_score(pn, s, f, rows, sub)
        if lt50 > 0.2:
            raise ValueError(f"S_FD flow trap: {pn.panel}, regime {k}")
        ph.append((fd + 2 * lwr_score(pn, s, f, rows, pn.rok)) / 3)
    return state, float(np.mean(ph))


def panel_days(pn, sp, fl, a, w):
    """{absolute train day: w_state * S_state(day) + w_phys * physics(day)} of one panel after pp2 (alphas a): the
    per-day analogue of panel_score's J term (the validation suite's bootstrap / public-private / adversarial unit)."""
    al = alpha_cells(sp, pn, a, pn.elig & ~pn.target)
    sp, fl = fd_soft(sp, fl, pn.target, pn, al, "v")
    day = np.arange(len(sp)) // 288; out = {}
    for d in range(len(sp) // 288):
        rows = day == d; m = pn.target & rows[:, None]
        if not m.any(): continue
        rs = np.sqrt(np.mean((sp[m] - pn.speed[m]).astype(np.float64) ** 2))
        rq = np.sqrt(np.mean(((fl[m] - pn.flow[m]) / np.broadcast_to(pn.lanes, m.shape)[m]).astype(np.float64) ** 2))
        st = 0.54 * max(0.0, 1 - rs / 25) + 0.46 * max(0.0, 1 - rq / 600)
        s, f = np.where(m, sp, pn.speed), np.where(m, fl, pn.flow)
        ph = (fd_score(pn, s, f, rows, m)[0] + 2 * lwr_score(pn, s, f, rows, pn.rok)) / 3
        out[int(pn.rows[d * 288]) // 288] = w[0] * st + w[1] * ph
    return out


def weights(bundle, hold):
    """{panel: (w_state, w_phys)}: evaluate's J = sum over panels of w_state * S_state + w_phys * physics."""
    manifest = json.loads((Path(bundle) / "manifest.json").read_text())
    panels = manifest["holdouts"][hold]["panels"]; fam = pd.Series({p: manifest["family"][p] for p in panels})
    n = fam.map(fam.value_counts())
    return {p: (0.35 / (fam.nunique() * n[p]), 0.15 / len(panels)) for p in panels}


def evaluate(bundle, hold, predictions, alphas=None, tree_predictions=None, unet=None):
    """unet=(dir, b): the blended state becomes (1 - b) * blend + b * U-Net before pp2 (the ded6ce recipe)."""
    if not predictions:raise ValueError('At least one prediction run is required')
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    state, physics = [], []
    for panel in manifest["holdouts"][hold]["panels"]:
        pn = context(bundle, hold, panel)
        arrays = []
        for path in predictions:
            with np.load(Path(path) / f"{panel}.npz") as z:
                if not np.array_equal(z["row"], pn.rows[pn.target_row]) or not np.array_equal(z["col"], pn.target_col):
                    raise ValueError(f"Prediction alignment mismatch: {path}/{panel}")
                a = np.column_stack((z["sp"], z["fl"]))
                if a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError(f"Non-finite/non-float32 predictions: {path}/{panel}")
                arrays.append(a)
        pred = np.mean(arrays, axis=0)
        if tree_predictions:
            trees = []
            for path in tree_predictions:
                with np.load(Path(path) / f"{panel}.npz") as z:
                    if not np.array_equal(z["row"], pn.rows[pn.target_row]) or not np.array_equal(z["col"], pn.target_col):
                        raise ValueError("Tree prediction alignment mismatch")
                    tree = np.column_stack((z["sp"], z["fl"]))
                    if not np.isfinite(tree).all(): raise ValueError("Nonfinite tree predictions")
                    trees.append(tree)
            tree = np.mean(trees, axis=0)
            r, c = pn.target_row, pn.target_col
            pn.lgb_sp[r, c] = 0.5 * pn.lgb_sp[r, c] + 0.5 * tree[:, 0]
            pn.lgb_fl[r, c] = 0.5 * pn.lgb_fl[r, c] + 0.5 * tree[:, 1]
        sp, fl = pn.speed.copy(), pn.flow.copy()
        r, c = pn.target_row, pn.target_col
        sp[r, c] = 0.5 * pn.lgb_sp[r, c] + 0.5 * pred[:, 0]
        fl[r, c] = 0.5 * pn.lgb_fl[r, c] + 0.5 * pred[:, 1]
        if unet is not None:
            with np.load(Path(unet[0]) / f"{panel}.npz") as z:
                if not np.array_equal(z["row"], pn.rows[pn.target_row]) or not np.array_equal(z["col"], pn.target_col):
                    raise ValueError(f"U-Net alignment mismatch: {unet[0]}/{panel}")
                b = float(unet[1])
                sp[r, c] = (1 - b) * sp[r, c] + b * z["sp"]; fl[r, c] = (1 - b) * fl[r, c] + b * z["fl"]
        st, ph = panel_score(pn, sp, fl, (manifest["alphas"] if alphas is None else alphas)[panel])
        state.append(dict(panel=panel, family=manifest["family"][panel], value=st)); physics.append(ph)
    ss = float(pd.DataFrame(state).groupby("family").value.mean().mean())
    pp = float(np.mean(physics))
    j=0.35*ss+0.15*pp
    if not all(np.isfinite(v) and 0<=v<=1 for v in (ss,pp)) or not 0<=j<=.5:
        raise ValueError('Invalid score range; state/physics must be in [0,1], J in [0,.5]')
    return dict(S_state=ss, S_phys=pp, J=j)


def delta_table(rows):
    """Every precision/seed uses the same Colab config 0 fp32 seed 42 control."""
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    ref = tab[(tab.config_id == 0) & (tab.seeds == "42") & (tab.precision == "fp32")].set_index("hold").J
    tab["dJ"] = [r.J - ref.loc[r.hold] for r in tab.itertuples()]
    return tab
