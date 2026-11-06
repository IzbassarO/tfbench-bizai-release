"""Task 4: split-local projection of the weak prior onto the released link counts.

f = argmin ||f - b||^2  s.t.  A_m f = c,  f >= 0   (A_m = incidence rows of counted links)
Solved as a bounded least squares on [w*A_m; I] f ~ [w*c; b] with a large weight w (near-exact
equality), checked by S_link. Fallback: NNLS on [A_m; sqrt(lam) I] with lam=1e-6.
usage: python -m tfb.t4.project  -> outputs/t4/path_flow_submission.csv (+ .meta.json)
"""
from __future__ import annotations
import json, time
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from .. import io

COLS = ["panel", "departure_time", "path_id", "origin_zone", "destination_zone", "path_flow"]


def operator(panel: str):
    nd = io.panel_dir(panel) / "network"
    paths = pd.read_csv(nd / "path_set.csv", dtype=str)
    inc = pd.read_csv(nd / "path_link_incidence.csv", dtype=str)
    pids = paths.path_id.tolist(); ppos = {p: i for i, p in enumerate(pids)}
    lids = inc.link_id.drop_duplicates().tolist(); lpos = {l: i for i, l in enumerate(lids)}
    A = np.zeros((len(lids), len(pids)))
    A[inc.link_id.map(lpos).to_numpy(), inc.path_id.map(ppos).to_numpy()] = 1.0
    return paths, pids, lids, A


def solve(A: np.ndarray, c: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, str]:
    """Exact Euclidean projection via the dual: f = max(b + A^T lam, 0), A f = c.
    Semi-smooth Newton on lam (dimension = number of counted links)."""
    m, n = A.shape
    lam = np.zeros(m)
    best, best_err = None, np.inf
    for it in range(200):
        f = np.maximum(b + A.T @ lam, 0.0)
        g = A @ f - c
        err = np.abs(g).sum() / max(c.sum(), 1e-9)
        if err < best_err:
            best, best_err = f.copy(), err
        if err < 1e-10:
            break
        S = (b + A.T @ lam) > 0
        J = A[:, S] @ A[:, S].T + 1e-9 * np.eye(m)
        step = np.linalg.solve(J, -g)
        # damped line search on the residual
        t = 1.0
        for _ in range(30):
            f2 = np.maximum(b + A.T @ (lam + t * step), 0.0)
            if np.abs(A @ f2 - c).sum() < np.abs(g).sum():
                break
            t *= 0.5
        lam = lam + t * step
    f = best
    s_link = 1 - best_err
    if s_link < 0.9999:
        lam_r = 1e-6
        f2, _ = nnls(np.vstack([A, np.sqrt(lam_r) * np.eye(n)]), np.concatenate([c, np.sqrt(lam_r) * b]), maxiter=50 * n)
        s2 = 1 - np.abs(A @ f2 - c).sum() / max(c.sum(), 1e-9)
        if s2 > s_link:
            return f2, f"nnls_lam1e-6 (dual S_link={s_link:.6f})"
    return f, f"dual_newton({it + 1} it)"


def run(splits=("validation", "private"), out=io.OUTPUTS / "t4" / "path_flow_submission.csv"):
    frames, meta = [], {}
    for panel in io.panels():
        paths, pids, lids, A = operator(panel)
        lpos = {l: i for i, l in enumerate(lids)}
        for split in splits:
            d = io.t4_dir(panel, split)
            counts = pd.read_csv(d / "synthetic_link_counts.csv", dtype={"link_id": str})
            prior = pd.read_csv(d / "synthetic_weak_prior.csv", dtype=str)
            tpl = pd.read_csv(d / "sample_submission_path_flow.csv", dtype=str)
            assert set(tpl.path_id) == set(pids) and len(tpl) == len(pids), f"{panel}/{split}: template/path_set mismatch"
            counts = counts[counts.link_id.isin(lpos)]
            Am = A[[lpos[l] for l in counts.link_id]]; c = counts["count"].to_numpy(float)
            b = prior.set_index("path_id").reindex(pids).path_flow.astype(float).fillna(0).to_numpy()
            t0 = time.time(); f, how = solve(Am, c, b); dt = time.time() - t0
            s_link = 1 - np.abs(Am @ f - c).sum() / c.sum()
            dev = np.abs(f - b).sum()
            assert np.isfinite(f).all() and (f >= 0).all()
            o = tpl[["panel", "departure_time", "path_id", "origin_zone", "destination_zone"]].copy()
            assert o.departure_time.nunique() == 1 and o.departure_time.iloc[0] == prior.departure_time.iloc[0]
            o["path_flow"] = pd.Series(f, index=pids).reindex(o.path_id).to_numpy()
            # zones must match path_set
            z = paths.set_index("path_id").reindex(o.path_id)
            assert (z.origin_zone.to_numpy() == o.origin_zone.to_numpy()).all() and (z.destination_zone.to_numpy() == o.destination_zone.to_numpy()).all()
            frames.append(o[COLS])
            meta[f"{panel}/{split}"] = {"n_paths": len(pids), "n_counted_links": len(c), "S_link": float(s_link),
                                        "D_pred": float(dev), "D_rel": float(dev / b.sum()), "solver": how, "sec": round(dt, 1),
                                        "f_sum": float(f.sum()), "b_sum": float(b.sum()), "zero_paths": int((f < 1e-9).sum())}
            print(panel, split, f"S_link={s_link:.6f} D/sum(b)={dev / b.sum():.4f} {how} {dt:.1f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(out, index=False)
    out.with_suffix(".meta.json").write_text(json.dumps({"method": "split-local projection min||f-b|| s.t. Af=c, f>=0", "panels": meta}, indent=2))
    return meta


if __name__ == "__main__":
    run()
