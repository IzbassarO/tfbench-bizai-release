"""Rebuild a final from its State member components (no training): state mix -> pp2 -> assembly.

  PYTHONPATH=src python -m tfb.finals v3|v1 --components DIR [--out DIR]
    DIR holds the four member components named in configs/finals.yaml (components:).
    -> <out>/state_raw_<final>.csv, state_pp2_<final>.csv, <name>.csv; prints each sha256 against the recorded one.
The mix is apply() below (v3: weights per traffic-regime bin) or two tfb.t1.blend calls (v1); pp2 is tfb.t3.apply, the
assembly tfb.submit.assemble with the final queue file.
"""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from . import cache, io
from .t1 import blend
from .t3.apply import main as pp2
from .submit.assemble import main as assemble

CFG = io.PROJECT / "configs" / "finals.yaml"
KEYS = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""): h.update(b)
    return h.hexdigest()


def states(S, F, vf, kc, ref_w):
    """Traffic-state coordinates of the reference mix's prediction (members S / F [n, >= 3], v_f / k_crit [n]):
    v = v / v_f, k = density (total flow / speed) / k_crit."""
    w = np.asarray(ref_w); v = S[:, :len(w)] @ w
    return dict(v=v / vf, k=(F[:, :len(w)] @ w) / np.maximum(v, 1.0) / kc)


def weights_at(spec, R, t):
    """[n, members] weights of target t ("sp" | "fl"): one row per bin of R[basis] (cuts)."""
    W = np.asarray(spec[t], np.float64); r = R[spec.get("basis", "v")]
    return W[np.digitize(r, spec["cuts"])]


def apply(spec, S, F, R):
    """Member speeds S / flows F [n, M], states R -> [n, 2] (speed, flow)."""
    ws, wf = weights_at(spec, R, "sp"), weights_at(spec, R, "fl"); m = ws.shape[1]
    return np.column_stack(((ws * S[:, :m]).sum(1), (wf * F[:, :m]).sum(1)))


def mix(spec: dict, paths: list[Path], out: Path):
    """Regime mix of the member components (template order): v_f / k_crit per (panel, link) from the cache network."""
    base = pd.read_csv(paths[0], dtype={k: str for k in KEYS}); S, F = [base.speed_kmh.to_numpy(float)], [base.flow_vph.to_numpy(float)]
    for p in paths[1:]:                                   # one extra frame in memory at a time
        f = pd.read_csv(p, dtype={k: str for k in KEYS})
        if not (f[KEYS].to_numpy() == base[KEYS].to_numpy()).all(): raise ValueError(f"{p}: row order differs")
        S.append(f.speed_kmh.to_numpy(float)); F.append(f.flow_vph.to_numpy(float)); del f
    net = {}
    for panel in base.panel.unique():
        pn = cache.load(panel, "validation"); net.update({(panel, str(l)): (float(v), float(k)) for l, v, k in zip(pn.links, pn.v_f, pn.k_crit)})
    v = pd.DataFrame.from_dict(net, orient="index").reindex(list(zip(base.panel, base.link_id))).to_numpy()
    if np.isnan(v).any(): raise ValueError(f"v_f missing for {int(np.isnan(v[:, 0]).sum())} rows")
    S, F = np.column_stack(S), np.column_stack(F)
    o = apply(spec, S, F, states(S, F, v[:, 0], v[:, 1], spec["ref_w"]))
    base["speed_kmh"] = o[:, 0].round(3); base["flow_vph"] = o[:, 1].round(2); base.to_csv(out, index=False)


def state_raw(f: dict, comp: dict, out: Path):
    if "mix" in f:
        mix(f["mix"], [comp[m] for m in f["mix"]["members"]], out); return
    b, c = f["blend"]["b"], f["blend"]["c"]; a = (1 - b) / 2
    ref = out.with_name(out.stem + "_3d90.csv")
    blend.main([f"{comp['lgbm']}@{a}", f"{comp['mlp']}@{a}", f"{comp['unet']}@{b}"], ref)
    blend.main([f"{ref}@{1 - c}", f"{comp['unet2']}@{c}"], out)


def build(final: str, components: Path, out: Path) -> Path:
    cfg = yaml.safe_load(CFG.read_text()); f = cfg[final]; out.mkdir(parents=True, exist_ok=True)
    comp = {k: Path(components) / n for k, n in cfg["components"].items()}
    raw, st = out / f"state_raw_{final}.csv", out / f"state_pp2_{final}.csv"
    state_raw(f, comp, raw); print(f"state_raw {sha(raw)} {'==' if sha(raw) == f['state_raw_sha256'] else '!='} recorded", flush=True)
    pp2(raw, st, cfg["pp2"], "", str(io.PROJECT / f["alphas"]))
    csv = assemble([f["name"], "--state", str(st), "--queue", str(io.PROJECT / f["queue"]), "--odme", str(io.PROJECT / cfg["odme"]),
                    "--out-dir", str(out), "--note", f"tfb.finals {final} rebuild from components"])
    h = sha(csv); print(f"{csv.name} {h} {'== RECORDED' if h == f['sha256'] else '!= recorded ' + f['sha256']}", flush=True)
    return csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("final", choices=["v3", "v1"]); ap.add_argument("--components", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=io.OUTPUTS / "finals" / "rebuild"); a = ap.parse_args()
    build(a.final, a.components, a.out)
