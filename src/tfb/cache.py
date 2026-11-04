"""Dense [time x link] tensors per (panel, split), cached as npz.
TFB_TESTMASK="share:seed" (tfb.t1.testmask only): load() hides that share of the OBSERVED cells of validation/private
(released data; 1-hour blocks per link) from the inputs, adds them to the targets and keeps their values as tm_speed /
tm_flow for scoring; nothing else changes."""
from __future__ import annotations
import os
import sys
import zlib
import numpy as np
import pandas as pd
from . import io

MAIN_COLS = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible"]


class Panel(dict):
    """dict with attribute access; arrays are [T, L] unless noted."""
    __getattr__ = dict.__getitem__


def _grid(files, date_pos, link_pos, T, L):
    sp = np.full((T, L), np.nan, np.float32); fl = sp.copy(); oc = sp.copy()
    pct = np.zeros((T, L), np.int8); el = np.zeros((T, L), bool)
    for f in files:
        d = pd.read_parquet(f, columns=MAIN_COLS)
        r = io.ts_index(d.timestamp, date_pos); c = d.link_id.astype(str).map(link_pos).to_numpy()
        sp[r, c] = d.speed_kmh.to_numpy(np.float32); fl[r, c] = d.flow_vph.to_numpy(np.float32)
        oc[r, c] = d.occupancy.to_numpy(np.float32); pct[r, c] = d.pct_observed.to_numpy(np.int16).clip(0, 100)
        el[r, c] = d.is_score_eligible.to_numpy(bool)
    return sp, fl, oc, pct, el


def build(panel: str, split: str) -> Panel:
    dates = io.split_dates(split); date_pos = {d: i for i, d in enumerate(dates)}
    T = len(dates) * io.STEPS_PER_DAY
    net = io.network(panel)
    first = pd.read_parquet(io.mainline_files(panel, split, True)[0], columns=["station_id", "link_id", "milepost"]).drop_duplicates("link_id")
    first = first.sort_values(["milepost", "link_id"])
    links = first.link_id.astype(str).to_numpy().astype(str); stations = first.station_id.astype(str).to_numpy().astype(str)
    link_pos = {l: i for i, l in enumerate(links)}; L = len(links)
    net = net.set_index("link_id").reindex(links)
    out = Panel(panel=panel, split=split, dates=np.array(dates, dtype=str), links=links, stations=stations,
                milepost=first.milepost.to_numpy(np.float32),
                lanes=net.lanes.to_numpy(np.float32), v_f=net.free_speed_kmh.to_numpy(np.float32),
                cap=net.capacity_vph.to_numpy(np.float32), k_crit=net.k_crit.to_numpy(np.float32),
                k_jam=net.k_jam.to_numpy(np.float32), length=net.length_km.to_numpy(np.float32))
    # masked layer + regime per day
    mfiles = io.mainline_files(panel, split, True)
    regime = np.full(len(dates), -1, np.int8)
    for f in mfiles:
        reg = f.parent.name.split("=")[1]; day = f.stem[-10:].replace("_", "-")
        regime[date_pos[day]] = io.REGIMES.index(reg)
    out["regime"] = regime
    out["mspeed"], out["mflow"], out["mocc"], out["mpct"], out["melig"] = _grid(mfiles, date_pos, link_pos, T, L)
    if split == "train":
        out["speed"], out["flow"], out["occ"], out["pct"], out["elig"] = _grid(io.mainline_files(panel, split, False), date_pos, link_pos, T, L)
    # Task 1 targets from the template
    tpl = io.t1_template(panel, split)
    tgt = np.zeros((T, L), bool)
    tgt[io.ts_index(tpl.timestamp, date_pos), tpl.link_id.map(link_pos).to_numpy()] = True
    out["target"] = tgt
    # ramps: flow NaN unless eligible (pct>=75 & finite); keep raw too
    rmap = io.ramp_map(panel)
    rids = rmap.ramp_link_id.to_numpy().astype(str); rpos = {r: i for i, r in enumerate(rids)}; R = len(rids)
    out["ramp_ids"] = rids; out["ramp_on"] = (rmap.ramp_type.str.upper().isin(["OR", "ON"])).to_numpy()
    out["ramp_link"] = rmap.nearest_mainline_link_id.map(link_pos).fillna(-1).to_numpy(int)
    rf = np.full((T, R), np.nan, np.float32); rraw = rf.copy(); rpct = np.zeros((T, R), np.int8)
    for f in io.ramp_files(panel, split):
        d = pd.read_parquet(f, columns=["timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible"])
        r = io.ts_index(d.timestamp, date_pos); c = d.ramp_link_id.astype(str).map(rpos).to_numpy()
        v = d.flow_vph.to_numpy(np.float32); ok = d.is_score_eligible.to_numpy(bool) & (d.pct_observed.to_numpy() >= 75) & np.isfinite(v)
        rraw[r, c] = v; rf[r, c] = np.where(ok, v, np.nan); rpct[r, c] = d.pct_observed.to_numpy(np.int16).clip(0, 100)
    out["rflow"] = rf; out["rflow_raw"] = rraw; out["rpct"] = rpct
    return out


def path(panel: str, split: str):
    return io.CACHE / f"{panel}_{split}.npz"


def load(panel: str, split: str) -> Panel:
    p = path(panel, split)
    if not p.exists():
        io.CACHE.mkdir(parents=True, exist_ok=True)
        pn = build(panel, split)
        np.savez(p, **{k: v for k, v in pn.items()})
        return pn
    z = np.load(p, allow_pickle=False)
    pn = Panel({k: z[k] for k in z.files})
    pn["panel"] = str(pn["panel"]); pn["split"] = str(pn["split"])
    tm = os.environ.get("TFB_TESTMASK")
    return test_mask(pn, *map(float, tm.split(":"))) if tm and split in ("validation", "private") else pn


def test_mask(pn: Panel, share: float, seed: float) -> Panel:
    obs = pn.melig & np.isfinite(pn.mspeed) & np.isfinite(pn.mflow) & ~pn.target
    T, L = obs.shape; rng = np.random.default_rng([int(seed), zlib.crc32(f"{pn.panel}/{pn.split}".encode())])
    hide = np.repeat(rng.random((T // 12 + 1, L)) < share, 12, 0)[:T] & obs
    pn = Panel(pn, tm_hidden=hide, tm_speed=np.where(hide, pn.mspeed, np.nan).astype(np.float32),
               tm_flow=np.where(hide, pn.mflow, np.nan).astype(np.float32))
    for k in ("mspeed", "mflow", "mocc"): pn[k] = np.where(hide, np.nan, pn[k]).astype(np.float32)
    pn["mpct"] = np.where(hide, 0, pn.mpct).astype(pn.mpct.dtype); pn["melig"] = pn.melig & ~hide; pn["target"] = pn.target | hide
    return pn


if __name__ == "__main__":
    ps = sys.argv[1:] or io.panels()
    for p in ps:
        for s in io.SPLITS:
            if not path(p, s).exists():
                load(p, s); print("cached", p, s, flush=True)
