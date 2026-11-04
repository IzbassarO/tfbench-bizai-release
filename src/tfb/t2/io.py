"""Task-2-local release readers. Never reads a Task 1 reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SPLIT_DATES = {
    "train": ("2030-06-01", "2031-03-01"),
    "validation": ("2031-03-01", "2031-04-01"),
    "private": ("2031-04-01", "2031-05-01"),
}


@dataclass
class Network:
    panel: str
    link_ids: np.ndarray
    vf: np.ndarray
    capacity: np.ndarray
    kcrit: np.ndarray
    lanes: np.ndarray
    distance_km: np.ndarray
    length_km: np.ndarray

    def feature_kwargs(self) -> dict:
        return {k: getattr(self, k) for k in
                ("vf", "capacity", "kcrit", "lanes", "distance_km")}


@dataclass
class PanelArrays:
    timestamps: np.ndarray
    speed: np.ndarray
    flow: np.ndarray
    occupancy: np.ndarray
    eligible: np.ndarray
    ramp_on: np.ndarray


def resolve_root(root: Path | str) -> Path:
    root = Path(root).resolve()
    if (root / "kaggle_public" / "config").is_dir():
        root = root / "kaggle_public"
    if not (root / "config").is_dir():
        raise FileNotFoundError(f"Not a release root: {root}")
    return root


def panel_names(root: Path) -> list[str]:
    return sorted(p.name for p in (root / "task2").iterdir()
                  if p.is_dir() and not p.name.startswith("D12_I405"))


def read_network(root: Path, panel: str) -> Network:
    base = root / "corridors" / panel / "network"
    links = pd.read_csv(base / "links.csv", dtype={"link_id": str})
    fd = pd.read_csv(base / "fd_parameters.csv", dtype={"link_id": str})
    topology = pd.read_csv(base / "lwr_mainline_topology.csv")
    order = topology.sort_values("order_index").mainline_link_id.astype(str).tolist()
    order = [x for x in order if x in set(links.link_id)]
    order += [x for x in links.link_id if x not in set(order)]
    links = links.set_index("link_id").reindex(order)
    fd = fd.set_index("link_id").reindex(order)

    def column(names, default):
        for name in names:
            if name in fd:
                out = pd.to_numeric(fd[name], errors="coerce")
                if name in links:
                    out = out.fillna(pd.to_numeric(links[name], errors="coerce"))
                return out.fillna(default).to_numpy(dtype=np.float32)
            if name in links:
                return pd.to_numeric(links[name], errors="coerce").fillna(default).to_numpy(dtype=np.float32)
        return np.full(len(order), default, dtype=np.float32)

    vf = column(["v_f", "free_speed_kmh", "free_flow_speed_kmh"], 105.)
    if "v_cut" in fd:
        vf = fd.v_cut.to_numpy(dtype=np.float32) / .6
    lanes = column(["lanes"], 3.)
    capacity = column(["capacity_vph", "capacity"], 6000.)
    kcrit = column(["critical_density", "k_crit"], np.nan)
    kcrit = np.where(np.isfinite(kcrit), kcrit, capacity / vf).astype(np.float32)
    length = column(["length_km"], .5)
    distance = (np.cumsum(length) - length / 2).astype(np.float32)
    return Network(panel, np.array(order), vf, capacity, kcrit, lanes, distance, length)


def parquet_frame(path: Path, requested: list[str] | None = None) -> pd.DataFrame:
    file = pq.ParquetFile(path)
    columns = None if requested is None else [c for c in requested if c in file.schema.names]
    return file.read(columns=columns).to_pandas()


def _dense_assign(frame, arrays, start, link_lookup, *, ramp=False):
    stamp = pd.to_datetime(frame.timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[m]")
    ti = ((stamp - start) / np.timedelta64(5, "m")).astype(np.int64)
    li = frame.link_id.astype(str).map(link_lookup).to_numpy()
    good = pd.notna(li) & (ti >= 0) & (ti < len(arrays[0]))
    ti, li = ti[good], li[good].astype(np.int64)
    for column, dest in zip(["speed_kmh", "flow_vph", "occupancy", "is_score_eligible"], arrays):
        if column in frame:
            vals = frame[column].to_numpy()[good]
            dest[ti, li] = vals


def read_ramps(root: Path, network: Network, split: str, timestamps: np.ndarray,
               dates: set[str] | None = None) -> np.ndarray:
    """Aggregate available on-ramp evidence; unavailable sensors remain NaN."""
    result = np.full((len(timestamps), len(network.link_ids)), np.nan, np.float32)
    base = root / "corridors" / network.panel
    attach = pd.read_csv(base / "network" / "ramp_attachment_map.csv")
    # OR = on-ramp, FR = freeway exit ramp in this release.
    attach = attach[attach.ramp_type.astype(str).str.upper().isin(["OR", "ON", "ON_RAMP"])]
    mapping = dict(zip(attach.ramp_link_id.astype(str), attach.nearest_mainline_link_id.astype(str)))
    link_lookup = dict(zip(network.link_ids, range(len(network.link_ids))))
    for path in sorted((base / split / "ramp_states").rglob("*.parquet")):
        day = path.stem[-10:].replace("_", "-")
        if dates is not None and day not in dates:
            continue
        frame = parquet_frame(path)
        id_col = "link_id" if "link_id" in frame else "ramp_link_id"
        frame["mainline"] = frame[id_col].astype(str).map(mapping)
        valid = frame.mainline.notna() & pd.to_numeric(frame.flow_vph, errors="coerce").notna()
        if "is_score_eligible" in frame:
            valid &= frame.is_score_eligible.astype(bool)
        elif "pct_observed" in frame:
            valid &= frame.pct_observed.ge(75)
        frame = frame[valid]
        if frame.empty:
            continue
        grouped = frame.groupby(["timestamp", "mainline"], sort=False).flow_vph.sum(min_count=1).reset_index()
        ts = pd.to_datetime(grouped.timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[m]")
        ti = ((ts - timestamps[0]) / np.timedelta64(5, "m")).astype(int)
        li = grouped.mainline.map(link_lookup).to_numpy()
        valid = pd.notna(li) & (ti >= 0) & (ti < len(timestamps))
        result[ti[valid], li[valid].astype(int)] = grouped.flow_vph.to_numpy(dtype=np.float32)[valid]
    return result


def read_panel_arrays(root: Path, network: Network, split: str, cache_dir: Path,
                      *, masked=False, refresh=False, dates: set[str] | None = None) -> PanelArrays:
    tag = "masked" if masked else "raw"
    cache = cache_dir / f"{network.panel}_{split}_{tag}.npz"
    if dates is None and cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as d:
            return PanelArrays(**{k: d[k] for k in PanelArrays.__dataclass_fields__})
    start, end = [np.datetime64(x, "m") for x in SPLIT_DATES[split]]
    timestamps = np.arange(start, end, np.timedelta64(5, "m"))
    shape = (len(timestamps), len(network.link_ids))
    speed, flow, occupancy = [np.full(shape, np.nan, np.float32) for _ in range(3)]
    eligible = np.zeros(shape, bool)
    base = root / "corridors" / network.panel / split
    channel = "mainline_states_masked" if masked else "mainline_states"
    paths = sorted((base / channel).rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(base / channel)
    lookup = dict(zip(network.link_ids, range(len(network.link_ids))))
    requested = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "is_score_eligible"]
    for path in paths:
        day = path.stem[-10:].replace("_", "-")
        if dates is not None and day not in dates:
            continue
        frame = parquet_frame(path, requested)
        _dense_assign(frame, (speed, flow, occupancy, eligible), start, lookup)
    ramp_on = read_ramps(root, network, split, timestamps, dates)
    result = PanelArrays(timestamps, speed, flow, occupancy, eligible, ramp_on)
    if dates is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp.npz")
        np.savez(temporary, **{k: getattr(result, k) for k in PanelArrays.__dataclass_fields__})
        temporary.replace(cache)
    return result


def read_window_assets(root: Path, panel: str, split: str):
    base = root / "task2" / panel / split
    index = pd.read_csv(base / "window_index.csv")
    history = parquet_frame(base / "window_history.parquet")
    template = pd.read_csv(base / "sample_submission_queue.csv", dtype={"window_id": str, "link_id": str})
    return index, history, template


def current_observations(root: Path, network: Network, origins: np.ndarray, *, include_context=False,
                         released_history: PanelArrays | None = None):
    """Read masked train observations only at requested T; no future values used."""
    shape = (len(origins), len(network.link_ids))
    out = {name: np.full(shape, np.nan, np.float32) for name in ("speed_kmh", "flow_vph", "occupancy")}
    contexts = {}
    context_names = None
    lookup = dict(zip(network.link_ids, range(len(network.link_ids))))
    by_time = {str(np.datetime64(t, "m")): i for i, t in enumerate(origins)}
    days = {str(np.datetime64(t, "D")) for t in origins}
    paths = sorted((root / "corridors" / network.panel / "train" / "mainline_states_masked").rglob("*.parquet"))
    for path in paths:
        if path.stem[-10:].replace("_", "-") not in days:
            continue
        frame = parquet_frame(path, ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy"])
        stamp = pd.to_datetime(frame.timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[m]")
        ti = pd.Series(stamp.astype(str)).map(by_time).to_numpy()
        li = frame.link_id.astype(str).map(lookup).to_numpy()
        valid = pd.notna(ti) & pd.notna(li)
        for name in out:
            out[name][ti[valid].astype(int), li[valid].astype(int)] = frame[name].to_numpy(dtype=np.float32)[valid]
        if include_context:
            from .context import build_day_context
            day = np.datetime64(path.stem[-10:].replace("_", "-"), "m")
            day_times = day + np.arange(288) * np.timedelta64(5, "m")
            dv, dq, do = [np.full((288, len(network.link_ids)), np.nan, np.float32) for _ in range(3)]
            de = np.zeros(dv.shape, bool)
            _dense_assign(frame, (dv, dq, do, de), day, lookup)
            for index in np.flatnonzero(origins.astype("datetime64[D]") == day.astype("datetime64[D]")):
                origin_slot = int((origins[index] - day) / np.timedelta64(5, "m"))
                context_v, context_q = dv, dq
                if released_history is not None:
                    # A generated window would publish its unmasked history,
                    # just as the actual release protects every shipped history.
                    # Overlay only T-60..T-5; retain the masked legal row at T.
                    at = int((origins[index] - released_history.timestamps[0]) / np.timedelta64(5, "m"))
                    count = min(12, origin_slot, at)
                    context_v, context_q = dv.copy(), dq.copy()
                    context_v[origin_slot-count:origin_slot] = released_history.speed[at-count:at]
                    context_q[origin_slot-count:origin_slot] = released_history.flow[at-count:at]
                contexts[int(index)], context_names = build_day_context(context_v, context_q, day_times, origin=origin_slot, vf=network.vf)
    if include_context:
        if len(contexts) != len(origins):
            raise ValueError("Missing masked day files for full-day context")
        out["day_context"] = np.stack([contexts[i] for i in range(len(origins))])
        out["day_context_names"] = context_names
    return out


def history_arrays(history: pd.DataFrame, origin, network: Network, current: PanelArrays | None = None):
    """History [T-60,T) from the window plus the legal masked observation at T."""
    origin = pd.Timestamp(origin)
    origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
    start = origin.tz_localize(None).to_datetime64().astype("datetime64[m]") - np.timedelta64(60, "m")
    stamps = start + np.arange(13) * np.timedelta64(5, "m")
    shape = (13, len(network.link_ids))
    speed, flow, occupancy = [np.full(shape, np.nan, np.float32) for _ in range(3)]
    eligible = np.zeros(shape, bool)
    lookup = dict(zip(network.link_ids, range(len(network.link_ids))))
    # Guard against an accidental future row in a staged history file.
    time = pd.to_datetime(history.timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    origin_ns = origin.tz_localize(None).to_datetime64().astype("datetime64[ns]")
    history = history[(time >= origin_ns - np.timedelta64(60, "m")) & (time <= origin_ns)]
    _dense_assign(history, (speed, flow, occupancy, eligible), start, lookup)
    ramps = np.full(shape, np.nan, np.float32)
    if current is not None:
        ids = ((stamps - current.timestamps[0]) / np.timedelta64(5, "m")).astype(int)
        valid = (ids >= 0) & (ids < len(current.timestamps))
        if len(current.timestamps):
            valid &= current.timestamps[np.clip(ids, 0, len(current.timestamps) - 1)] == stamps
        ramps[valid] = current.ramp_on[ids[valid]]
        at = ids[-1]
        if 0 <= at < len(current.timestamps) and current.timestamps[at] == stamps[-1]:
            for dest, source in [(speed, current.speed), (flow, current.flow), (occupancy, current.occupancy)]:
                dest[-1] = source[at]
            eligible[-1] = current.eligible[at]
    return PanelArrays(stamps, speed, flow, occupancy, eligible, ramps)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
