"""Paths, manifests and raw readers for the TrafficFlowBench release."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RELEASE = Path(os.environ.get("TFB_RELEASE", PROJECT / "data" / "kaggle_public"))
CACHE = Path(os.environ.get("TFB_CACHE", PROJECT / "cache"))
OUTPUTS = PROJECT / "outputs"
SPLITS = ("train", "validation", "private")
REGIMES = ("R1", "R2", "R3")
STEPS_PER_DAY = 288


def manifest() -> dict:
    return json.loads((RELEASE / "config" / "corridors.json").read_text())


def panels() -> list[str]:
    return [p["corridor_id"] for p in manifest()["panels"]]


def family_of() -> dict[str, str]:
    return {p["corridor_id"]: p["family_id"] for p in manifest()["panels"]}


T2_PANELS = ("D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S")


def panel_dir(panel: str) -> Path:
    return RELEASE / "corridors" / panel


def network(panel: str) -> pd.DataFrame:
    """Per-link FD parameters as the Task 3 scorer uses them (fd_parameters authoritative)."""
    nd = panel_dir(panel) / "network"
    links = pd.read_csv(nd / "links.csv", dtype={"link_id": str})
    fd = pd.read_csv(nd / "fd_parameters.csv", dtype={"link_id": str})
    fd = fd.groupby("link_id", as_index=False)[["free_speed_kmh", "capacity_vph", "lanes", "k_jam"]].mean()
    out = links[["link_id", "length_km"]].merge(fd, on="link_id", how="left")
    out["k_crit"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
    out["k_jam"] = np.maximum(out.k_jam.fillna(100.0 * out.lanes), out.k_crit * 1.05)
    out["v_cut"] = 0.6 * out.free_speed_kmh
    return out


def topology(panel: str) -> pd.DataFrame:
    return pd.read_csv(panel_dir(panel) / "network" / "lwr_mainline_topology.csv", dtype=str)


def ramp_map(panel: str) -> pd.DataFrame:
    return pd.read_csv(panel_dir(panel) / "network" / "ramp_attachment_map.csv", dtype=str)


def mainline_files(panel: str, split: str, masked: bool) -> list[Path]:
    sub = "mainline_states_masked" if masked else "mainline_states"
    return sorted((panel_dir(panel) / split / sub).glob("**/*.parquet"))


def ramp_files(panel: str, split: str) -> list[Path]:
    return sorted((panel_dir(panel) / split / "ramp_states").glob("**/*.parquet"))


def t1_template(panel: str, split: str) -> pd.DataFrame:
    return pd.read_csv(RELEASE / "task1" / panel / split / "sample_submission_state.csv", dtype=str)


def t2_dir(panel: str, split: str) -> Path:
    return RELEASE / "task2" / panel / split


def t4_dir(panel: str, split: str) -> Path:
    return RELEASE / "task4" / panel / split


def split_dates(split: str) -> list[str]:
    cal = json.loads((RELEASE / "config" / "synthetic_release_v1.json").read_text())["synthetic_calendar"][split]
    start = pd.Timestamp(cal["start"]); end = pd.Timestamp(cal["end_exclusive"])
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D", inclusive="left")]


def ts_index(ts: pd.Series, date_pos: dict[str, int]) -> np.ndarray:
    """Row index into the dense [day*288 + slot] axis from released timestamp text."""
    s = ts.astype(str)
    day = s.str[:10].map(date_pos).to_numpy()
    slot = s.str[11:13].astype(int).to_numpy() * 12 + s.str[14:16].astype(int).to_numpy() // 5
    return day * STEPS_PER_DAY + slot
