"""Feature masks and low-order model variants of the V4 onset family (explicit local traffic physics only: no location,
historical activation or static FD/geometry identity).
"""
from __future__ import annotations


import numpy as np



FEATURES = (
    "v4_flow_discharge_only_ratio_now", "v4_flow_discharge_only_ratio_slope30",
    "v4_flow_discharge_only_projected30", "v4_minutes_to_discharge_capacity",
    "v4_upstream_flow_discharge_only_ratio_now", "v4_upstream_arrival30_discharge_only_ratio",
    "v4_density_ratio_now", "v4_density_ratio_lag15", "v4_density_ratio_lag30",
    "v4_density_ratio_slope15", "v4_density_ratio_slope30",
    "v4_speed_ratio_now", "v4_speed_ratio_lag15", "v4_speed_ratio_lag30",
    "v4_speed_ratio_slope15", "v4_speed_ratio_slope30", "v4_speed_drop15",
    "v4_upstream3_density_ratio_now", "v4_upstream3_density_imbalance", "v4_upstream3_speed_ratio_now",
    "v4_downstream3_density_ratio_now", "v4_downstream3_density_imbalance", "v4_downstream3_speed_ratio_now",
    "v4_day_discharge_flow_peak_ratio", "v4_day_discharge_flow_peak120_ratio",
    "v4_day_minutes_since_discharge_near_capacity", "v4_day_slow70_observed_fraction",
    "v4_day_minutes_since_slow70", "v4_day_speed_minimum_ratio", "v4_day_density_peak_ratio",
    "ramp_on_ratio_now", "ramp_on_ratio_slope15", "ramp_on_ratio_slope30", "ramp_on_observed_fraction",
)
CONTRAST_FEATURES = tuple(f"v4_downstream{width}_discharge_{name}" for width in (5, 10)
    for name in ("capacity_contrast", "demand_stress", "bottleneck_stress", "demand_projected30"))
VARIANTS = {"stump500": (1, 500, 2), "stump1000": (1, 1000, 2),
            "depth2_500": (2, 500, 4), "depth2_1000": (2, 1000, 4),
            "depth4_100": (4, 100, 15), "depth4_200": (4, 200, 15),
            "depth6_100": (6, 100, 31), "contrast_depth4_100": (4, 100, 15),
            "contrast_depth6_100": (6, 100, 31), "fd_depth4_100": (4, 100, 15),
            "fd_depth6_100": (6, 100, 31), "balanced_contrast_depth6_100": (6, 100, 31),
            "balanced_fd_depth6_100": (6, 100, 31), "extra_contrast_depth6_100": (6, 100, 31),
            "extra_balanced_contrast_depth6_100": (6, 100, 31)}


def compact_mask(names, variant=None):
    """Shared full-schema mask: 34 base, 42 contrast, or 33 FD-dropout fields."""
    if len(names) != len(set(names)):
        raise ValueError("Compact onset schema contains duplicate names")
    family = variant.removeprefix("extra_").removeprefix("balanced_") if variant else ""
    features = FEATURES + (CONTRAST_FEATURES if family.startswith("contrast_") else ())
    if family.startswith("fd_"):
        features = tuple(n for n in features if n != "v4_day_minutes_since_discharge_near_capacity")
    missing = set(features) - set(names)
    if missing:
        raise ValueError(f"Compact onset physics fields missing: {sorted(missing)}")
    return np.array([name in features for name in names], bool)
