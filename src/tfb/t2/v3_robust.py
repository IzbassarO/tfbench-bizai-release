"""Feature mask without absolute identity and calendar columns.
"""
from __future__ import annotations


import numpy as np



STATIC = {"free_speed_kmh", "capacity_total_vph", "critical_density_total", "lanes",
          "spacing_up_km", "spacing_down_km"}
CALENDAR = {"origin_time_sin", "origin_time_cos", "origin_day_of_week", "origin_is_weekend",
            "forecast_time_sin", "forecast_time_cos", "forecast_day_of_week"}


def portable_mask(names):
    """Keep relative queue/traffic physics; discard absolute identity/calendar."""
    return np.array([not (n.startswith("location_") or n in STATIC | CALENDAR) for n in names])
