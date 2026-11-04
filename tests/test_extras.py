"""Causality and geometry checks for the separate slowdown feature schema."""
from __future__ import annotations

import numpy as np
import pytest

from tfb.t2.extras import build_short_extras


def _inputs():
    return dict(
        speed=np.full((25, 7), 100.0), flow=np.full((25, 7), 3000.0),
        timestamps=np.datetime64("2031-02-01T08:00", "m") + np.arange(25) * np.timedelta64(5, "m"),
        origin=12, vf=np.full(7, 100.0), capacity=np.full(7, 6000.0),
        distance_km=np.arange(7) * 0.5,
    )


def test_extras_ignore_all_future_values_and_timestamps():
    values = _inputs()
    before, names = build_short_extras(**values)
    for name in ("speed", "flow"):
        values[name][13:] = -1e10
    values["timestamps"][13:] = np.datetime64("NaT", "m")
    after, other_names = build_short_extras(**values)
    assert names == other_names
    np.testing.assert_array_equal(before, after)
    assert before.shape == (7, 30)
    assert before.dtype == np.float32
    assert len(set(names)) == len(names)


def test_filled_neighbor_values_are_causal_and_do_not_wrap():
    values = _inputs()
    values["speed"][11, 3] = 65
    values["speed"][12, 3] = np.nan
    values["flow"][10, 3] = 2000
    values["flow"][11:13, 3] = np.nan
    result, names = build_short_extras(**values)
    assert result[2, names.index("extra_down1_last_speed_ratio")] == pytest.approx(0.65)
    assert result[2, names.index("extra_down1_last_flow_ratio")] == pytest.approx(1 / 3)
    assert result[3, names.index("extra_last_speed_observation_age_minutes")] == 5
    assert np.isnan(result[0, names.index("extra_up1_last_speed_ratio")])
    assert np.isnan(result[6, names.index("extra_down1_last_speed_ratio")])


def test_slow_link_geometry_thresholds_direction_and_capacity_drop():
    values = _inputs()
    values["speed"][12, 2] = 70  # Equality counts at both thresholds.
    values["speed"][12, 5] = 80  # Equality counts at 0.8 only.
    values["capacity"][4:] = 4000
    result, names = build_short_extras(**values)
    assert result[3, names.index("extra_slow70_nearest_signed_km")] == -0.5
    assert result[3, names.index("extra_slow70_nearest_absolute_km")] == 0.5
    assert result[4, names.index("extra_slow80_nearest_signed_km")] == 0.5
    assert result[3, names.index("extra_slow80_upstream_distance_km")] == 0.5
    assert result[3, names.index("extra_slow80_downstream_distance_km")] == 1
    assert result[0, names.index("extra_slow70_count")] == 1
    assert result[0, names.index("extra_slow80_count")] == 2
    assert result[3, names.index("extra_capacity_ratio_down1")] == pytest.approx(2 / 3)
    assert result[4, names.index("extra_capacity_ratio_up1")] == pytest.approx(1.5)
    assert result[0, names.index("extra_corridor_min_speed_ratio")] == pytest.approx(0.7)


def test_no_slow_links_and_missing_columns_remain_explicitly_unknown():
    values = _inputs()
    values["speed"][:, 3] = np.nan
    result, names = build_short_extras(**values)
    assert np.isnan(result[:, names.index("extra_slow80_nearest_signed_km")]).all()
    assert (result[:, names.index("extra_slow80_count")] == 0).all()
    assert np.isnan(result[2, names.index("extra_down1_last_speed_ratio")])
    values["speed"][:] = np.nan
    result, _ = build_short_extras(**values)
    assert np.isnan(result[:, names.index("extra_corridor_min_speed_ratio")]).all()
    assert np.isnan(result[:, names.index("extra_corridor_speed_ratio_lower_quartile")]).all()
    assert not np.isinf(result).any()


def test_last_observation_search_does_not_reach_before_last_thirteen_rows():
    values = _inputs()
    values["origin"] = 24
    values["speed"][0:12, 1] = 30
    values["speed"][12:, 1] = np.nan
    result, names = build_short_extras(**values)
    assert np.isnan(result[0, names.index("extra_down1_last_speed_ratio")])
    assert result[0, names.index("extra_slow70_count")] == 0
