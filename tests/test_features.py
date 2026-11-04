"""Behavioral checks for causal queue features and physical units."""
from __future__ import annotations

import numpy as np
import pytest

from tfb.t2.features import build_features, location_feature_mask


def _inputs(n_times: int = 25, n_links: int = 8) -> dict:
    rng = np.random.default_rng(20261023)
    speed = rng.uniform(70, 110, size=(n_times, n_links))
    return {
        "speed": speed,
        "flow": rng.uniform(1000, 4500, size=speed.shape),
        "occupancy": rng.uniform(0.02, 0.3, size=speed.shape),
        "origin": 12,
        "vf": np.full(n_links, 100.0),
        "capacity": np.full(n_links, 6000.0),
        "kcrit": np.full(n_links, 60.0),
        "lanes": np.full(n_links, 3.0),
        "distance_km": np.arange(n_links) * 0.5,
        "timestamps": np.datetime64("2031-01-06T07:00", "m") + np.arange(n_times) * np.timedelta64(5, "m"),
        "ramp_on": rng.uniform(20, 300, size=speed.shape),
        "location_prior": rng.uniform(0, 1, size=(288, n_links)),
        "panel_id": 2,
    }


def test_overwriting_every_future_input_leaves_features_bit_identical():
    inputs = _inputs()
    original, names = build_features(**inputs)
    origin = inputs["origin"]
    changed = dict(inputs)
    for key in ("speed", "flow", "occupancy", "ramp_on"):
        values = inputs[key].copy()
        values[origin + 1:] = -1e20
        values[origin + 2::2] = np.nan
        changed[key] = values
    times = inputs["timestamps"].copy()
    times[origin + 1:] = np.datetime64("NaT", "m")
    changed["timestamps"] = times
    after, after_names = build_features(**changed)
    assert names == after_names
    np.testing.assert_array_equal(original, after)
    truncated = dict(inputs)
    for key in ("speed", "flow", "occupancy", "ramp_on", "timestamps"):
        truncated[key] = inputs[key][:origin + 1]
    final, _ = build_features(**truncated)
    np.testing.assert_array_equal(original, final)


def test_row_at_origin_is_used_and_rows_are_horizon_major():
    inputs = _inputs()
    inputs["speed"][:] = 100
    inputs["speed"][12, 3] = 40
    result, names = build_features(**inputs)
    n_links = inputs["speed"].shape[1]
    assert result.shape[0] == n_links * 6
    assert result.dtype == np.float32
    assert len(names) == len(set(names)) == result.shape[1]
    np.testing.assert_array_equal(
        result[:, names.index("horizon_minutes")], np.repeat(np.arange(5, 31, 5), n_links),
    )
    assert result[3, names.index("queue_now")] == 1
    assert result[3, names.index("speed_ratio_lag0")] == pytest.approx(0.4)
    assert result[3, names.index("speed_ratio_lag5")] == pytest.approx(1.0)


def test_total_flow_capacity_and_density_are_not_divided_by_lanes_twice():
    inputs = _inputs()
    inputs["speed"][:] = 50
    inputs["flow"][:] = 3000
    result, names = build_features(**inputs)
    np.testing.assert_allclose(result[:, names.index("flow_ratio_lag0")], 0.5)
    np.testing.assert_allclose(result[:, names.index("density_ratio_lag0")], 1.0)


def test_neighbors_do_not_wrap_at_corridor_boundaries():
    inputs = _inputs()
    inputs["speed"][:] = np.arange(8)[None, :] * 10 + 20
    result, names = build_features(**inputs)
    assert np.isnan(result[0, names.index("up1_speed_ratio_now")])
    assert np.isnan(result[7, names.index("down1_speed_ratio_now")])
    assert result[2, names.index("down3_speed_ratio_now")] == pytest.approx(0.7)
    assert result[6, names.index("up5_speed_ratio_now")] == pytest.approx(0.3)


def test_queue_components_distances_and_shockwave_use_physical_units():
    inputs = _inputs()
    inputs["speed"][:] = 100
    inputs["flow"][:] = 4000
    inputs["speed"][:, 3:6] = 40
    inputs["flow"][:, 3:6] = 2000
    result, names = build_features(**inputs)
    # Upstream side: q=4000, k=40; queued side: q=2000, k=50.
    # (4000-2000)/(40-50)=-200 km/h, clipped to -40 for robustness.
    assert result[3, names.index("queue_tail_wave_15_kmh")] == -40
    assert result[3, names.index("queue_tail_distance_km")] == 0
    assert result[2, names.index("queue_tail_distance_km")] == -0.5
    assert result[5, names.index("queue_head_distance_km")] == 0
    assert result[6, names.index("queue_head_distance_km")] == 0.5
    assert result[3, names.index("queue_component_length_km")] == 1.0
    assert result[3, names.index("queue_tail_velocity_15_kmh")] == 0
    assert result[3, names.index("queue_motion_forecast_inside")] == 1
    assert result[2, names.index("queue_motion_forecast_inside")] == 0


def test_missing_readings_remain_missing_and_use_only_past_observation():
    inputs = _inputs()
    inputs["speed"][:] = 100
    inputs["speed"][11, 2] = 40
    inputs["speed"][12, 2] = np.nan
    inputs["speed"][:, 4] = np.nan
    inputs["ramp_on"][:] = np.nan
    result, names = build_features(**inputs)
    assert np.isnan(result[2, names.index("speed_ratio_lag0")])
    assert result[2, names.index("speed_ratio_last_observed")] == pytest.approx(0.4)
    assert result[2, names.index("speed_observation_age_minutes")] == 5
    assert result[2, names.index("queue_now")] == 1
    assert np.isnan(result[4, names.index("speed_ratio_last_observed")])
    assert np.isnan(result[4, names.index("speed_observation_age_minutes")])
    assert np.isnan(result[:, names.index("ramp_on_ratio_now")]).all()
    assert (result[:, names.index("ramp_on_observed_fraction")] == 0).all()
    assert not np.isinf(result).any()


def test_no_location_member_removes_direct_identifiers_and_historical_priors():
    inputs = _inputs()
    result, names = build_features(**inputs)
    changed = dict(inputs)
    changed["panel_id"] = 7
    changed["distance_km"] = inputs["distance_km"] + 50
    changed["location_prior"] = 1 - inputs["location_prior"]
    other, names2 = build_features(**changed)
    assert names == names2
    keep = ~location_feature_mask(names)
    np.testing.assert_allclose(result[:, keep], other[:, keep], equal_nan=True, atol=1e-6)
    assert not keep[names.index("location_link_index")]
    assert not keep[names.index("location_panel_id")]
    assert not keep[names.index("location_queue_prior_target")]
    assert keep[names.index("queue_tail_distance_km")]
    assert keep[names.index("free_speed_kmh")]


def test_early_window_and_missing_optional_inputs_have_stable_schema():
    inputs = _inputs()
    full, full_names = build_features(**inputs)
    inputs.update(origin=0, occupancy=None, ramp_on=None, location_prior=None)
    early, early_names = build_features(**inputs)
    assert early_names == full_names
    assert early.shape == full.shape
    assert not np.isinf(early).any()


def test_midnight_calendar_rollover_and_target_prior():
    inputs = _inputs()
    inputs["timestamps"] = np.datetime64("2031-01-05T22:50", "m") + np.arange(25) * np.timedelta64(5, "m")
    inputs["location_prior"] = np.broadcast_to(np.arange(288)[:, None], (288, 8)) / 288
    result, names = build_features(**inputs)
    # Origin is Sunday 23:50; T+5 Sunday, T+10 Monday.
    assert result[0, names.index("origin_day_of_week")] == 6
    assert result[0, names.index("forecast_day_of_week")] == 6
    assert result[8, names.index("forecast_day_of_week")] == 0
    assert result[0, names.index("location_queue_prior_target")] == pytest.approx(287 / 288)
    assert result[8, names.index("location_queue_prior_target")] == 0
