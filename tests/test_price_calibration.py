from __future__ import annotations

from house_price_prediction.application.services.price_calibration import (
    _state_multiplier,
    apply_state_calibration,
)


ALL_STATE_ABBRS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


def test_state_multiplier_covers_all_50_states():
    multipliers = {_state_multiplier(state) for state in ALL_STATE_ABBRS}

    assert len(multipliers) > 1
    for state in ALL_STATE_ABBRS:
        multiplier = _state_multiplier(state)
        assert multiplier > 0.0
        assert multiplier != 1.0


def test_known_overrides_still_apply_for_calibrated_states():
    calibrated_price, multiplier = apply_state_calibration(
        raw_price=100000.0,
        state="CA",
        feature_source="census_context",
    )

    assert multiplier == 2.8
    assert calibrated_price == 280000.0


def test_state_baseline_uses_conservative_texas_uplift():
    calibrated_price, multiplier = apply_state_calibration(
        raw_price=100000.0,
        state="TX",
        feature_source="census_context",
    )

    assert multiplier == 1.1
    assert calibrated_price == 110000.0


def test_zip_level_market_overrides_apply_on_top_of_state_scaling():
    miami_price, miami_multiplier = apply_state_calibration(
        raw_price=100000.0,
        state="FL",
        feature_source="census_context",
        postal_code="33130",
    )
    chicago_price, chicago_multiplier = apply_state_calibration(
        raw_price=100000.0,
        state="IL",
        feature_source="census_context",
        postal_code="60601",
    )
    phoenix_price, phoenix_multiplier = apply_state_calibration(
        raw_price=100000.0,
        state="AZ",
        feature_source="census_context",
        postal_code="85020",
    )
    leander_price, leander_multiplier = apply_state_calibration(
        raw_price=100000.0,
        state="TX",
        feature_source="census_context",
        postal_code="78641",
    )

    assert round(miami_multiplier, 4) == 1.1572
    assert miami_price == 115720.0
    assert round(chicago_multiplier, 4) == 1.2662
    assert chicago_price == 126620.0
    assert round(phoenix_multiplier, 4) == 1.2018
    assert phoenix_price == 120176.0
    assert round(leander_multiplier, 4) == 3.41
    assert leander_price == 341000.0


def test_derived_fallback_is_used_for_states_without_explicit_override():
    calibrated_price, multiplier = apply_state_calibration(
        raw_price=100000.0,
        state="IA",
        feature_source="census_context",
    )

    assert multiplier != 1.0
    assert calibrated_price != 100000.0
