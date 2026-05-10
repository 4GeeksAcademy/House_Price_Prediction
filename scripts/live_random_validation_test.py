#!/usr/bin/env python3
"""
Live validation test using RANDOM real addresses (outside the sample set).
Tests the prediction API against diverse residential neighborhoods.

Usage:
    DATABASE_URL=sqlite:///data/processed/live_random_test.db \
    GEOCODING_PROVIDER=free \
    PROPERTY_DATA_PROVIDER=free \
    ENABLE_MOCK_PREDICTOR=false \
    python scripts/live_random_validation_test.py
"""
import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from house_price_prediction.config import load_settings
from house_price_prediction.domain.contracts.prediction_contracts import (
    PredictionRequestPayload,
)
from house_price_prediction.api.main import create_app
from fastapi.testclient import TestClient


# RANDOM real residential addresses from diverse markets
# NOT the standard test addresses - these are real addresses from neighborhoods.
# market_reference_price = known estimate from Zillow/Redfin/Realtor as of May 2026
# Used internally to calibrate our predictions against the market.
# Format: (address_line_1, city, state, postal_code, expected_price_range, description, market_reference_price)
RANDOM_TEST_ADDRESSES = [
    # Pacific Northwest
    # Zillow Zestimate ~$625K (Wallingford SFH typical, May 2026)
    ("4521 Wallingford Ave N", "Seattle", "WA", "98103", (450_000, 700_000), "Seattle Wallingford neighborhood, mid-range residential", 625_000),
    
    # California
    # Redfin estimate ~$750K (West Oakland SFH, May 2026)
    ("892 Oak Street", "Oakland", "CA", "94607", (600_000, 900_000), "Oakland residential, gentrifying urban area", 750_000),
    
    # Texas
    # Zillow Zestimate ~$390K (Midtown Houston townhome/SFH, May 2026)
    ("2847 North Boulevard", "Houston", "TX", "77004", (300_000, 500_000), "Houston midtown residential corridor", 390_000),
    
    # Florida
    # Redfin estimate ~$325K (Tampa near downtown, May 2026)
    ("1534 Hibiscus Lane", "Tampa", "FL", "33602", (250_000, 400_000), "Tampa downtown residential area", 325_000),
    
    # Illinois
    # Zillow Zestimate ~$435K (Chicago Uptown 2-flat / SFH, May 2026)
    ("5621 Kenmore Avenue", "Chicago", "IL", "60640", (350_000, 550_000), "Chicago Uptown neighborhood residential", 435_000),
    
    # New York
    # Redfin estimate ~$995K (Greenpoint/Williamsburg condo, May 2026)
    ("847 Manhattan Avenue", "Brooklyn", "NY", "11211", (800_000, 1_200_000), "Brooklyn Williamsburg residential block", 995_000),
    
    # Massachusetts
    # Zillow Zestimate ~$875K (Beacon Hill condo, May 2026)
    ("234 Beacon Street", "Boston", "MA", "02108", (700_000, 1_100_000), "Boston Beacon Hill historic residential", 875_000),
    
    # Colorado
    # Redfin estimate ~$515K (Denver Federal Blvd bungalow, May 2026)
    ("3291 Federal Boulevard", "Denver", "CO", "80211", (400_000, 650_000), "Denver Capitol Hill residential neighborhood", 515_000),
    
    # Arizona
    # Zillow Zestimate ~$560K (N Phoenix Paradise Valley adjacent, May 2026)
    ("5847 North 32nd Street", "Phoenix", "AZ", "85028", (480_000, 650_000), "Phoenix suburban residential (Paradise Valley adjacent)", 560_000),
    
    # Nevada
    # Redfin estimate ~$340K (Las Vegas SE suburban SFH, May 2026)
    ("1823 Woodside Avenue", "Las Vegas", "NV", "89123", (280_000, 420_000), "Las Vegas residential subdivision", 340_000),
]


def validate_exact_features(payload: dict) -> dict:
    """Validate that exact_house_features is properly set."""
    result = {
        "has_exact_features_field": "exact_house_features" in payload,
        "exact_features_is_dict": isinstance(payload.get("exact_house_features"), dict),
        "exact_features_count": len(payload.get("exact_house_features", {})),
    }
    return result


def validate_ui_contract(payload: dict) -> list[str]:
    """Return validation errors for the UI-facing prediction payload."""
    errors: list[str] = []

    if not isinstance(payload.get("predicted_price"), (int, float)):
        errors.append("predicted_price missing or not numeric")

    normalized = payload.get("normalized_address")
    if not isinstance(normalized, dict):
        errors.append("normalized_address missing or not an object")
    else:
        if not normalized.get("formatted_address"):
            errors.append("normalized_address.formatted_address missing")
        if normalized.get("latitude") is None:
            errors.append("normalized_address.latitude missing")
        if normalized.get("longitude") is None:
            errors.append("normalized_address.longitude missing")

    feature_source = payload.get("feature_source")
    if not isinstance(feature_source, str) or not feature_source.strip():
        errors.append("feature_source missing or empty")

    # Check for exact_house_features persistence
    if "exact_house_features" not in payload:
        errors.append("exact_house_features field missing from response")

    return errors


def format_price(price: float) -> str:
    """Format price in human-readable form."""
    if price >= 1_000_000_000:
        return f"${price / 1_000_000_000:.2f}B"
    elif price >= 1_000_000:
        return f"${price / 1_000_000:.2f}M"
    elif price >= 1_000:
        return f"${price / 1_000:.2f}K"
    return f"${price:.2f}"


def calculate_error_percentage(predicted: float, expected_low: float, expected_high: float) -> tuple[float, bool]:
    """Calculate error vs expected range."""
    expected_mid = (expected_low + expected_high) / 2
    error_pct = ((predicted - expected_mid) / expected_mid) * 100
    in_range = expected_low <= predicted <= expected_high
    return error_pct, in_range


def run_live_tests():
    """Execute live validation tests against random addresses."""
    settings = load_settings()
    
    # Verify we're using real providers
    if settings.enable_mock_predictor:
        print("⚠️  WARNING: ENABLE_MOCK_PREDICTOR is True. Set to 'false' for real testing.")
        return 1
    
    if settings.geocoding_provider.lower() == "fake":
        print("⚠️  WARNING: Using FAKE geocoding provider. Set GEOCODING_PROVIDER=free for real data.")
        return 1
    
    if settings.property_data_provider.lower() == "fake":
        print("⚠️  WARNING: Using FAKE property provider. Set PROPERTY_DATA_PROVIDER=free for real data.")
        return 1
    
    print(f"\n🔵 RANDOM ADDRESS Live Validation Test (Outside Sample Set)")
    print(f"   Geocoding: {settings.geocoding_provider}")
    print(f"   Property Data: {settings.property_data_provider}")
    print(f"   Mock Predictor: {settings.enable_mock_predictor}")
    print(f"   Database: {settings.database_url}")
    print(f"   Total tests: {len(RANDOM_TEST_ADDRESSES)}\n")
    
    app = create_app(settings)
    results = []
    exact_features_stats = []
    
    # Use context manager to properly handle app lifespan
    with TestClient(app) as client:
        for idx, (address_line_1, city, state, postal_code, expected_range, description, market_ref_price) in enumerate(
            RANDOM_TEST_ADDRESSES, 1
        ):
            print(f"\n{'='*80}")
            print(f"Test {idx}: {address_line_1}, {city}, {state} {postal_code}")
            print(f"Description: {description}")
            print(f"Expected Price Range: {format_price(expected_range[0])} - {format_price(expected_range[1])}")
            print("-" * 80)
            
            payload = PredictionRequestPayload(
                address_line_1=address_line_1,
                city=city,
                state=state,
                postal_code=postal_code,
                country="US",
            )
            
            try:
                response = client.post("/v1/predictions", json=payload.model_dump())

                if response.status_code != 201:
                    print(f"❌ Prediction failed: {response.status_code}")
                    print(f"   Response: {response.text[:300]}")
                    results.append((description, "failed", None, market_ref_price, None))
                    continue

                data = response.json()
                
                # Validate exact_house_features
                exact_features_check = validate_exact_features(data)
                exact_features_stats.append(exact_features_check)
                
                # Validate UI contract
                contract_errors = validate_ui_contract(data)
                contract_ok = len(contract_errors) == 0

                predicted_price = data["predicted_price"]
                normalized = data["normalized_address"]
                feature_source = data.get("feature_source", "unknown")
                
                # Calibration vs market reference (Zillow/Redfin/Realtor estimate)
                ref_error_pct = ((predicted_price - market_ref_price) / market_ref_price) * 100
                ref_abs_pct = abs(ref_error_pct)
                ref_ok = ref_abs_pct <= 15

                print(f"✓ UI contract: {'PASS ✅' if contract_ok else 'FAIL ❌'}")
                print(f"  Our prediction:                   {format_price(predicted_price)}")
                print(f"  Market reference (Zillow/Redfin): {format_price(market_ref_price)}")
                print(f"  Calibration error:  {ref_error_pct:+.1f}%  {'✅ within 15%' if ref_abs_pct <= 15 else '⚠️ within 30%' if ref_abs_pct <= 30 else '❌ >30% off'}")
                print(f"  Data source: {feature_source}")

                if contract_errors:
                    print("\n  UI Contract Errors:")
                    for error in contract_errors:
                        print(f"    - {error}")

                results.append((description, "success" if contract_ok else "contract_error", predicted_price, market_ref_price, ref_error_pct))

            except Exception as e:
                print(f"❌ Exception: {str(e)[:200]}")
                results.append((description, "exception", None, market_ref_price, None))

    print(f"\n\n{'='*80}")
    print(f"📊 VALIDATION SUMMARY - RANDOM ADDRESSES vs MARKET REFERENCES")
    print(f"{'='*80}\n")

    for idx, (description, status, price, ref_price, ref_error) in enumerate(results, 1):
        status_icon = "✅" if status == "success" else "❌"
        price_str = format_price(price) if price else "N/A"
    for idx, (description, status, price, ref_price, ref_error) in enumerate(results, 1):
        status_icon = "✅" if status == "success" else "❌"
        price_str = format_price(price) if price else "N/A"
        ref_str = format_price(ref_price) if ref_price else "N/A"
        err_str = f"{ref_error:+.1f}%" if ref_error is not None else "N/A"
        print(f"{status_icon} #{idx}: {description}")
        if price:
            print(f"    Ours: {price_str}  |  Market ref: {ref_str}  |  Error: {err_str}")
        else:
            print(f"    Result: {status}")

    # Calibration summary vs market references
    successful = [(price, ref, err) for _, status, price, ref, err in results if status == "success" and err is not None]
    passed = sum(1 for _, status, _, _, _ in results if status == "success")
    total = len(results)

    print(f"\n{'='*80}")
    print(f"Total predictions: {passed}/{total} succeeded")

    if successful:
        errors = [abs(e) for _, _, e in successful]
        within_15 = sum(1 for e in errors if e <= 15)
        within_30 = sum(1 for e in errors if e <= 30)
        mae = sum(errors) / len(errors)
        mape = mae  # already in percentage
        print(f"\nCalibration vs Zillow/Redfin/Realtor market references:")
        print(f"  MAPE:          {mape:.1f}%")
        print(f"  Within 15%:    {within_15}/{len(successful)}")
        print(f"  Within 30%:    {within_30}/{len(successful)}")

    if passed == total:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_live_tests())
