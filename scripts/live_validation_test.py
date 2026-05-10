#!/usr/bin/env python3
"""
Live validation test using real addresses and real data providers.
compares predictions against known market prices from major real estate sites.

Usage:
    DATABASE_URL=sqlite:///data/processed/live_test.db \
    GEOCODING_PROVIDER=free \
    PROPERTY_DATA_PROVIDER=free \
    ENABLE_MOCK_PREDICTOR=false \
    python scripts/live_validation_test.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from house_price_prediction.config import load_settings
from house_price_prediction.domain.contracts.prediction_contracts import (
    PredictionRequestPayload,
)
from house_price_prediction.api.main import create_app
from fastapi.testclient import TestClient


# Test addresses with known market prices from major RE sites
# Format: (address_line_1, city, state, postal_code, expected_price_range, source)
TEST_ADDRESSES = [
    # Typical residential addresses for accuracy validation
    (
        "1600 Pennsylvania Avenue NW",
        "Washington",
        "DC",
        "20500",
        (8_000_000, 12_000_000),
        "White House area, DC luxury neighborhood",
    ),
    (
        "123 Main Street",
        "Denver",
        "CO",
        "80202",
        (350_000, 550_000),
        "Downtown Denver residential area",
    ),
    (
        "2707 California Ave W",
        "Seattle",
        "WA",
        "98119",
        (400_000, 600_000),
        "Seattle residential neighborhood",
    ),
]


def validate_ui_contract(payload: dict, expected_reuse: bool | None = None) -> list[str]:
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

    if not isinstance(payload.get("key_features"), dict):
        errors.append("key_features missing or not an object")
    if not isinstance(payload.get("actual_house_features"), dict):
        errors.append("actual_house_features missing or not an object")

    feature_source = payload.get("feature_source")
    if not isinstance(feature_source, str) or not feature_source.strip():
        errors.append("feature_source missing or empty")

    feature_provenance = payload.get("feature_provenance")
    if feature_provenance is not None and not isinstance(feature_provenance, dict):
        errors.append("feature_provenance must be null or an object")

    if expected_reuse is not None and payload.get("was_reused") is not expected_reuse:
        errors.append(
            f"was_reused expected {expected_reuse} but got {payload.get('was_reused')}"
        )

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


def calculate_calibration_metrics(predicted: float, expected_low: float, expected_high: float) -> dict[str, float]:
    """Return simple calibration metrics against the midpoint of the expected range."""
    expected_mid = (expected_low + expected_high) / 2
    error_dollars = predicted - expected_mid
    abs_error_dollars = abs(error_dollars)
    error_pct = (error_dollars / expected_mid) * 100 if expected_mid else 0.0
    abs_error_pct = abs(error_pct)
    return {
        "expected_mid": expected_mid,
        "error_dollars": error_dollars,
        "abs_error_dollars": abs_error_dollars,
        "error_pct": error_pct,
        "abs_error_pct": abs_error_pct,
    }


def run_live_tests():
    """Execute live validation tests."""
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
    
    print(f"\n🔵 Live Validation Test with Real Providers")
    print(f"   Geocoding: {settings.geocoding_provider}")
    print(f"   Property Data: {settings.property_data_provider}")
    print(f"   Mock Predictor: {settings.enable_mock_predictor}")
    print(f"   Database: {settings.database_url}\n")
    
    app = create_app(settings)
    results = []
    
    # Use context manager to properly handle app lifespan
    with TestClient(app) as client:
        for idx, (address_line_1, city, state, postal_code, expected_range, description) in enumerate(
            TEST_ADDRESSES, 1
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
                first_response = client.post("/v1/predictions", json=payload.model_dump())

                if first_response.status_code != 201:
                    print(f"❌ Fresh prediction failed: {first_response.status_code}")
                    print(f"   Response: {first_response.text[:200]}")
                    results.append((description, "failed", None))
                    continue

                second_response = client.post("/v1/predictions", json=payload.model_dump())
                if second_response.status_code != 201:
                    print(f"❌ Reused prediction failed: {second_response.status_code}")
                    print(f"   Response: {second_response.text[:200]}")
                    results.append((description, "failed", None))
                    continue

                first_data = first_response.json()
                second_data = second_response.json()

                first_errors = validate_ui_contract(first_data, expected_reuse=False)
                second_errors = validate_ui_contract(second_data, expected_reuse=True)
                reuse_consistency_errors: list[str] = []
                if second_data.get("actual_house_features") != first_data.get("actual_house_features"):
                    reuse_consistency_errors.append("actual_house_features changed between fresh and reused responses")
                if second_data.get("feature_source") != first_data.get("feature_source"):
                    reuse_consistency_errors.append("feature_source changed between fresh and reused responses")
                if second_data.get("feature_provenance") != first_data.get("feature_provenance"):
                    reuse_consistency_errors.append("feature_provenance changed between fresh and reused responses")

                contract_errors = first_errors + second_errors + reuse_consistency_errors
                contract_ok = len(contract_errors) == 0

                predicted_price = second_data["predicted_price"]
                normalized = second_data["normalized_address"]
                features = second_data.get("actual_house_features", {})
                calibration = calculate_calibration_metrics(
                    first_data["predicted_price"], expected_range[0], expected_range[1]
                )
                error_pct, in_range = calculate_error_percentage(
                    predicted_price, expected_range[0], expected_range[1]
                )

                print(f"✓ Fresh response UI contract: {'PASS' if not first_errors else 'FAIL'}")
                print(f"✓ Reused response UI contract: {'PASS' if not second_errors else 'FAIL'}")
                print(f"✓ Reuse consistency: {'PASS' if not reuse_consistency_errors else 'FAIL'}")
                print(f"  Reused response price: {format_price(predicted_price)}")
                print(f"  Geocoded to: {normalized['formatted_address']}")
                print(f"  Lat/Lon: {normalized['latitude']:.4f}, {normalized['longitude']:.4f}")
                print(f"  Feature source: {second_data.get('feature_source')}")
                print(f"  Reused: {second_data.get('was_reused')}")
                print(f"  Actual feature count: {len(features)}")
                print(f"  Error vs expected range: {error_pct:+.1f}%  {'✅' if in_range else '⚠️'}")

                if contract_errors:
                    print("\n  UI Contract Errors:")
                    for error in contract_errors:
                        print(f"    - {error}")

                if features:
                    key_features = {
                        "Bedrooms": features.get("BedroomAbvGr"),
                        "Bathrooms": features.get("FullBath"),
                        "Sqft": features.get("GrLivArea"),
                        "Year Built": features.get("YearBuilt"),
                        "Lot Area": features.get("LotArea"),
                    }
                    print("\n  Features Used:")
                    for k, v in key_features.items():
                        if v is not None:
                            print(f"    - {k}: {v}")

                    census_features = {
                        "Census Median Value": features.get("CensusMedianValue"),
                        "Median Income (K)": features.get("MedianIncomeK"),
                        "Owner Occupied %": features.get("OwnerOccupiedRate"),
                    }
                    print("\n  Census Context:")
                    for k, v in census_features.items():
                        if v is not None:
                            if "%" in k:
                                print(f"    - {k}: {v*100:.1f}%")
                            elif "Value" in k:
                                print(f"    - {k}: {format_price(v)}")
                            else:
                                print(f"    - {k}: ${v:.1f}K")

                results.append(
                    (
                        description,
                        "success",
                        {
                            "predicted_price": predicted_price,
                            "fresh_predicted_price": first_data["predicted_price"],
                            "calibration": calibration,
                            "error_pct": error_pct,
                            "in_range": in_range,
                            "contract_ok": contract_ok,
                            "feature_source": second_data.get("feature_source"),
                            "feature_count": len(features),
                            "was_reused": second_data.get("was_reused"),
                        },
                    )
                )

            except Exception as e:
                import traceback
                print(f"❌ Error: {type(e).__name__}: {e}")
                traceback.print_exc()
                results.append((description, "error", str(e)))
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    range_passed = 0
    contract_passed = 0
    for description, status, data in results:
        if status == "success":
            predicted_price = data["predicted_price"]
            fresh_predicted_price = data["fresh_predicted_price"]
            error_pct = data["error_pct"]
            in_range = data["in_range"]
            contract_ok = data["contract_ok"]
            calibration = data["calibration"]
            icon = "✅" if contract_ok else "❌"
            print(f"{icon} {description}")
            print(
                "   "
                f"UI contract: {'PASS' if contract_ok else 'FAIL'}, "
                f"source: {data['feature_source']}, "
                f"features: {data['feature_count']}, "
                f"price: {format_price(predicted_price)}, "
                f"error: {error_pct:+.1f}%\n"
            )
            print(
                f"   Calibration (fresh): pred={format_price(fresh_predicted_price)}, "
                f"mid={format_price(calibration['expected_mid'])}, "
                f"bias={calibration['error_dollars']:+,.0f} ({calibration['error_pct']:+.1f}%), "
                f"abs={calibration['abs_error_dollars']:,.0f} ({calibration['abs_error_pct']:.1f}%)"
            )
            if in_range:
                range_passed += 1
            if contract_ok:
                contract_passed += 1
        elif status == "failed":
            print(f"❌ {description}")
            print(f"   Prediction endpoint returned error\n")
        else:
            print(f"❌ {description}")
            print(f"   Exception: {data}\n")
    
    print(f"\n{'='*80}")
    print(f"UI contract results: {contract_passed}/{len(results)} passed")
    print(f"Price range results: {range_passed}/{len(results)} in expected range")
    if results:
        fresh_abs_errors = [item[2]["calibration"]["abs_error_dollars"] for item in results if item[1] == "success"]
        fresh_abs_error_pcts = [item[2]["calibration"]["abs_error_pct"] for item in results if item[1] == "success"]
        fresh_biases = [item[2]["calibration"]["error_dollars"] for item in results if item[1] == "success"]
        if fresh_abs_errors:
            mean_abs_error = sum(fresh_abs_errors) / len(fresh_abs_errors)
            mean_abs_error_pct = sum(fresh_abs_error_pcts) / len(fresh_abs_error_pcts)
            mean_bias = sum(fresh_biases) / len(fresh_biases)
            print(
                f"Calibration summary: MAE=${mean_abs_error:,.0f}, "
                f"MAPE={mean_abs_error_pct:.1f}%, bias=${mean_bias:+,.0f}"
            )
    print(f"{'='*80}\n")
    
    return 0 if contract_passed == len(results) else 1



if __name__ == "__main__":
    sys.exit(run_live_tests())
