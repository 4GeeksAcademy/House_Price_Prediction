from __future__ import annotations

from datetime import UTC, datetime

from house_price_prediction.application.services.data_orchestration_service import (
    DataOrchestrationLayer,
)
from house_price_prediction.domain.contracts.prediction_contracts import (
    ProviderResponseContract,
)


def test_landmark_like_geocoder_match_boosts_structural_features() -> None:
    base_payload = {
        "OverallQual": 6,
        "GrLivArea": 2420,
        "LotArea": 11799,
        "FullBath": 1,
        "BedroomAbvGr": 3,
        "TotRmsAbvGrd": 7,
        "GarageCars": 1,
        "GarageArea": 280,
        "Fireplaces": 1,
        "ViewScore": 0,
        "feature_provenance": {"strategy": "heuristic"},
    }
    geocoding_response = ProviderResponseContract(
        provider_name="nominatim_geocoding",
        status="success",
        payload={
            "result": {
                "type": "government",
                "addresstype": "office",
                "importance": 0.6958339719610435,
                "name": "White House",
                "display_name": "White House, 1600, Pennsylvania Avenue Northwest, Washington, DC, 20500, United States",
            }
        },
        fetched_at=datetime.now(UTC),
    )

    adjusted = DataOrchestrationLayer._apply_geocoding_feature_overrides(
        base_payload,
        geocoding_response,
    )

    assert adjusted is not base_payload
    assert adjusted["PropertyType"] == "luxury"
    assert adjusted["OverallQual"] >= 9
    assert adjusted["GrLivArea"] >= 6000
    assert adjusted["LotArea"] >= 15000
    assert adjusted["FullBath"] >= 3
    assert adjusted["BedroomAbvGr"] >= 4
    assert adjusted["TotRmsAbvGrd"] >= 10
    assert adjusted["GarageCars"] >= 3
    assert adjusted["GarageArea"] >= 900
    assert adjusted["Fireplaces"] >= 2
    assert adjusted["ViewScore"] >= 2.0
    assert adjusted["feature_provenance"]["geocoding_adjustment"]["strategy"] == "landmark_like_geocoder_override"
    assert base_payload["OverallQual"] == 6


def test_regular_house_geocoder_match_is_left_unchanged() -> None:
    base_payload = {
        "OverallQual": 6,
        "GrLivArea": 1860,
        "LotArea": 7077,
        "FullBath": 2,
        "BedroomAbvGr": 4,
        "TotRmsAbvGrd": 7,
        "GarageCars": 2,
        "GarageArea": 480,
        "Fireplaces": 1,
        "ViewScore": 0,
    }
    geocoding_response = ProviderResponseContract(
        provider_name="nominatim_geocoding",
        status="success",
        payload={
            "result": {
                "type": "house",
                "addresstype": "place",
                "importance": 0.0000772983901731444,
                "name": "",
                "display_name": "2707, California Avenue Southwest, Seattle, WA, 98116, United States",
            }
        },
        fetched_at=datetime.now(UTC),
    )

    adjusted = DataOrchestrationLayer._apply_geocoding_feature_overrides(
        base_payload,
        geocoding_response,
    )

    assert adjusted == base_payload
