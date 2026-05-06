from __future__ import annotations

from uuid import uuid4

from house_price_prediction.application.services.feature_assembly_service import (
    FeatureAssemblyService,
)
from house_price_prediction.infrastructure.model_runtime.predictor import PredictionRuntime


def test_quality_first_policy_biases_completeness_toward_key_features():
    request_id = uuid4()
    expected_feature_names = ("OverallQual", "LotArea")
    payload = {
        "OverallQual": 7,
        "LotArea": None,
    }

    balanced_service = FeatureAssemblyService(
        model_name="house-price-random-forest",
        model_version="test-version",
        expected_feature_names=expected_feature_names,
        feature_policy_name="balanced-v1",
    )
    quality_first_service = FeatureAssemblyService(
        model_name="house-price-random-forest",
        model_version="test-version",
        expected_feature_names=expected_feature_names,
        feature_policy_name="quality-first-v1",
    )

    balanced_vector = balanced_service.assemble(request_id=request_id, provider_payload=payload)
    quality_first_vector = quality_first_service.assemble(request_id=request_id, provider_payload=payload)

    assert balanced_vector.completeness_score == 0.5
    assert quality_first_vector.completeness_score > balanced_vector.completeness_score
    assert quality_first_vector.feature_policy_name == "quality-first-v1"
    assert quality_first_vector.feature_policy_version == "v1"


def test_unknown_feature_policy_falls_back_to_balanced_weights():
    request_id = uuid4()
    expected_feature_names = ("OverallQual", "LotArea")
    payload = {
        "OverallQual": 7,
        "LotArea": None,
    }

    unknown_policy_service = FeatureAssemblyService(
        model_name="house-price-random-forest",
        model_version="test-version",
        expected_feature_names=expected_feature_names,
        feature_policy_name="future-experimental-policy",
    )

    vector = unknown_policy_service.assemble(request_id=request_id, provider_payload=payload)

    assert vector.completeness_score == 0.5
    assert vector.weight_total == 2.0
    assert vector.feature_policy_name == "future-experimental-policy"


def test_state_override_selects_policy_for_context():
    request_id = uuid4()
    expected_feature_names = ("OverallQual", "LotArea")
    payload = {
        "OverallQual": 7,
        "LotArea": None,
    }

    service = FeatureAssemblyService(
        model_name="house-price-random-forest",
        model_version="test-version",
        expected_feature_names=expected_feature_names,
        feature_policy_name="balanced-v1",
        feature_policy_state_overrides={"FL": "quality-first-v1"},
    )

    vector = service.assemble(
        request_id=request_id,
        provider_payload=payload,
        context={"state": "fl"},
    )

    assert vector.feature_policy_name == "quality-first-v1"
    assert vector.completeness_score > 0.5


def test_mock_predictor_uses_all_major_features():
    base_features = {
        "LotArea": 8500,
        "OverallQual": 6,
        "OverallCond": 5,
        "YearBuilt": 1990,
        "YearRemodAdd": 1995,
        "GrLivArea": 1500,
        "FullBath": 2,
        "HalfBath": 0,
        "BedroomAbvGr": 3,
        "TotRmsAbvGrd": 6,
        "Fireplaces": 0,
        "GarageCars": 2,
        "GarageArea": 480,
        "HouseStyle": "1Story",
        "NeighborhoodScore": 5.0,
        "CensusMedianValue": 200_000.0,
    }
    base_price = PredictionRuntime._mock_predict(base_features)

    feature_variants = {
        "OverallCond": 9,
        "YearRemodAdd": 2022,
        "HalfBath": 1,
        "TotRmsAbvGrd": 10,
        "GarageArea": 900,
        "NeighborhoodScore": 85.0,
        "HouseStyle": "2Story",
    }

    for feature_name, changed_value in feature_variants.items():
        variant = dict(base_features)
        variant[feature_name] = changed_value
        assert PredictionRuntime._mock_predict(variant) != base_price
