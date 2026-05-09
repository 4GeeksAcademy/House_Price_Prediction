from __future__ import annotations

from typing import Any

import pandas as pd

from house_price_prediction.config import Settings
from house_price_prediction.feature_schema import DEFAULT_PREDICTION_FEATURES
from house_price_prediction.model import TrainedModelArtifact, load_model_artifact


class ModelNotReadyError(RuntimeError):
    pass


class ModelInferenceError(RuntimeError):
    pass


class PredictionRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._artifact: TrainedModelArtifact | None = None

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def model_version(self) -> str:
        return self._settings.model_version

    def is_available(self) -> bool:
        return self._settings.enable_mock_predictor or self._settings.model_path.exists()

    def expected_feature_names(self) -> tuple[str, ...]:
        if self._settings.model_path.exists():
            metadata = self._load_artifact().metadata
            if metadata.feature_columns:
                return metadata.feature_columns
        return DEFAULT_PREDICTION_FEATURES

    def predict(self, features: dict[str, Any]) -> float:
        if self._settings.enable_mock_predictor:
            return self._mock_predict(features)

        if not self._settings.model_path.exists():
            raise ModelNotReadyError(
                f"Model file not found at {self._settings.model_path}. Train the model or enable the mock predictor."
            )

        model = self._load_artifact().model
        try:
            prediction = model.predict(pd.DataFrame([features]))
        except Exception as exc:
            raise ModelInferenceError(f"Prediction failed: {exc}") from exc
        return float(prediction[0])

    def _load_artifact(self) -> TrainedModelArtifact:
        if self._artifact is None:
            self._artifact = load_model_artifact(self._settings.model_path)
        return self._artifact

    @staticmethod
    def _mock_predict(features: dict[str, Any]) -> float:
        lot_area = float(features.get("LotArea", 8500))
        overall_qual = float(features.get("OverallQual", 6))
        overall_cond = float(features.get("OverallCond", 5))
        gr_liv_area = float(features.get("GrLivArea", 1500))
        garage_cars = float(features.get("GarageCars", 2))
        garage_area = float(features.get("GarageArea", 480))
        full_bath = float(features.get("FullBath", 2))
        half_bath = float(features.get("HalfBath", 0))
        total_rooms = float(features.get("TotRmsAbvGrd", 6))
        fireplaces = float(features.get("Fireplaces", 1))
        year_built = float(features.get("YearBuilt", 1995))
        year_remod_add = float(features.get("YearRemodAdd", year_built))
        house_style = str(features.get("HouseStyle", "1Story"))

        # Market-context features
        neighborhood_score = features.get("NeighborhoodScore")
        census_median_value = features.get("CensusMedianValue")
        median_income_k = features.get("MedianIncomeK")
        owner_occupied_rate = features.get("OwnerOccupiedRate")
        property_type = str(features.get("PropertyType", "single_family"))

        age_premium = max(year_built - 1950, 0) * 700
        remodel_premium = max(year_remod_add - year_built, 0) * 220
        house_style_adjustment = {
            "2Story": 6000,
            "SLvl": 3500,
        }.get(house_style, 0)

        # Market-context adjustments: included when the provider supplied the values.
        # Scaled so they contribute meaningfully but don't dominate structural features.
        # NeighborhoodScore is on a 0–100 scale (not 0–10); 50 is the neutral midpoint.
        # Range: score 0 → –$22,500, score 100 → +$22,500.
        neighborhood_score_adjustment = (
            (float(neighborhood_score) - 50.0) * 450.0
            if neighborhood_score is not None
            else 0.0
        )
        census_value_adjustment = (
            (float(census_median_value) - 200_000) * 0.12
            if census_median_value is not None
            else 0.0
        )
        income_adjustment = (
            (float(median_income_k) - 50.0) * 200
            if median_income_k is not None
            else 0.0
        )
        owner_rate_adjustment = (
            (float(owner_occupied_rate) - 0.6) * 8000
            if owner_occupied_rate is not None
            else 0.0
        )
        property_type_adjustment = {
            "luxury": 55000,
            "multifamily": 25000,
            "townhouse": 5000,
            "condo": -8000,
        }.get(property_type, 0)

        predicted_price = (
            50000
            + (lot_area * 2.3)
            + (overall_qual * 18000)
            + (overall_cond * 3200)
            + (gr_liv_area * 92)
            + (total_rooms * 1400)
            + (garage_cars * 9500)
            + (garage_area * 11)
            + (full_bath * 6500)
            + (half_bath * 2600)
            + (fireplaces * 4500)
            + age_premium
            + remodel_premium
            + house_style_adjustment
            + neighborhood_score_adjustment
            + census_value_adjustment
            + income_adjustment
            + owner_rate_adjustment
            + property_type_adjustment
        )
        return round(predicted_price, 2)