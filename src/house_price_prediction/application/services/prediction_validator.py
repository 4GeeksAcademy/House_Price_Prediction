"""Validates predictions against census context to catch anomalies."""
from __future__ import annotations

from typing import TypedDict

from house_price_prediction.telemetry import get_logger

logger = get_logger(__name__)


class PredictionValidationResult(TypedDict):
    """Result of prediction validation."""
    is_valid: bool
    confidence_score: float  # 0.0 to 1.0
    validation_notes: list[str]
    anomaly_flags: list[str]


class PredictionValidator:
    """Validates predictions using Census context and statistical bounds."""

    # How far can prediction deviate from Census median before we flag it
    _CENSUS_MEDIAN_DEVIATION_THRESHOLD = 0.40  # 40%
    
    # Minimum acceptable confidence score (0-1)
    _MIN_CONFIDENCE_THRESHOLD = 0.50

    def validate_prediction(
        self,
        predicted_price: float,
        actual_house_features: dict,
        feature_source: str | None = None,
    ) -> PredictionValidationResult:
        """
        Validate a prediction against available context data.
        
        Returns:
            PredictionValidationResult with is_valid flag and confidence score.
            If is_valid=False, the prediction should not be shown to users.
        """
        notes: list[str] = []
        flags: list[str] = []
        confidence_score = 1.0

        # Check 1: Do we have Census context?
        census_median_value = actual_house_features.get("CensusMedianValue")
        has_census_data = census_median_value is not None and census_median_value > 0

        if not has_census_data:
            flags.append("no_census_context")
            confidence_score = 0.3
            notes.append("No Census context data available - prediction is based on fallback data only")
        else:
            # Check 2: Is prediction reasonable vs Census median?
            deviation_ratio = abs(predicted_price - census_median_value) / census_median_value
            
            if deviation_ratio > self._CENSUS_MEDIAN_DEVIATION_THRESHOLD:
                flags.append("large_census_deviation")
                confidence_score -= 0.35
                notes.append(
                    f"Prediction ${predicted_price:,.0f} deviates {deviation_ratio*100:.1f}% "
                    f"from Census median ${census_median_value:,.0f}"
                )
            elif deviation_ratio > 0.25:
                confidence_score -= 0.15
                notes.append(
                    f"Prediction ${predicted_price:,.0f} deviates {deviation_ratio*100:.1f}% "
                    f"from Census median ${census_median_value:,.0f}"
                )

        # Check 3: Data source quality
        if feature_source == "heuristic":
            flags.append("heuristic_data_source")
            confidence_score -= 0.40
            notes.append("Using estimated/heuristic property data (not from authoritative source)")
        elif feature_source == "census_context":
            confidence_score += 0.10  # Boost confidence for solid data source
            notes.append("Prediction based on authoritative Census data")

        # Check 4: Feature completeness
        required_features = ["BedroomAbvGr", "FullBath", "GrLivArea", "YearBuilt"]
        missing_features = [f for f in required_features if actual_house_features.get(f) is None]
        
        if missing_features:
            confidence_score -= (0.05 * len(missing_features))
            notes.append(f"Missing {len(missing_features)} key features: {', '.join(missing_features)}")

        # Final confidence score bounds
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Determine validity
        is_valid = confidence_score >= self._MIN_CONFIDENCE_THRESHOLD

        if not is_valid:
            logger.warning(
                "prediction_validation_failed "
                "predicted_price=%s confidence_score=%.2f flags=%s notes=%s",
                predicted_price,
                confidence_score,
                flags,
                notes,
            )
        
        return PredictionValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            validation_notes=notes,
            anomaly_flags=flags,
        )
