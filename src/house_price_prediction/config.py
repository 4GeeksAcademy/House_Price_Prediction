
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _get_bool_env(key: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    value = os.getenv(key, str(default)).strip().lower()
    return value in ("true", "1", "yes", "on")


def _parse_feature_policy_state_overrides(env_str: str) -> dict[str, str]:
    """Parse FEATURE_POLICY_STATE_OVERRIDES from env string."""
    if not env_str or not env_str.strip():
        return {}
    try:
        pairs = env_str.split(",")
        result = {}
        for pair in pairs:
            state, policy = pair.split("=", 1)
            result[state.strip()] = policy.strip()
        return result
    except (ValueError, AttributeError):
        return {}


@dataclass(frozen=True)
class Settings:
    raw_data_path: Path
    target_column: str
    model_path: Path
    test_size: float
    random_state: int
    app_name: str
    app_env: str
    api_host: str
    api_port: int
    database_url: str
    model_name: str
    model_version: str
    enable_mock_predictor: bool
    property_data_provider: str
    geocoding_provider: str
    prediction_reuse_max_age_hours: int
    provider_timeout_seconds: float
    provider_max_retries: int
    model_type: str = "lightgbm"
    provider_response_cache_max_age_hours: int = 24
    training_min_rows: int = 0
    feature_policy_name: str = "balanced-v1"
    feature_policy_version: str = "v1"
    feature_policy_state_overrides: dict[str, str] = field(
        default_factory=dict)
    walkscore_api_key: str = ""
    rentcast_api_key: str = ""
    rentcast_api_base_url: str = "https://api.rentcast.io/v1"
    neighborhood_scorer_path: Path = field(
        default_factory=lambda: Path("models/neighborhood_scorer.joblib"))


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load settings from environment variables with sensible defaults."""
    load_dotenv()

    return Settings(
        raw_data_path=Path(
            os.getenv("RAW_DATA_PATH", "data/raw/housing.csv")),
        target_column=os.getenv("TARGET_COLUMN", "SalePrice"),
        model_path=Path(
            os.getenv("MODEL_PATH", "models/nationwide_smart_router.joblib")),
        model_type=os.getenv("MODEL_TYPE", "lightgbm").strip().lower(),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        app_name=os.getenv("APP_NAME", "House Price Prediction API"),
        app_env=os.getenv("APP_ENV", "development"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        database_url=os.getenv(
            "DATABASE_URL", "sqlite:///data/processed/house_price_prediction.db"
        ),
        model_name=os.getenv("MODEL_NAME", "nationwide-smart-router"),
        model_version=os.getenv("MODEL_VERSION", "2.0.0"),
        enable_mock_predictor=_get_bool_env("ENABLE_MOCK_PREDICTOR", False),
        property_data_provider=os.getenv(
            "PROPERTY_DATA_PROVIDER", "free-fallback"),
        geocoding_provider=os.getenv("GEOCODING_PROVIDER", "free-fallback"),
        prediction_reuse_max_age_hours=int(
            os.getenv("PREDICTION_REUSE_MAX_AGE_HOURS", "24")),
        provider_response_cache_max_age_hours=int(
            os.getenv("PROVIDER_RESPONSE_CACHE_MAX_AGE_HOURS", "24")
        ),
        training_min_rows=int(os.getenv("TRAINING_MIN_ROWS", "0")),
        provider_timeout_seconds=float(
            os.getenv("PROVIDER_TIMEOUT_SECONDS", "25.0")),
        provider_max_retries=int(os.getenv("PROVIDER_MAX_RETRIES", "2")),
        feature_policy_name=os.getenv("FEATURE_POLICY_NAME", "balanced-v1"),
        feature_policy_version=os.getenv("FEATURE_POLICY_VERSION", "v1"),
        feature_policy_state_overrides=_parse_feature_policy_state_overrides(
            os.getenv("FEATURE_POLICY_STATE_OVERRIDES", "")
        ),
        walkscore_api_key=os.getenv("WALKSCORE_API_KEY", ""),
        rentcast_api_key=os.getenv("RENTCAST_API_KEY", ""),
        rentcast_api_base_url=os.getenv("RENTCAST_API_BASE_URL", "https://api.rentcast.io/v1"),
        neighborhood_scorer_path=Path(
            os.getenv("NEIGHBORHOOD_SCORER_PATH",
                      "models/neighborhood_scorer.joblib")
        ),
    )
