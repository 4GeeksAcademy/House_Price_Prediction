
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    raw_data_path: Path
    target_column: str
    model_path: Path
    model_type: str
    test_size: float
    random_state: int


def load_settings() -> Settings:
    """Load settings from environment variables with sensible defaults."""
    load_dotenv()

    return Settings(
        raw_data_path=Path(
            os.getenv("RAW_DATA_PATH", "data/raw/housing.csv")),
        target_column=os.getenv("TARGET_COLUMN", "PRICE"),
        model_path=Path(
            os.getenv("MODEL_PATH", "models/house_price_model.pkl")),
        model_type=os.getenv("MODEL_TYPE", "lightgbm").strip().lower(),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
    )
