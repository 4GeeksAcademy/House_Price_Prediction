import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMRegressor


@dataclass(frozen=True)
class ModelMetadata:
    feature_columns: tuple[str, ...]
    target_column: str | None
    model_name: str | None
    model_version: str | None


@dataclass(frozen=True)
class TrainedModelArtifact:
    model: Any
    metadata: ModelMetadata
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from .config import Settings
from .data import load_dataset, make_train_test_split, split_features_target
from .feature_schema import DEFAULT_PREDICTION_FEATURES
from .features import build_preprocessor


def train_and_save_model(settings: Settings) -> dict[str, float]:
    df = load_dataset(settings.raw_data_path)
    x, y = split_features_target(df, settings.target_column)

    # Restrict to the canonical feature schema so the trained artifact
    # always matches DEFAULT_PREDICTION_FEATURES regardless of extra
    # columns the source JSONL may carry (e.g. "Neighborhood" zip strings).
    schema_cols = [col for col in DEFAULT_PREDICTION_FEATURES if col in x.columns]
    x = x[schema_cols]

    x_train, x_test, y_train, y_test = make_train_test_split(
        x, y, test_size=settings.test_size, random_state=settings.random_state
    )

    preprocessor = build_preprocessor(x_train)
    regressor = LGBMRegressor(
        objective="fair",
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=95,
        min_child_samples=8,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.01,
        reg_lambda=8.0,
        n_jobs=-1,
        random_state=settings.random_state,
        verbose=-1,
    )

    transformed_regressor = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", transformed_regressor),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

    non_zero_mask = y_test != 0
    if non_zero_mask.any():
        mape = float(
            mean_absolute_percentage_error(
                y_test[non_zero_mask], predictions[non_zero_mask])
            * 100
        )
    else:
        mape = float("nan")

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mape": mape,
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.model_path)
    return metrics


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train first using scripts/train.py."
        )
    return joblib.load(model_path)


def load_model_artifact(model_path: Path) -> TrainedModelArtifact:
    model = load_model(model_path)
    return TrainedModelArtifact(
        model=model,
        metadata=ModelMetadata(
            feature_columns=tuple(),
            target_column=None,
            model_name="LightGBM",
            model_version="1.0",
        ),
    )
