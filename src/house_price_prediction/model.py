from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import joblib
import numpy as np
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import Settings
from .data import load_dataset, make_train_test_split, split_features_target
from .feature_schema import DEFAULT_PREDICTION_FEATURES
from .features import build_preprocessor

# Prices above this percentile are ultra-luxury outliers that inflate RMSE and
# reduce accuracy on typical single-family homes.  ~99th percentile cut.
_OUTLIER_PRICE_CAP_PERCENTILE = 99.0


def train_and_save_model(settings: Settings) -> dict[str, float]:
    df = load_dataset(settings.raw_data_path)
    x, y = split_features_target(df, settings.target_column)

    # --- Outlier removal -------------------------------------------------------
    # Cap at p99 to exclude ultra-luxury properties.  These ~1% of rows have
    # outsized influence on the loss without representing the typical US home.
    price_cap = float(np.percentile(y, _OUTLIER_PRICE_CAP_PERCENTILE))
    cap_mask = y <= price_cap
    x, y = x[cap_mask].reset_index(drop=True), y[cap_mask].reset_index(drop=True)

    # Restrict to the canonical feature schema so the trained artifact
    # always matches DEFAULT_PREDICTION_FEATURES regardless of extra
    # columns the source JSONL may carry (e.g. "Neighborhood" zip strings).
    schema_cols = [col for col in DEFAULT_PREDICTION_FEATURES if col in x.columns]
    x = x[schema_cols]

    x_train, x_test, y_train, y_test = make_train_test_split(
        x, y, test_size=settings.test_size, random_state=settings.random_state
    )

    # --- Early stopping --------------------------------------------------------
    # Hold out 10 % of the training set to determine the optimal n_estimators.
    # We fit the preprocessor on the training sub-split only (no leakage), then
    # transform the val split for the LightGBM eval_set.  Once the ideal tree
    # count is found, the final model is retrained on the full training set.
    x_tr_es, x_val_es, y_tr_es, y_val_es = train_test_split(
        x_train, y_train, test_size=0.10, random_state=settings.random_state
    )

    pre_es = build_preprocessor(x_tr_es)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_tr_es_t = pre_es.fit_transform(x_tr_es)
        x_val_es_t = pre_es.transform(x_val_es)

    lgbm_es = LGBMRegressor(
        objective="huber",          # huber is robust to residual outliers
        alpha=0.9,                  # huber threshold (focus on typical homes)
        n_estimators=3000,          # high upper bound; early stopping controls actual count
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=20,
        min_split_gain=0.01,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=5.0,
        n_jobs=-1,
        random_state=settings.random_state,
        verbose=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm_es.fit(
            x_tr_es_t,
            np.log1p(y_tr_es),
            eval_set=[(x_val_es_t, np.log1p(y_val_es))],
            callbacks=[
                lgb_early_stopping(stopping_rounds=150, verbose=False),
                lgb_log_evaluation(period=0),
            ],
        )
    best_n = int(lgbm_es.best_iteration_)
    print(f"  [early-stopping] best n_estimators = {best_n}")

    # --- Final model on full training set with best n_estimators ---------------
    preprocessor = build_preprocessor(x_train)
    regressor = LGBMRegressor(
        objective="huber",
        alpha=0.9,
        n_estimators=best_n,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=20,
        min_split_gain=0.01,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=5.0,
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

    r2 = float(r2_score(y_test, predictions)) if len(y_test) >= 2 else float("nan")

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mape": mape,
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": r2,
        "best_n_estimators": float(best_n),
        "price_cap": price_cap,
    }

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model_artifact(
        model=model,
        model_path=settings.model_path,
        feature_columns=list(schema_cols),
        target_column=settings.target_column,
        model_name=settings.model_name,
        model_version=settings.model_version,
    )
    return metrics


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train first using scripts/train.py."
        )
    return joblib.load(model_path)


@dataclass(frozen=True)
class ModelArtifactMetadata:
    feature_columns: tuple[str, ...]
    target_column: str
    model_name: str
    model_version: str


@dataclass(frozen=True)
class TrainedModelArtifact:
    model: Any
    metadata: ModelArtifactMetadata


def save_model_artifact(
    model: Any,
    model_path: Path,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str,
    model_name: str,
    model_version: str,
) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = TrainedModelArtifact(
        model=model,
        metadata=ModelArtifactMetadata(
            feature_columns=tuple(feature_columns),
            target_column=target_column,
            model_name=model_name,
            model_version=model_version,
        ),
    )
    joblib.dump(artifact, model_path)


def load_model_artifact(model_path: Path) -> TrainedModelArtifact:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Train first using scripts/train.py."
        )
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    scripts_dir_str = str(scripts_dir)
    path_was_added = False
    if scripts_dir.exists() and scripts_dir_str not in sys.path:
        sys.path.insert(0, scripts_dir_str)
        path_was_added = True
    try:
        obj = joblib.load(model_path)
    finally:
        if path_was_added:
            try:
                sys.path.remove(scripts_dir_str)
            except ValueError:
                pass
    # Support legacy bare-model files (pre-artifact format)
    if not isinstance(obj, TrainedModelArtifact):
        if obj.__class__.__name__ == "SmartRouter" or hasattr(obj, "CHAMPION_PATHS"):
            return TrainedModelArtifact(
                model=obj,
                metadata=ModelArtifactMetadata(
                    feature_columns=(
                        "LotArea",
                        "OverallQual",
                        "OverallCond",
                        "YearBuilt",
                        "YearRemodAdd",
                        "GrLivArea",
                        "FullBath",
                        "HalfBath",
                        "BedroomAbvGr",
                        "TotRmsAbvGrd",
                        "Fireplaces",
                        "GarageCars",
                        "GarageArea",
                        "NeighborhoodScore",
                        "PropertyType",
                        "HouseStyle",
                        "Neighborhood",
                    ),
                    target_column="SalePrice",
                    model_name="nationwide-smart-router",
                    model_version="v1",
                ),
            )
        return TrainedModelArtifact(
            model=obj,
            metadata=ModelArtifactMetadata(
                feature_columns=tuple(DEFAULT_PREDICTION_FEATURES),
                target_column="SalePrice",
                model_name="house-price",
                model_version="legacy",
            ),
        )
    return obj
