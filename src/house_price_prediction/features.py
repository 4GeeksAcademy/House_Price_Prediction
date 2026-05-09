from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# Year used as the reference for age-based derived features.
# Using a fixed year keeps training and inference consistent.
_REFERENCE_YEAR = 2026


def _col(df: pd.DataFrame, name: str, default: float) -> pd.Series:
    """Return a numeric column from df, or a constant Series if the column is absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _engineer_features(x: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features that improve model generalization.

    These features are universally meaningful for any US single-family home
    and help the gradient-boosted trees find price relationships more
    efficiently than raw columns alone.

    Derived columns added:
      Age                 – years since construction (universal depreciation signal)
      RenovationAge       – years since last renovation
      TotalBath           – full baths + 0.5 × half baths
      TotalSF             – GrLivArea + 0.5 × BasementSF (weighted total living SF)
      QualArea            – OverallQual × GrLivArea (quality-weighted size; highest
                            correlation with SalePrice at 0.755 on KC data)
      QualCond            – OverallQual × OverallCond (combined condition signal)
      log_LotArea         – log1p of LotArea (skew=13 → removes extreme distortion;
                            doubles raw LotArea correlation from 0.09 → 0.16)
      HasBasement         – 1 if any finished basement area exists, else 0
      IsNewHome           – 1 if built after 2000 (new-construction premium)
    """
    x = x.copy()

    year_built  = _col(x, "YearBuilt", _REFERENCE_YEAR - 30)
    year_remod  = _col(x, "YearRemodAdd", _REFERENCE_YEAR - 20)
    full_bath   = _col(x, "FullBath", 1)
    half_bath   = _col(x, "HalfBath", 0)
    gr_liv_area = _col(x, "GrLivArea", 1500)
    basement_sf = _col(x, "BasementSF", 0)
    lot_area    = _col(x, "LotArea", 7500)
    overall_qual = _col(x, "OverallQual", 5)
    overall_cond = _col(x, "OverallCond", 5)

    x["Age"]          = (_REFERENCE_YEAR - year_built).clip(lower=0)
    x["RenovationAge"]= (_REFERENCE_YEAR - year_remod).clip(lower=0)
    x["TotalBath"]    = full_bath + 0.5 * half_bath
    x["TotalSF"]      = gr_liv_area + 0.5 * basement_sf
    x["QualArea"]     = overall_qual * gr_liv_area
    x["QualCond"]     = overall_qual * overall_cond
    x["log_LotArea"]  = np.log1p(lot_area)
    x["HasBasement"]  = (basement_sf > 0).astype(float)
    x["IsNewHome"]    = (year_built >= 2000).astype(float)

    return x


def build_preprocessor(x: pd.DataFrame) -> Pipeline:
    """Build a full preprocessing pipeline including feature engineering.

    Returns a sklearn Pipeline (not a bare ColumnTransformer) so that the
    feature engineering step runs consistently at both training and inference.

    Pipeline steps:
      1. engineer  – FunctionTransformer that adds Age, RenovationAge,
                     TotalBath, TotalSF derived columns.
      2. transform – ColumnTransformer with median-imputed numeric columns
                     and OHE categorical columns.

    keep_empty_features=True preserves columns that are entirely null in
    training data (e.g. CensusMedianValue / MedianIncomeK / OwnerOccupiedRate
    which come from the live Census API and are absent from CSV sources).
    LightGBM is scale-invariant so no StandardScaler is applied.
    """
    # Apply engineering to discover the full column set (including derived cols)
    x_eng = _engineer_features(x)
    numeric_columns = x_eng.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = x_eng.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("engineer", FunctionTransformer(_engineer_features, validate=False)),
            ("transform", column_transformer),
        ]
    )
