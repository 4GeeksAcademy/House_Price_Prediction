from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = x.select_dtypes(exclude=["number"]).columns.tolist()

    # keep_empty_features=True preserves columns that are entirely null in
    # training data (e.g. CensusMedianValue / MedianIncomeK / OwnerOccupiedRate
    # which come from the live Census API and are absent from CSV sources).
    # Those slots stay in the pipeline so live values are not silently dropped
    # at inference time.  RandomForestRegressor is scale-invariant, so no
    # StandardScaler is needed here; adding one would cause zero-variance
    # warnings for the all-null census columns and incorrect scaling at
    # inference where those columns have real values.
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
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )
