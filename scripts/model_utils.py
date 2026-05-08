"""Shared utilities for training and inference scripts."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline


class LogTargetPipeline:
    """
    Wraps a sklearn Pipeline and applies log1p/expm1 to the target so
    the model learns on log-price.  Kept for backward-compat with older
    saved models; new models should use LocationAwarePipeline instead.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogTargetPipeline":
        self.pipeline.fit(X, np.log1p(y))
        return self

    def predict(self, X) -> np.ndarray:
        return np.expm1(self.pipeline.predict(X))


class LocationAwarePipeline:
    """
    Wraps a sklearn Pipeline and injects location and engineered features
    before the inner pipeline sees the data:

    Location features (fitted from training labels — no leakage):
      zip_median_price — median sale price per ZIP
      zip3             — 3-digit ZIP prefix (~250 regional market groups)

    Engineered features (computed from raw columns, no labels required):
      TotalBaths        = FullBath + 0.5 * HalfBath
      QualScore         = GrLivArea * OverallQual  (quality-weighted area)
      LogLotArea        = log1p(LotArea)           (de-skews right tail)
      PropertyAge       = CURRENT_YEAR - YearBuilt
      YearsSinceRemodel = CURRENT_YEAR - YearRemodAdd
      HasFireplace      = (Fireplaces > 0).astype(int)
      QualToZip         = QualScore / zip_median_price  (property quality
                          relative to ZIP — catches cheap properties in
                          expensive ZIPs and vice-versa)

    OverallCond is dropped: it is a constant (=7) in the training dataset
    and adds only noise.

    ZIP encoding uses Bayesian smoothing (shrink toward global median for
    small-sample ZIPs) and caps zip_median_price to prevent extreme ZIP
    medians from dominating predictions.

    Also optionally applies log1p/expm1 to the target (log_target=True).
    """

    CURRENT_YEAR: int = 2026

    # Bayesian smoothing strength: shrink ZIP median toward global median
    # for ZIPs with fewer than ~ZIP_SMOOTH_K training samples.
    ZIP_SMOOTH_K: int = 10

    # Maximum ratio of zip_median_price to global_median (prevents extreme
    # ZIP markets from completely overriding property-level features).
    ZIP_CAP_RATIO: float = 5.0

    def __init__(
        self,
        pipeline: Pipeline,
        log_target: bool = False,
        zip_col: str = "Neighborhood",
    ) -> None:
        self.pipeline = pipeline
        self.log_target = log_target
        self.zip_col = zip_col

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _engineer(X: pd.DataFrame) -> pd.DataFrame:
        """Compute engineered features; drop constant OverallCond."""
        cy = LocationAwarePipeline.CURRENT_YEAR
        X = X.copy()
        X["TotalBaths"] = X["FullBath"] + 0.5 * X.get("HalfBath", 0)
        X["QualScore"] = X["GrLivArea"] * X["OverallQual"]
        X["LogLotArea"] = np.log1p(X["LotArea"].clip(lower=0))
        X["PropertyAge"] = (cy - X["YearBuilt"].clip(1800, cy)).clip(lower=0)
        X["YearsSinceRemodel"] = (
            cy - X["YearRemodAdd"].clip(1800, cy)).clip(lower=0)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        # Average room size — larger rooms per sqft indicate higher quality homes
        X["SqftPerRoom"] = X["GrLivArea"] / X["TotRmsAbvGrd"].clip(lower=1)
        # Garage as a fraction of living area — attached garages in luxury homes
        X["GarageRatio"] = X["GarageArea"] / X["GrLivArea"].clip(lower=1)
        # OverallCond is constant (=7) — drop to avoid adding noise
        X = X.drop(columns=["OverallCond"], errors="ignore")
        # The Redfin "luxury" PropertyType label is a marketing tag that does
        # not reliably predict price — a "luxury" listing can sell at any tier.
        # Remapping it to "single_family" prevents the one-hot encoder from
        # learning a spurious 2× price multiplier for this label.
        if "PropertyType" in X.columns:
            X["PropertyType"] = X["PropertyType"].replace(
                "luxury", "single_family")
        return X

    def _enrich(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer features then inject ZIP location columns."""
        X = self._engineer(X)
        zips = X[self.zip_col].fillna("").astype(str).str[:5]
        zip_med = zips.map(self.zip_medians_).fillna(self.global_median_)
        # Cap extreme ZIP medians to prevent them from dominating
        cap = self.global_median_ * self.ZIP_CAP_RATIO
        floor = self.global_median_ * (1.0 / self.ZIP_CAP_RATIO)
        zip_med = zip_med.clip(lower=floor, upper=cap)
        X["zip_median_price"] = zip_med
        X["zip3"] = zips.str[:3]
        # Property quality relative to ZIP — helps model handle cheap
        # properties in expensive ZIPs and expensive in cheap ZIPs
        X["QualToZip"] = X["QualScore"] / zip_med.clip(lower=1.0)
        return X

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y) -> "LocationAwarePipeline":
        y_arr = np.asarray(y, dtype=float)
        self.global_median_: float = float(np.median(y_arr))
        # Compute ZIP-level median price from training labels only (no leakage)
        # Apply Bayesian smoothing: shrink toward global median for small ZIPs
        zips = X[self.zip_col].fillna("").astype(str).str[:5]
        ser = pd.Series(y_arr, index=zips.values)
        zip_raw = ser.groupby(level=0).median()
        zip_counts = ser.groupby(level=0).count()
        k = self.ZIP_SMOOTH_K
        # smoothed = (raw_median * n + global_median * k) / (n + k)
        self.zip_medians_: pd.Series = (
            zip_raw * zip_counts + self.global_median_ * k
        ) / (zip_counts + k)
        X_enriched = self._enrich(X)
        y_fit = np.log1p(y_arr) if self.log_target else y_arr
        self.pipeline.fit(X_enriched, y_fit)
        return self

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            preds = self.pipeline.predict(self._enrich(X))
        else:
            preds = self.pipeline.predict(X)
        return np.expm1(preds) if self.log_target else preds


class EnsembleModel:
    """
    Level-2 stacking ensemble using tier-stratified out-of-fold (OOF) predictions.

    Architecture mirrors the 4-tier champion system exactly:
      budget  <$200K        -- shallow LightGBM + log-target (champion params)
      low     $200K-500K    -- deeper LightGBM (champion params)
      mid     $500K-1.5M    -- deep LightGBM, moderate regularization
      luxury  >$1.5M        -- deep LightGBM + log-target

    Training (no data leakage):
      1. 5-fold CV: in each fold, fit 4 TIER-SPECIFIC sub-models.  Budget model
         trains only on budget homes, etc.  Get OOF predictions from ALL 4 models
         for every held-out record regardless of its actual tier.
      2. Build meta-features: 4 OOF predictions + statistical aggregates +
         key raw property attributes.
      3. Train a LightGBM meta-learner on log1p(SalePrice) using those features.
      4. Retrain 4 tier-specific sub-models on full data for inference.

    At inference the meta-learner implicitly learns optimal routing AND
    border-case blending without ever seeing the actual price.

    The 4 champion models under models/ are NOT modified.
    """

    _TIER_BOUNDS = [0, 200_000, 500_000, 1_500_000, float("inf")]
    _TIER_NAMES = ["budget", "low", "mid", "luxury"]

    # Champion hyperparameters replicated in each fold sub-model
    _TIER_CONFIGS = {
        "budget": dict(
            n_estimators=500, max_depth=6,  learning_rate=0.03,
            num_leaves=24,  min_child_samples=8,  subsample=0.8,
            colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5,
            log_target=True,
        ),
        "low": dict(
            n_estimators=400, max_depth=12, learning_rate=0.05,
            num_leaves=48,  min_child_samples=5,  reg_lambda=0.5,
            reg_alpha=0.2,  log_target=False,
        ),
        "mid": dict(
            n_estimators=400, max_depth=13, learning_rate=0.05,
            num_leaves=50,  min_child_samples=20, reg_lambda=1.0,
            reg_alpha=0.5,  log_target=False,
        ),
        "luxury": dict(
            n_estimators=400, max_depth=8,  learning_rate=0.04,
            num_leaves=32,  min_child_samples=5,  reg_lambda=2.0,
            log_target=True,
        ),
    }

    _NUMERIC_FEATURES = [
        "LotArea", "OverallQual", "YearBuilt", "YearRemodAdd",
        "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
        "Fireplaces", "GarageCars", "GarageArea", "NeighborhoodScore",
        "zip_median_price",
        "TotalBaths", "QualScore", "LogLotArea",
        "PropertyAge", "YearsSinceRemodel", "HasFireplace",
        "QualToZip", "SqftPerRoom", "GarageRatio",
    ]
    _CATEGORICAL_FEATURES = ["PropertyType", "HouseStyle", "zip3"]

    def _build_sub(self, log_target, **lgbm_kwargs):
        from sklearn.pipeline import Pipeline as SKPipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from lightgbm import LGBMRegressor

        pre = ColumnTransformer([
            ("num", "passthrough", self._NUMERIC_FEATURES),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
             self._CATEGORICAL_FEATURES),
        ])
        lgbm = LGBMRegressor(n_jobs=1, verbose=-1,
                             random_state=42, **lgbm_kwargs)
        return LocationAwarePipeline(
            SKPipeline([("pre", pre), ("lgbm", lgbm)]), log_target=log_target
        )

    def _predict_four(self, X):
        return np.column_stack([
            self.sub_models_[name].predict(X) for name in self._TIER_NAMES
        ])

    @staticmethod
    def _make_meta_X(X_raw, pm):
        p = np.clip(pm, 10_000, 100_000_000).astype(float)
        pb, pl, pm_, px = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        p_sorted = np.sort(p, axis=1)
        lux_blend = 0.6 * px + 0.4 * pm_

        meta = np.column_stack([
            pb, pl, pm_, px, lux_blend,
            np.log1p(pb), np.log1p(pl), np.log1p(pm_), np.log1p(px),
            p.mean(axis=1), p.std(axis=1), p.max(axis=1), p.min(axis=1),
            p_sorted[:, 1], p_sorted[:, 2],
            px / np.maximum(pb, 1.0),
            pm_ / np.maximum(pl, 1.0),
            (px - pb) / np.maximum(pb, 1.0),
            X_raw["GrLivArea"].values.astype(float),
            X_raw["OverallQual"].values.astype(float),
            X_raw["LotArea"].values.astype(float),
            X_raw["GarageCars"].values.astype(float),
            X_raw["BedroomAbvGr"].values.astype(float),
            X_raw["TotRmsAbvGrd"].values.astype(float),
            X_raw["NeighborhoodScore"].values.astype(float),
            (2026 - X_raw["YearBuilt"].values.astype(float)).clip(0),
            X_raw["GarageArea"].values.astype(float),
        ])
        cols = [
            "p_budget", "p_low", "p_mid", "p_luxury", "p_lux_blend",
            "log_p_budget", "log_p_low", "log_p_mid", "log_p_luxury",
            "pred_mean", "pred_std", "pred_max", "pred_min",
            "pred_q1", "pred_q3",
            "ratio_lux_budget", "ratio_mid_low", "range_ratio",
            "GrLivArea", "OverallQual", "LotArea", "GarageCars",
            "BedroomAbvGr", "TotRmsAbvGrd", "NeighborhoodScore",
            "PropertyAge", "GarageArea",
        ]
        return pd.DataFrame(meta, columns=cols)

    def fit(self, X, y, n_folds=5):
        from sklearn.model_selection import KFold
        from lightgbm import LGBMRegressor

        y_arr = np.asarray(y, dtype=float)
        valid = np.isfinite(y_arr) & (y_arr > 0)
        if not valid.all():
            print(
                f"  Dropping {(~valid).sum()} records with invalid SalePrice")
            X = X[valid].reset_index(drop=True)
            y_arr = y_arr[valid]
        n = len(X)
        X = X.reset_index(drop=True)

        # ── Step 1: Tier-stratified OOF predictions ──────────────────────
        print(f"  Tier-stratified OOF ({n_folds}-fold CV)...")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof = np.zeros((n, 4))

        for fi, (tr_idx, va_idx) in enumerate(kf.split(X)):
            print(
                f"    Fold {fi + 1}/{n_folds} — fitting tier sub-models...", flush=True)
            X_tr = X.iloc[tr_idx].reset_index(drop=True)
            y_tr = y_arr[tr_idx]
            X_va = X.iloc[va_idx].reset_index(drop=True)

            for j, (name, cfg_orig) in enumerate(self._TIER_CONFIGS.items()):
                cfg = dict(cfg_orig)
                lo = self._TIER_BOUNDS[j]
                hi = self._TIER_BOUNDS[j + 1]
                tier_mask = (y_tr >= lo) & (y_tr < hi)
                if tier_mask.sum() < 50:
                    tier_mask = np.ones(len(y_tr), dtype=bool)

                log = cfg.pop("log_target")
                m = self._build_sub(log_target=log, **cfg)
                m.fit(X_tr[tier_mask].reset_index(drop=True), y_tr[tier_mask])
                oof[np.array(va_idx), j] = m.predict(X_va)

        # OOF tier-routed error (upper bound for meta-learner)
        routed = np.zeros(n)
        for j in range(4):
            lo, hi = self._TIER_BOUNDS[j], self._TIER_BOUNDS[j + 1]
            mask = (y_arr >= lo) & (y_arr < hi)
            if j == 3:
                routed[mask] = 0.6 * oof[mask, 3] + 0.4 * oof[mask, 2]
            else:
                routed[mask] = oof[mask, j]
        vm = np.isfinite(routed) & (y_arr > 0)
        oof_err = np.abs(routed[vm] - y_arr[vm]) / y_arr[vm] * 100
        print(
            f"    OOF tier-routed error (meta ceiling): {oof_err.mean():.1f}%")

        # ── Step 2: Meta-learner on OOF predictions ───────────────────────
        print("  Training LightGBM meta-learner...")
        meta_X = self._make_meta_X(X, oof)
        self.meta_learner_ = LGBMRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.04,
            num_leaves=32, reg_lambda=2.0, n_jobs=1, verbose=-1, random_state=42,
        )
        self.meta_learner_.fit(meta_X, np.log1p(y_arr))

        # ── Step 3: Final tier-stratified sub-models on full data ─────────
        print("  Training final sub-models (tier-stratified, full data)...")
        self.sub_models_ = {}
        for j, (name, cfg_orig) in enumerate(self._TIER_CONFIGS.items()):
            lo, hi = self._TIER_BOUNDS[j], self._TIER_BOUNDS[j + 1]
            tier_mask = (y_arr >= lo) & (y_arr < hi)
            print(f"    [{name}]  {tier_mask.sum():,} records...", flush=True)
            cfg = dict(cfg_orig)
            log = cfg.pop("log_target")
            m = self._build_sub(log_target=log, **cfg)
            m.fit(X[tier_mask].reset_index(drop=True), y_arr[tier_mask])
            self.sub_models_[name] = m

        return self

    def predict(self, X):
        X = X.reset_index(drop=True)
        pm = self._predict_four(X)
        meta_X = self._make_meta_X(X, pm)
        return np.expm1(self.meta_learner_.predict(meta_X))


class SmartRouter:
    """
    Combines all 4 champion models using a trained tier classifier.

    Instead of retraining sub-models (like EnsembleModel), SmartRouter:
      1. Loads the 4 existing champion models from disk — they are NEVER
         retrained or modified.
      2. Trains a lightweight LightGBM multiclass classifier to predict
         which tier (budget / low / mid / luxury) a property belongs to
         based purely on its features (no oracle price needed).
      3. At inference, gets predictions from ALL 4 champions and blends
         them using the classifier's soft tier probabilities.

    Benefits over stacking:
      - Uses full champion models (not weaker CV fold sub-models).
      - Handles tier boundaries smoothly via soft probability weighting.
      - No meta-learner overfitting — the blending is probabilistic.
      - Realistic for production: no known price required for routing.

    Expected performance: ~9–10% mean error (vs. oracle routing ~8.8%).
    """

    CHAMPION_PATHS = {
        "budget":  "nationwide_budget_model.joblib",
        "low":     "nationwide_low_price_model.joblib",
        "mid":     "nationwide_mid_price_model_v2.joblib",
        "luxury":  "nationwide_luxury_model.joblib",
    }

    _TIER_BOUNDS = [0, 200_000, 500_000, 1_500_000, float("inf")]
    ZIP_SMOOTH_K: int = 10
    ZIP_CAP_RATIO: float = 5.0
    CURRENT_YEAR: int = 2026

    def __init__(self):
        self.tier_classifier_ = None
        self.zip_medians_: pd.Series = pd.Series(dtype=float)
        self.global_median_: float = 0.0
        self.champions_: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label_tiers(self, y: np.ndarray) -> np.ndarray:
        labels = np.full(len(y), 3, dtype=int)   # luxury default
        labels[y < 1_500_000] = 2                 # mid
        labels[y < 500_000] = 1                 # low
        labels[y < 200_000] = 0                 # budget
        return labels

    def _compute_zip_medians(self, X: pd.DataFrame, y: np.ndarray) -> pd.Series:
        zips = X["Neighborhood"].fillna("").astype(str).str[:5]
        ser = pd.Series(y, index=zips.values)
        zip_raw = ser.groupby(level=0).median()
        zip_counts = ser.groupby(level=0).count()
        k = self.ZIP_SMOOTH_K
        return (zip_raw * zip_counts + self.global_median_ * k) / (zip_counts + k)

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        cy = self.CURRENT_YEAR
        X = X.copy()
        X["TotalBaths"] = X["FullBath"] + 0.5 * X.get("HalfBath", 0)
        X["QualScore"] = X["GrLivArea"] * X["OverallQual"]
        X["LogLotArea"] = np.log1p(X["LotArea"].clip(lower=0))
        X["PropertyAge"] = (cy - X["YearBuilt"].clip(1800, cy)).clip(lower=0)
        X["YearsSinceRemodel"] = (
            cy - X["YearRemodAdd"].clip(1800, cy)).clip(lower=0)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        X["SqftPerRoom"] = X["GrLivArea"] / X["TotRmsAbvGrd"].clip(lower=1)
        X["GarageRatio"] = X["GarageArea"] / X["GrLivArea"].clip(lower=1)
        X = X.drop(columns=["OverallCond"], errors="ignore")
        if "PropertyType" in X.columns:
            X["PropertyType"] = X["PropertyType"].replace(
                "luxury", "single_family")
        return X

    def _enrich(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._engineer(X)
        zips = X["Neighborhood"].fillna("").astype(str).str[:5]
        zip_med = zips.map(self.zip_medians_).fillna(self.global_median_)
        cap = self.global_median_ * self.ZIP_CAP_RATIO
        floor = self.global_median_ * (1.0 / self.ZIP_CAP_RATIO)
        zip_med = zip_med.clip(lower=floor, upper=cap)
        X["zip_median_price"] = zip_med
        X["zip3"] = zips.str[:3]
        X["QualToZip"] = X["QualScore"] / zip_med.clip(lower=1.0)
        return X

    @staticmethod
    def _set_cat_dtypes(X: pd.DataFrame) -> pd.DataFrame:
        for col in ["PropertyType", "HouseStyle", "Neighborhood", "zip3"]:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y, models_dir) -> "SmartRouter":
        """
        Train the tier classifier on features; load champions from models_dir.

        Parameters
        ----------
        X          : raw feature DataFrame (same columns as training pipeline)
        y          : sale prices (1-D array-like)
        models_dir : directory containing the 4 champion .joblib files
        """
        import joblib as jl
        from lightgbm import LGBMClassifier

        models_dir = Path(models_dir)
        y_arr = np.asarray(y, dtype=float)
        self.global_median_ = float(np.median(y_arr))
        self.zip_medians_ = self._compute_zip_medians(X, y_arr)

        # ── Load champion models ──────────────────────────────────────
        print("  Loading champion models...")
        for name, fname in self.CHAMPION_PATHS.items():
            path = models_dir / fname
            self.champions_[name] = jl.load(path)
            print(f"    [{name}]  {fname}")

        # ── Engineer + enrich features ────────────────────────────────
        X_enc = self._set_cat_dtypes(self._enrich(X))
        cat_cols = [c for c in ["PropertyType", "HouseStyle", "Neighborhood", "zip3"]
                    if c in X_enc.columns]

        # ── Tier labels from known prices ─────────────────────────────
        tier_labels = self._label_tiers(y_arr)
        counts = {n: int((tier_labels == i).sum())
                  for i, n in enumerate(["budget", "low", "mid", "luxury"])}
        print(f"  Tier distribution: {counts}")

        # ── Train classifier ─────────────────────────────────────────
        print("  Training tier classifier...")
        self.tier_classifier_ = LGBMClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=48,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight="balanced",
            verbose=-1,
            n_jobs=1,
            random_state=42,
        )
        self.tier_classifier_.fit(
            X_enc, tier_labels, categorical_feature=cat_cols)

        acc = (self.tier_classifier_.predict(X_enc) == tier_labels).mean()
        print(f"  Tier classifier train accuracy: {acc:.1%}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return blended champion predictions weighted by soft tier probabilities.

        For a clearly budget property the budget champion carries ~100% weight;
        for a near-boundary property adjacent models blend smoothly.
        """
        # Champion predictions (each handles its own feature engineering)
        p_budget = self.champions_["budget"].predict(X)
        p_low = self.champions_["low"].predict(X)
        p_mid = self.champions_["mid"].predict(X)
        p_lux_raw = self.champions_["luxury"].predict(X)
        p_lux = 0.6 * p_lux_raw + 0.4 * p_mid   # same blend as oracle routing

        # Tier probabilities from classifier (no oracle price needed)
        X_enc = self._set_cat_dtypes(self._enrich(X.reset_index(drop=True)))
        cat_cols = [c for c in ["PropertyType", "HouseStyle", "Neighborhood", "zip3"]
                    if c in X_enc.columns]
        for col in cat_cols:
            X_enc[col] = X_enc[col].astype("category")
        tier_probs = self.tier_classifier_.predict_proba(X_enc)   # (n, 4)

        # Weighted blend
        return (
            tier_probs[:, 0] * p_budget +
            tier_probs[:, 1] * p_low +
            tier_probs[:, 2] * p_mid +
            tier_probs[:, 3] * p_lux
        )
