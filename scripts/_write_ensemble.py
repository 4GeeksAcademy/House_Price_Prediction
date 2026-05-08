"""
Helper script that rewrites EnsembleModel in model_utils.py.
Run once: python scripts/_write_ensemble.py
"""
import pathlib

ROOT = pathlib.Path(__file__).parent
src = (ROOT / "model_utils.py").read_text(encoding="utf-8")

MARKER = "\nclass EnsembleModel:"
cut = src.index(MARKER)
before = src[:cut]

NEW_CLASS = r'''

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
    _TIER_NAMES  = ["budget", "low", "mid", "luxury"]

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
        lgbm = LGBMRegressor(n_jobs=1, verbose=-1, random_state=42, **lgbm_kwargs)
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
            print(f"  Dropping {(~valid).sum()} records with invalid SalePrice")
            X = X[valid].reset_index(drop=True)
            y_arr = y_arr[valid]
        n = len(X)
        X = X.reset_index(drop=True)

        # ── Step 1: Tier-stratified OOF predictions ──────────────────────
        print(f"  Tier-stratified OOF ({n_folds}-fold CV)...")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof = np.zeros((n, 4))

        for fi, (tr_idx, va_idx) in enumerate(kf.split(X)):
            print(f"    Fold {fi + 1}/{n_folds} — fitting tier sub-models...", flush=True)
            X_tr = X.iloc[tr_idx].reset_index(drop=True)
            y_tr = y_arr[tr_idx]
            X_va = X.iloc[va_idx].reset_index(drop=True)

            for j, (name, cfg_orig) in enumerate(self._TIER_CONFIGS.items()):
                cfg = dict(cfg_orig)
                lo  = self._TIER_BOUNDS[j]
                hi  = self._TIER_BOUNDS[j + 1]
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
        print(f"    OOF tier-routed error (meta ceiling): {oof_err.mean():.1f}%")

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
'''

(ROOT / "model_utils.py").write_text(before + NEW_CLASS, encoding="utf-8")
print(
    f"model_utils.py updated. New length: {len((ROOT / 'model_utils.py').read_text())}")
