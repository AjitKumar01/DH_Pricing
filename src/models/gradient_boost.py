"""Gradient Boosting demand models (XGBoost and LightGBM)."""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb


class GradientBoostDemandModel:
    """Per-product gradient boosting demand model.

    Supports both XGBoost and LightGBM backends. Gradient-boosted
    trees often outperform both linear models and random forests
    on tabular data with moderate sample sizes.

    References:
        - Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"
        - Ke et al. (2017), "LightGBM: A Highly Efficient Gradient Boosting
          Decision Tree"
        - Ban & Rudin (2019), "The Big Data Newsvendor"
    """

    def __init__(self, backend: str = "xgboost", n_estimators: int = 100,
                 max_depth: int = 5, learning_rate: float = 0.1,
                 random_state: int = 42):
        self.backend = backend
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.models = {}
        self.feature_dims = {}  # track per-product feature dimension

    def _build_features(self, grp: pd.DataFrame) -> np.ndarray:
        cols = [
            np.log(grp["avg_price"].clip(lower=0.01).values),
            grp["discount_depth"].values,
        ]
        # Prefer continuous exposure fractions over binary indicators
        if "display_pct" in grp.columns:
            cols.append(grp["display_pct"].values)
        elif "has_display" in grp.columns:
            cols.append(grp["has_display"].values)
        if "mailer_pct" in grp.columns:
            cols.append(grp["mailer_pct"].values)
        elif "has_mailer" in grp.columns:
            cols.append(grp["has_mailer"].values)
        if "log_store_coverage" in grp.columns:
            cols.append(grp["log_store_coverage"].values)
        if "campaign_active" in grp.columns:
            cols.append(grp["campaign_active"].values)
        return np.column_stack(cols)

    def _fit_one(self, X: np.ndarray, y: np.ndarray):
        if self.backend == "xgboost":
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbosity=0,
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbose=-1,
            )
        model.fit(X, y)
        return model

    def fit(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Fit per-product gradient boosting models."""
        results = []

        for pid, grp in panel.groupby("PRODUCT_ID"):
            if len(grp) < 15:
                continue

            y = np.log(grp["quantity"].clip(lower=0.5).values)
            X = self._build_features(grp)
            self.feature_dims[pid] = X.shape[1]

            model = self._fit_one(X, y)
            y_pred = model.predict(X)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Finite-difference elasticity
            X_base = X.copy()
            X_up = X.copy()
            X_up[:, 0] += 0.01
            elasticity = (model.predict(X_up) - model.predict(X_base)).mean() / 0.01

            # Discount effect: marginal effect of 1pp discount depth
            X_no = X.copy(); X_no[:, 1] = 0
            X_yes = X.copy(); X_yes[:, 1] = 0.15
            disc_effect = (model.predict(X_yes) - model.predict(X_no)).mean() / 0.15

            residuals = y - y_pred
            self.models[pid] = model

            results.append({
                "PRODUCT_ID": pid,
                "intercept": y.mean(),
                "elasticity": elasticity,
                "disc_effect": disc_effect,
                "r2": r2,
                "residual_std": residuals.std(),
                "n_obs": len(y),
            })

        return pd.DataFrame(results)

    def predict(self, product_id, log_price: float, discount_depth: float = 0.0,
                has_display: int = 0, has_mailer: int = 0,
                log_store_coverage: float = 0.0, campaign_active: int = 0) -> float:
        features = [log_price, discount_depth, has_display, has_mailer,
                     log_store_coverage, campaign_active]
        n_feats = self.feature_dims.get(product_id, 4)
        X = np.array([features[:n_feats]])
        return self.models[product_id].predict(X)[0]

    def predict_level(self, product_id, price: float, discount_depth: float = 0.0,
                      has_display: int = 0, has_mailer: int = 0,
                      log_store_coverage: float = 0.0, campaign_active: int = 0) -> float:
        log_pred = self.predict(product_id, np.log(max(price, 0.01)),
                                discount_depth, has_display, has_mailer,
                                log_store_coverage, campaign_active)
        return np.exp(log_pred)
