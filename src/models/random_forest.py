"""Random Forest demand model."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class RandomForestDemandModel:
    """Per-product Random Forest demand model.

    Uses price, discount depth, and engineered features to predict
    log-demand. Captures nonlinear price-demand relationships that
    the log-log OLS model cannot.

    References:
        - Breiman (2001), "Random Forests"
        - Ferreira et al. (2016), "Analytics for an Online Retailer:
          Demand Forecasting and Price Optimization"
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 8, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = {}
        self.feature_dims = {}  # track per-product feature dimension

    def _build_features(self, grp: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame({
            "log_price": np.log(grp["avg_price"].clip(lower=0.01)),
            "discount_depth": grp["discount_depth"],
        })
        # Prefer continuous exposure fractions over binary indicators
        if "display_pct" in grp.columns:
            X["display"] = grp["display_pct"].values
        elif "has_display" in grp.columns:
            X["display"] = grp["has_display"].values
        if "mailer_pct" in grp.columns:
            X["mailer"] = grp["mailer_pct"].values
        elif "has_mailer" in grp.columns:
            X["mailer"] = grp["has_mailer"].values
        if "log_store_coverage" in grp.columns:
            X["log_store_coverage"] = grp["log_store_coverage"].values
        if "campaign_active" in grp.columns:
            X["campaign_active"] = grp["campaign_active"].values
        return X

    def fit(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Fit per-product RF models."""
        results = []

        for pid, grp in panel.groupby("PRODUCT_ID"):
            if len(grp) < 15:
                continue

            y = np.log(grp["quantity"].clip(lower=0.5)).values
            X_df = self._build_features(grp)
            X = X_df.values
            self.feature_dims[pid] = X.shape[1]

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X, y)
            y_pred = rf.predict(X)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Estimate elasticity via finite difference
            X_base = X.copy()
            X_up = X.copy()
            X_up[:, 0] += 0.01  # +1% log price
            elasticity = (rf.predict(X_up) - rf.predict(X_base)).mean() / 0.01

            # Estimate discount effect: marginal effect of 1pp discount depth
            X_no_disc = X.copy()
            X_no_disc[:, 1] = 0  # discount_depth = 0
            X_with_disc = X.copy()
            X_with_disc[:, 1] = 0.15  # discount_depth = 15%
            disc_effect = (rf.predict(X_with_disc) - rf.predict(X_no_disc)).mean() / 0.15

            residuals = y - y_pred
            self.models[pid] = rf

            results.append({
                "PRODUCT_ID": pid,
                "intercept": y.mean(),
                "elasticity": elasticity,
                "disc_effect": disc_effect,
                "r2": r2,
                "residual_std": residuals.std(),
                "n_obs": len(y),
                "feature_importance_price": rf.feature_importances_[0],
                "feature_importance_disc": rf.feature_importances_[1],
            })

        return pd.DataFrame(results)

    def predict(self, product_id, log_price: float, discount_depth: float = 0.0,
                has_display: int = 0, has_mailer: int = 0,
                log_store_coverage: float = 0.0, campaign_active: int = 0) -> float:
        """Predict log-demand, matching the feature dimension used during fit."""
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
