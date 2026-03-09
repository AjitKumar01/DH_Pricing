"""Log-log OLS demand model (baseline).

Enhanced with continuous discount depth, promotional covariates
(display/mailer exposure from causal data), lagged demand (AR(1)
persistence), and within-segment cross-product substitution.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


class LogLogDemandModel:
    """Per-product log-log OLS demand model with promotional controls.

    Model: ln(Q_it) = alpha_i + epsilon_i * ln(P_it) + gamma_i * d_it
           + delta1_i * display_it + delta2_i * mailer_it
           + rho_i * ln(Q_{i,t-1})
           + lambda_i * seg_disc_other_{it}
           + eta_i * ln(n_stores_{it})
           + kappa_i * campaign_{it}
           + u_it

    where d_it is the continuous discount depth (not a binary flag),
    display_it, mailer_it are promotional exposure fractions,
    ln(Q_{i,t-1}) is the lagged log-demand capturing demand persistence,
    seg_disc_other_{it} is the mean discount depth of other products
    in the same segment (cross-product substitution),
    ln(n_stores_{it}) is log store coverage (distribution/availability),
    and campaign_{it} is a binary indicator for active campaign targeting.

    References:
        - Tellis (1988), "The Price Elasticity of Selective Demand"
        - Bijmolt et al. (2005), "New Empirical Generalizations on the
          Determinants of Price Elasticity"
        - Bronnenberg et al. (2009), "Do Pharmacists Buy Bayer?"
        - Wan et al. (2017), "Modeling consumer preferences and price
          sensitivities from large-scale grocery shopping transaction logs"
    """

    def __init__(self):
        self.params = {}

    @staticmethod
    def _build_segment_discount(panel: pd.DataFrame, seg_mapping: pd.DataFrame) -> pd.DataFrame:
        """Compute mean discount depth of other products in the same segment.

        For substitution estimation: for each product-week, computes the
        average discount depth among other products in the same segment.
        """
        if seg_mapping is None:
            return panel
        merged = panel.merge(
            seg_mapping[["PRODUCT_ID", "segment"]], on="PRODUCT_ID", how="left"
        )
        # For each segment-week, compute the total discount mass and count
        seg_week = (
            merged.groupby(["segment", "WEEK_NO"])
            .agg(
                seg_total_disc=("discount_depth", "sum"),
                seg_count=("discount_depth", "count"),
            )
            .reset_index()
        )
        merged = merged.merge(seg_week, on=["segment", "WEEK_NO"], how="left")
        # Mean discount of OTHER products = (total - own) / (count - 1)
        merged["seg_disc_other"] = np.where(
            merged["seg_count"] > 1,
            (merged["seg_total_disc"] - merged["discount_depth"]) / (merged["seg_count"] - 1),
            0.0,
        )
        panel = panel.copy()
        panel["seg_disc_other"] = merged["seg_disc_other"].values
        return panel

    def fit(self, panel: pd.DataFrame, seg_mapping: pd.DataFrame = None) -> pd.DataFrame:
        """Fit individual OLS models for each product.

        Uses continuous discount_depth instead of binary flag, and
        includes display/mailer promotional controls when available.
        Adds cosine seasonality controls (annual cycle), lagged demand
        (AR(1) persistence), cross-product substitution, store coverage,
        and campaign activity.
        """
        has_display = "display_pct" in panel.columns or "has_display" in panel.columns
        has_mailer = "mailer_pct" in panel.columns or "has_mailer" in panel.columns
        has_week = "WEEK_NO" in panel.columns
        has_store_cov = "log_store_coverage" in panel.columns
        has_campaign = "campaign_active" in panel.columns

        # Prefer continuous exposure fractions over binary indicators
        display_col = "display_pct" if "display_pct" in panel.columns else "has_display"
        mailer_col = "mailer_pct" if "mailer_pct" in panel.columns else "has_mailer"

        # Build cross-product substitution feature
        panel = self._build_segment_discount(panel, seg_mapping)

        results = []

        for pid, grp in panel.groupby("PRODUCT_ID"):
            if len(grp) < 10:
                continue

            # Sort by week for proper lagging
            grp = grp.sort_values("WEEK_NO").copy()

            log_qty = np.log(grp["quantity"].clip(lower=0.5))

            # Create lagged log-quantity (AR(1) term)
            lag_log_qty = log_qty.shift(1)

            # Drop first observation (no lag available)
            valid = lag_log_qty.notna()
            if valid.sum() < 10:
                continue

            y = log_qty[valid]
            x_dict = {
                "log_price": np.log(grp.loc[valid.index[valid], "avg_price"].clip(lower=0.01)),
                "discount_depth": grp.loc[valid.index[valid], "discount_depth"],
                "lag_log_qty": lag_log_qty[valid],
            }
            # Cross-product substitution
            if "seg_disc_other" in grp.columns:
                x_dict["seg_disc_other"] = grp.loc[valid.index[valid], "seg_disc_other"]
            if has_display:
                x_dict["display"] = grp.loc[valid.index[valid], display_col]
            if has_mailer:
                x_dict["mailer"] = grp.loc[valid.index[valid], mailer_col]
            if has_week:
                week_vals = grp.loc[valid.index[valid], "WEEK_NO"]
                x_dict["cos_season"] = np.cos(2 * np.pi * week_vals / 52)
                x_dict["sin_season"] = np.sin(2 * np.pi * week_vals / 52)
            # Store coverage (log-transformed)
            if has_store_cov:
                x_dict["log_store_coverage"] = grp.loc[valid.index[valid], "log_store_coverage"]
            # Campaign activity indicator
            if has_campaign:
                camp_vals = grp.loc[valid.index[valid], "campaign_active"]
                # Only include if there's variation (some products never campaign-targeted)
                if camp_vals.std() > 0:
                    x_dict["campaign_active"] = camp_vals

            X = pd.DataFrame(x_dict)
            X = sm.add_constant(X)

            try:
                model = sm.OLS(y, X).fit(cov_type="HC1")
                row = {
                    "PRODUCT_ID": pid,
                    "intercept": model.params.get("const", 0),
                    "elasticity": model.params.get("log_price", 0),
                    "disc_effect": model.params.get("discount_depth", 0),
                    "demand_persistence": model.params.get("lag_log_qty", 0),
                    "substitution_effect": model.params.get("seg_disc_other", 0),
                    "elasticity_pval": model.pvalues.get("log_price", 1),
                    "disc_effect_pval": model.pvalues.get("discount_depth", 1),
                    "demand_persistence_pval": model.pvalues.get("lag_log_qty", 1),
                    "substitution_effect_pval": model.pvalues.get("seg_disc_other", 1),
                    "elasticity_se": model.bse.get("log_price", np.nan),
                    "disc_effect_se": model.bse.get("discount_depth", np.nan),
                    "demand_persistence_se": model.bse.get("lag_log_qty", np.nan),
                    "substitution_effect_se": model.bse.get("seg_disc_other", np.nan),
                    "r2": model.rsquared,
                    "r2_adj": model.rsquared_adj,
                    "residual_std": np.sqrt(model.mse_resid),
                    "n_obs": int(model.nobs),
                    "aic": model.aic,
                    "bic": model.bic,
                }
                if has_display:
                    row["display_effect"] = model.params.get("display", 0)
                    row["display_pval"] = model.pvalues.get("display", 1)
                if has_mailer:
                    row["mailer_effect"] = model.params.get("mailer", 0)
                    row["mailer_pval"] = model.pvalues.get("mailer", 1)
                if has_week:
                    row["cos_season_coef"] = model.params.get("cos_season", 0)
                    row["sin_season_coef"] = model.params.get("sin_season", 0)
                if has_store_cov:
                    row["store_coverage_effect"] = model.params.get("log_store_coverage", 0)
                    row["store_coverage_pval"] = model.pvalues.get("log_store_coverage", 1)
                    row["store_coverage_se"] = model.bse.get("log_store_coverage", np.nan)
                if "campaign_active" in model.params.index:
                    row["campaign_effect"] = model.params.get("campaign_active", 0)
                    row["campaign_pval"] = model.pvalues.get("campaign_active", 1)
                    row["campaign_se"] = model.bse.get("campaign_active", np.nan)
                else:
                    row["campaign_effect"] = 0
                    row["campaign_pval"] = 1.0
                    row["campaign_se"] = np.nan

                # Store mean log-qty for AR(1) deviation computation in validation
                row["mean_log_qty"] = log_qty.mean()

                results.append(row)
                self.params[pid] = row
            except Exception:
                continue

        return pd.DataFrame(results)

    def predict(self, product_id, log_price: float, discount_flag: float) -> float:
        """Predict log-demand for a single product."""
        p = self.params[product_id]
        return p["intercept"] + p["elasticity"] * log_price + p["disc_effect"] * discount_flag

    def predict_level(self, product_id, price: float, discount_flag: float) -> float:
        """Predict demand in levels (exp of log prediction)."""
        log_pred = self.predict(product_id, np.log(max(price, 0.01)), discount_flag)
        return np.exp(log_pred)
