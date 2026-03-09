"""Empirical Bayes shrinkage for product-level demand parameters."""

import numpy as np
import pandas as pd


class EmpiricalBayesShrinkage:
    """Shrink product-level estimates toward segment medians.

    The shrinkage weight is determined by each product's R-squared:
    products with well-fitting models retain their individual estimates,
    while noisy models are pulled toward the segment consensus.

    References:
        - Morris (1983), "Parametric Empirical Bayes Inference"
        - Efron & Morris (1975), "Data Analysis Using Stein's Estimator"
        - James & Stein (1961), "Estimation with Quadratic Loss"
    """

    def __init__(self, max_weight: float = 0.9):
        self.max_weight = max_weight

    def shrink(
        self,
        product_params: pd.DataFrame,
        segment_mapping: pd.DataFrame,
        params_to_shrink: list = None,
    ) -> pd.DataFrame:
        """Apply empirical Bayes shrinkage.

        Uses James-Stein shrinkage weight:
            w_i = sigma^2_seg / (sigma^2_seg + sigma^2_i / n_i)

        Products with large estimation variance (high residual_std,
        low n_obs) get pulled more toward the segment median.

        Args:
            product_params: DataFrame with PRODUCT_ID, residual_std, n_obs,
                and parameter columns.
            segment_mapping: DataFrame with PRODUCT_ID, segment.
            params_to_shrink: List of column names to shrink
                (default: elasticity, disc_effect).

        Returns:
            DataFrame with shrunk parameters and calibrated intercepts.
        """
        if params_to_shrink is None:
            params_to_shrink = ["elasticity", "disc_effect"]

        # Also shrink display, mailer, persistence, substitution,
        # store_coverage, campaign if present
        optional_shrink = ["display_effect", "mailer_effect",
                           "demand_persistence", "substitution_effect",
                           "store_coverage_effect", "campaign_effect"]
        for col in optional_shrink:
            if col in product_params.columns and col not in params_to_shrink:
                params_to_shrink.append(col)

        df = product_params.merge(segment_mapping, on="PRODUCT_ID", how="left")

        # Compute segment medians and segment-level variance
        seg_medians = {}
        seg_vars = {}
        for col in params_to_shrink:
            seg_medians[col] = df.groupby("segment")[col].median()
            seg_vars[col] = df.groupby("segment")[col].var()

        # James-Stein shrinkage weight per product
        # w_i = sigma^2_seg / (sigma^2_seg + se_i^2)
        # Higher w_i → keep more of the product-specific estimate

        # Global median as fallback for single-product segments
        global_medians = {col: df[col].median() for col in params_to_shrink}
        # Global variance as fallback
        global_vars = {col: df[col].var() for col in params_to_shrink}

        # Fallback estimation variance from residual_std^2 / n_obs
        n_obs = df["n_obs"].clip(lower=5)
        residual_var = df["residual_std"].clip(lower=0.01) ** 2
        fallback_var = residual_var / n_obs

        for col in params_to_shrink:
            seg_var = df["segment"].map(seg_vars[col])
            # For segments with <2 products, use global variance
            seg_var = seg_var.fillna(global_vars[col]).clip(lower=1e-6)

            seg_med = df["segment"].map(seg_medians[col])
            # For segments with 1 product, use global median
            seg_med = seg_med.fillna(global_medians[col])

            # Use parameter-specific standard errors when available
            se_col = f"{col}_se"
            if se_col in df.columns:
                product_var = df[se_col].fillna(np.sqrt(fallback_var)).clip(lower=1e-6) ** 2
            else:
                product_var = fallback_var

            w = seg_var / (seg_var + product_var)
            w = w.clip(0, self.max_weight)

            df[f"{col}_shrunk"] = w * df[col] + (1 - w) * seg_med

        # Post-shrinkage winsorization: clip extreme values at 2.5th/97.5th
        # percentiles to prevent outlier coefficients from producing
        # unrealistic demand predictions (Bijmolt et al. 2005: most
        # promotional elasticities fall within a moderate range).
        for col in params_to_shrink:
            col_shrunk = f"{col}_shrunk"
            lo = df[col_shrunk].quantile(0.025)
            hi = df[col_shrunk].quantile(0.975)
            df[col_shrunk] = df[col_shrunk].clip(lo, hi)

        # Economic sign constraint: price elasticity should be non-positive
        # (law of demand). Positive estimates arise from endogeneity and
        # limited shelf-price variation. Clipping to 0 is conservative:
        # it means "no detectable price effect" rather than "anomalous
        # positive effect."
        if "elasticity_shrunk" in df.columns:
            df["elasticity_shrunk"] = df["elasticity_shrunk"].clip(upper=0)

        # AR(1) persistence: clip to [0, 0.95] for stationarity and
        # economic sign (demand should not negatively persist)
        if "demand_persistence_shrunk" in df.columns:
            df["demand_persistence_shrunk"] = df["demand_persistence_shrunk"].clip(0, 0.95)

        # Substitution: clip to non-positive (other-product discounting
        # should cannibalize, not boost, own demand)
        if "substitution_effect_shrunk" in df.columns:
            df["substitution_effect_shrunk"] = df["substitution_effect_shrunk"].clip(upper=0)

        return df

    @staticmethod
    def calibrate_intercepts(
        df: pd.DataFrame, panel: pd.DataFrame
    ) -> pd.DataFrame:
        """Calibrate intercepts so mean prediction matches observed mean demand.

        alpha_cal = ln(mean_Q) - epsilon_shrunk * ln(mean_P) - gamma_shrunk * mean_d
                    - display_eff * mean_display - mailer_eff * mean_mailer
                    - subst_eff * mean_seg_disc_other
                    - store_cov_eff * mean_log_store_coverage
                    - campaign_eff * mean_campaign_active
        
        The AR(1) persistence term does not need calibration because the
        mean deviation of lagged log-demand from its own mean is zero
        over the training set by construction. Mean quantity uses only
        observations after the first (dropped for AR(1) lag), consistent
        with the OLS estimation sample.

        Uses continuous discount_depth for calibration consistency with 
        the updated demand model specification.
        """
        # Compute means from training data, dropping first obs per product
        # (matching the OLS estimation sample which loses 1 obs for AR(1) lag)
        panel_sorted = panel.sort_values(["PRODUCT_ID", "WEEK_NO"])
        # Drop first observation per product (used as lag, not in OLS estimation)
        first_idx = panel_sorted.groupby("PRODUCT_ID").head(1).index
        panel_no_first = panel_sorted.drop(first_idx)

        means = (
            panel_no_first.groupby("PRODUCT_ID")
            .agg(
                mean_qty=("quantity", "mean"),
                mean_price=("avg_price", "mean"),
                mean_disc=("discount_depth", "mean"),
            )
            .reset_index()
        )

        # Add mean display/mailer exposure if available
        display_col = "display_pct" if "display_pct" in panel_no_first.columns else "has_display"
        mailer_col = "mailer_pct" if "mailer_pct" in panel_no_first.columns else "has_mailer"
        if display_col in panel_no_first.columns:
            disp_means = panel_no_first.groupby("PRODUCT_ID")[display_col].mean().rename("mean_display")
            means = means.merge(disp_means, on="PRODUCT_ID", how="left")
            means["mean_display"] = means["mean_display"].fillna(0)
        else:
            means["mean_display"] = 0.0
        if mailer_col in panel_no_first.columns:
            mail_means = panel_no_first.groupby("PRODUCT_ID")[mailer_col].mean().rename("mean_mailer")
            means = means.merge(mail_means, on="PRODUCT_ID", how="left")
            means["mean_mailer"] = means["mean_mailer"].fillna(0)
        else:
            means["mean_mailer"] = 0.0

        # Add mean seg_disc_other if available for substitution calibration
        if "seg_disc_other" in panel_no_first.columns:
            sdo_means = panel_no_first.groupby("PRODUCT_ID")["seg_disc_other"].mean().rename("mean_seg_disc_other")
            means = means.merge(sdo_means, on="PRODUCT_ID", how="left")
            means["mean_seg_disc_other"] = means["mean_seg_disc_other"].fillna(0)
        else:
            means["mean_seg_disc_other"] = 0.0

        # Add mean store coverage if available
        if "log_store_coverage" in panel_no_first.columns:
            sc_means = panel_no_first.groupby("PRODUCT_ID")["log_store_coverage"].mean().rename("mean_log_store_coverage")
            means = means.merge(sc_means, on="PRODUCT_ID", how="left")
            means["mean_log_store_coverage"] = means["mean_log_store_coverage"].fillna(0)
        else:
            means["mean_log_store_coverage"] = 0.0

        # Add mean campaign activity if available
        if "campaign_active" in panel_no_first.columns:
            camp_means = panel_no_first.groupby("PRODUCT_ID")["campaign_active"].mean().rename("mean_campaign_active")
            means = means.merge(camp_means, on="PRODUCT_ID", how="left")
            means["mean_campaign_active"] = means["mean_campaign_active"].fillna(0)
        else:
            means["mean_campaign_active"] = 0.0

        df = df.merge(means, on="PRODUCT_ID", how="left")

        # Absorb mean display/mailer/substitution/store_cov/campaign effects into calibrated intercept
        display_eff = df.get("display_effect_shrunk", df.get("display_effect", pd.Series(np.zeros(len(df))))).fillna(0)
        mailer_eff = df.get("mailer_effect_shrunk", df.get("mailer_effect", pd.Series(np.zeros(len(df))))).fillna(0)
        subst_eff = df.get("substitution_effect_shrunk", df.get("substitution_effect", pd.Series(np.zeros(len(df))))).fillna(0)
        store_cov_eff = df.get("store_coverage_effect_shrunk", df.get("store_coverage_effect", pd.Series(np.zeros(len(df))))).fillna(0)
        campaign_eff = df.get("campaign_effect_shrunk", df.get("campaign_effect", pd.Series(np.zeros(len(df))))).fillna(0)

        df["intercept_calibrated"] = (
            np.log(df["mean_qty"].clip(lower=0.5))
            - df["elasticity_shrunk"] * np.log(df["mean_price"].clip(lower=0.01))
            - df["disc_effect_shrunk"] * df["mean_disc"]
            - display_eff * df["mean_display"]
            - mailer_eff * df["mean_mailer"]
            - subst_eff * df["mean_seg_disc_other"]
            - store_cov_eff * df["mean_log_store_coverage"]
            - campaign_eff * df["mean_campaign_active"]
        )

        # Store base demand (demand at mean price, no discount)
        df["base_demand"] = np.exp(
            df["intercept_calibrated"]
            + df["elasticity_shrunk"] * np.log(df["mean_price"].clip(lower=0.01))
        )
        df["base_price"] = df["mean_price"]

        return df

    @staticmethod
    def compute_phi(df: pd.DataFrame, panel: pd.DataFrame) -> float:
        """Compute log-retransformation bias correction factor.

        phi = sum(Q_obs) / sum(Q_pred_uncorrected)

        Includes lagged demand, substitution, store coverage, and
        campaign terms when available.

        References:
            - Duan (1983), "Smearing Estimate: A Nonparametric
              Retransformation Method"
        """
        total_obs = 0
        total_pred = 0

        for _, row in df.iterrows():
            pid = row["PRODUCT_ID"]
            prod_data = panel[panel["PRODUCT_ID"] == pid].sort_values("WEEK_NO")
            if prod_data.empty:
                continue

            display_col = "display_pct" if "display_pct" in prod_data.columns else "has_display"
            mailer_col = "mailer_pct" if "mailer_pct" in prod_data.columns else "has_mailer"
            disp_eff = row.get("display_effect_shrunk", row.get("display_effect", 0)) or 0
            mail_eff = row.get("mailer_effect_shrunk", row.get("mailer_effect", 0)) or 0
            persistence = row.get("demand_persistence_shrunk", row.get("demand_persistence", 0)) or 0
            subst_eff = row.get("substitution_effect_shrunk", row.get("substitution_effect", 0)) or 0
            store_cov_eff = row.get("store_coverage_effect_shrunk", row.get("store_coverage_effect", 0)) or 0
            campaign_eff = row.get("campaign_effect_shrunk", row.get("campaign_effect", 0)) or 0

            log_pred = (
                row["intercept_calibrated"]
                + row["elasticity_shrunk"] * np.log(prod_data["avg_price"].clip(lower=0.01))
                + row["disc_effect_shrunk"] * prod_data["discount_depth"]
            )
            if display_col in prod_data.columns:
                log_pred = log_pred + disp_eff * prod_data[display_col]
            if mailer_col in prod_data.columns:
                log_pred = log_pred + mail_eff * prod_data[mailer_col]

            # AR(1) term: deviation from product's mean log-demand
            if persistence != 0:
                log_qty = np.log(prod_data["quantity"].clip(lower=0.5))
                lag_log_qty = log_qty.shift(1)
                mean_log_qty = log_qty.mean()
                ar_term = persistence * (lag_log_qty - mean_log_qty)
                ar_term = ar_term.fillna(0)
                log_pred = log_pred + ar_term

            # Substitution term
            if subst_eff != 0 and "seg_disc_other" in prod_data.columns:
                mean_seg_disc = row.get("mean_seg_disc_other", prod_data["seg_disc_other"].mean())
                log_pred = log_pred + subst_eff * (prod_data["seg_disc_other"] - mean_seg_disc)

            # Store coverage effect
            if store_cov_eff != 0 and "log_store_coverage" in prod_data.columns:
                mean_log_sc = row.get("mean_log_store_coverage", prod_data["log_store_coverage"].mean())
                log_pred = log_pred + store_cov_eff * (prod_data["log_store_coverage"] - mean_log_sc)

            # Campaign effect
            if campaign_eff != 0 and "campaign_active" in prod_data.columns:
                mean_camp = row.get("mean_campaign_active", prod_data["campaign_active"].mean())
                log_pred = log_pred + campaign_eff * (prod_data["campaign_active"] - mean_camp)

            total_obs += prod_data["quantity"].sum()
            total_pred += np.exp(log_pred).sum()

        return total_obs / total_pred if total_pred > 0 else 1.0
