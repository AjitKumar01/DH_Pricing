"""Demand model validation and comparison."""

import numpy as np
import pandas as pd
from typing import Dict


class DemandValidator:
    """Validate demand models on held-out test data and compare models."""

    @staticmethod
    def validate_direct(
        product_params: pd.DataFrame,
        test_panel: pd.DataFrame,
        phi: float = 1.0,
    ) -> Dict[str, float]:
        """Validate direct prediction accuracy on test data.

        Includes all estimated terms: price elasticity, discount effect,
        display/mailer, AR(1) demand persistence, cross-product substitution,
        seasonality, store coverage, and campaign effects.
        """
        log_obs_list = []
        log_pred_list = []
        obs_weekly = {}
        pred_weekly = {}

        for _, row in product_params.iterrows():
            pid = row["PRODUCT_ID"]
            prod_test = test_panel[test_panel["PRODUCT_ID"] == pid].sort_values("WEEK_NO")
            if prod_test.empty:
                continue

            log_price = np.log(prod_test["avg_price"].clip(lower=0.01))
            disp_eff = row.get("display_effect_shrunk", row.get("display_effect", 0)) or 0
            mail_eff = row.get("mailer_effect_shrunk", row.get("mailer_effect", 0)) or 0
            persistence = row.get("demand_persistence_shrunk", row.get("demand_persistence", 0)) or 0
            subst_eff = row.get("substitution_effect_shrunk", row.get("substitution_effect", 0)) or 0
            cos_coef = row.get("cos_season_coef", 0) or 0
            sin_coef = row.get("sin_season_coef", 0) or 0
            store_cov_eff = row.get("store_coverage_effect_shrunk", row.get("store_coverage_effect", 0)) or 0
            campaign_eff = row.get("campaign_effect_shrunk", row.get("campaign_effect", 0)) or 0

            display_col = "display_pct" if "display_pct" in prod_test.columns else "has_display"
            mailer_col = "mailer_pct" if "mailer_pct" in prod_test.columns else "has_mailer"

            log_pred = (
                row["intercept_calibrated"]
                + row["elasticity_shrunk"] * log_price
                + row["disc_effect_shrunk"] * prod_test["discount_depth"]
            )
            if display_col in prod_test.columns:
                log_pred = log_pred + disp_eff * prod_test[display_col]
            if mailer_col in prod_test.columns:
                log_pred = log_pred + mail_eff * prod_test[mailer_col]

            # AR(1) demand persistence (deviation from product mean)
            if persistence != 0:
                log_qty_test = np.log(prod_test["quantity"].clip(lower=0.5))
                lag_log_qty = log_qty_test.shift(1)
                mean_log_qty = row.get("mean_log_qty", log_qty_test.mean())
                ar_term = persistence * (lag_log_qty - mean_log_qty)
                ar_term = ar_term.fillna(0)
                log_pred = log_pred + ar_term

            # Cross-product substitution (deviation from training mean)
            if subst_eff != 0 and "seg_disc_other" in prod_test.columns:
                mean_seg_disc = row.get("mean_seg_disc_other", 0)
                log_pred = log_pred + subst_eff * (prod_test["seg_disc_other"] - mean_seg_disc)

            # Seasonality (deviation from training mean)
            if (cos_coef != 0 or sin_coef != 0) and "WEEK_NO" in prod_test.columns:
                week_vals = prod_test["WEEK_NO"]
                cos_val = np.cos(2 * np.pi * week_vals / 52)
                sin_val = np.sin(2 * np.pi * week_vals / 52)
                train_weeks = np.arange(18, 83)
                mean_cos = np.mean(np.cos(2 * np.pi * train_weeks / 52))
                mean_sin = np.mean(np.sin(2 * np.pi * train_weeks / 52))
                log_pred = log_pred + cos_coef * (cos_val - mean_cos) + sin_coef * (sin_val - mean_sin)

            # Store coverage effect
            if store_cov_eff != 0 and "log_store_coverage" in prod_test.columns:
                mean_log_sc = row.get("mean_log_store_coverage", 0)
                log_pred = log_pred + store_cov_eff * (prod_test["log_store_coverage"] - mean_log_sc)

            # Campaign effect
            if campaign_eff != 0 and "campaign_active" in prod_test.columns:
                mean_camp = row.get("mean_campaign_active", 0)
                log_pred = log_pred + campaign_eff * (prod_test["campaign_active"] - mean_camp)

            log_obs = np.log(prod_test["quantity"].clip(lower=0.5))

            log_obs_list.extend(log_obs.tolist())
            log_pred_list.extend(log_pred.tolist())

            # Aggregate by week (use same full prediction for weekly MAPE)
            log_pred_vals = log_pred.values
            for j, (_, tr) in enumerate(prod_test.iterrows()):
                w = tr["WEEK_NO"]
                obs_weekly[w] = obs_weekly.get(w, 0) + tr["quantity"]
                pred_v = phi * np.exp(log_pred_vals[j])
                pred_weekly[w] = pred_weekly.get(w, 0) + pred_v

        log_obs_arr = np.array(log_obs_list)
        log_pred_arr = np.array(log_pred_list)

        # Filter out NaN values
        valid = np.isfinite(log_obs_arr) & np.isfinite(log_pred_arr)
        log_obs_arr = log_obs_arr[valid]
        log_pred_arr = log_pred_arr[valid]

        if len(log_obs_arr) < 2:
            return {
                "log_correlation": float("nan"),
                "log_rmse": float("nan"),
                "log_bias": float("nan"),
                "weekly_mape_pct": float("nan"),
                "n_observations": 0,
            }

        # Log-space metrics
        log_corr = np.corrcoef(log_obs_arr, log_pred_arr)[0, 1]
        log_rmse = np.sqrt(np.mean((log_obs_arr - log_pred_arr) ** 2))
        log_bias = np.mean(log_pred_arr - log_obs_arr)

        # Weekly aggregate MAPE
        weeks = sorted(set(obs_weekly.keys()) & set(pred_weekly.keys()))
        weekly_ape = []
        for w in weeks:
            ape = abs(pred_weekly[w] - obs_weekly[w]) / max(obs_weekly[w], 1)
            weekly_ape.append(ape)

        weekly_mape = np.mean(weekly_ape) * 100 if weekly_ape else float("inf")

        return {
            "log_correlation": log_corr,
            "log_rmse": log_rmse,
            "log_bias": log_bias,
            "weekly_mape_pct": weekly_mape,
            "n_observations": len(log_obs_arr),
        }

    @staticmethod
    def validate_lift(
        product_params: pd.DataFrame,
        test_panel: pd.DataFrame,
    ) -> Dict[str, float]:
        """Validate discount-lift predictions.

        For each product, compare observed vs. predicted demand lift
        from discounting. Uses mean discount depth of discounted
        observations to properly scale the continuous disc_effect
        coefficient. Includes store coverage and campaign effects.
        """
        obs_lifts = []
        pred_lifts = []

        for _, row in product_params.iterrows():
            pid = row["PRODUCT_ID"]
            prod_test = test_panel[test_panel["PRODUCT_ID"] == pid]
            if len(prod_test) < 5:
                continue

            disc = prod_test[prod_test["discount_depth"] > 0.01]
            no_disc = prod_test[prod_test["discount_depth"] <= 0.01]
            if disc.empty or no_disc.empty:
                continue

            obs_lift = disc["quantity"].mean() / max(no_disc["quantity"].mean(), 0.01)

            # Predicted lift: accounts for both the price reduction and
            # the direct discount effect at the mean observed depth
            mean_depth = disc["discount_depth"].mean()
            mean_price_disc = disc["avg_price"].mean()
            mean_price_nodisc = no_disc["avg_price"].mean()
            elast = row.get("elasticity_shrunk", row.get("elasticity", 0))
            disc_eff = row.get("disc_effect_shrunk", row.get("disc_effect", 0))

            # Full predicted lift including price effect
            pred_lift = np.exp(
                elast * (np.log(max(mean_price_disc, 0.01))
                         - np.log(max(mean_price_nodisc, 0.01)))
                + disc_eff * mean_depth
            )

            if not np.isfinite(obs_lift) or not np.isfinite(pred_lift):
                continue

            obs_lifts.append(obs_lift)
            pred_lifts.append(pred_lift)

        if not obs_lifts:
            return {"lift_correlation": float("nan"), "correct_direction_pct": float("nan")}

        obs_arr = np.array(obs_lifts)
        pred_arr = np.array(pred_lifts)

        lift_corr = np.corrcoef(obs_arr, pred_arr)[0, 1]
        correct_dir = np.mean(
            ((obs_arr > 1) & (pred_arr > 1)) | ((obs_arr <= 1) & (pred_arr <= 1))
        ) * 100

        return {
            "lift_correlation": lift_corr,
            "correct_direction_pct": correct_dir,
            "n_products": len(obs_lifts),
        }

    @staticmethod
    def compare_models(
        model_results: Dict[str, pd.DataFrame],
        test_panel: pd.DataFrame,
        segment_mapping: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Compare multiple demand models on test data.

        Args:
            model_results: Dict of model_name -> DataFrame with
                PRODUCT_ID, elasticity, disc_effect, r2 etc.
            test_panel: Test period panel data.
            segment_mapping: Optional segment assignments.

        Returns:
            Comparison DataFrame with metrics per model.
        """
        comparisons = []

        for name, params in model_results.items():
            # Need shrunk params for validation
            if "elasticity_shrunk" not in params.columns:
                params = params.rename(columns={
                    "elasticity": "elasticity_shrunk",
                    "disc_effect": "disc_effect_shrunk",
                })
            if "intercept_calibrated" not in params.columns:
                # Use raw intercept as calibrated if not available
                if "intercept" in params.columns:
                    params = params.rename(columns={"intercept": "intercept_calibrated"})

            direct = DemandValidator.validate_direct(params, test_panel)
            lift = DemandValidator.validate_lift(params, test_panel)

            mean_r2 = params["r2"].mean() if "r2" in params.columns else float("nan")
            median_r2 = params["r2"].median() if "r2" in params.columns else float("nan")

            pct_negative_elast = (
                (params.get("elasticity_shrunk", params.get("elasticity", pd.Series())) < 0).mean() * 100
            )

            comparisons.append({
                "model": name,
                "mean_r2": mean_r2,
                "median_r2": median_r2,
                "log_correlation": direct["log_correlation"],
                "log_rmse": direct["log_rmse"],
                "log_bias": direct["log_bias"],
                "weekly_mape_pct": direct["weekly_mape_pct"],
                "lift_correlation": lift["lift_correlation"],
                "correct_direction_pct": lift["correct_direction_pct"],
                "pct_negative_elasticity": pct_negative_elast,
                "n_products": len(params),
            })

        return pd.DataFrame(comparisons).sort_values("weekly_mape_pct")
