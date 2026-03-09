#!/usr/bin/env python3
"""Main experiment script: runs all demand models, compares them,
builds the best simulator, and runs validation + policy analysis.

Usage:
    python scripts/run_experiments.py
"""

import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import DunnhumbyLoader
from src.segmentation.product_seg import ProductSegmenter
from src.segmentation.representative import RepresentativeSelector
from src.models.loglog import LogLogDemandModel
from src.models.random_forest import RandomForestDemandModel
from src.models.gradient_boost import GradientBoostDemandModel
from src.models.neural_net import NeuralNetDemandModel
from src.models.shrinkage import EmpiricalBayesShrinkage
from src.evaluation.validation import DemandValidator
from src.evaluation.policy import PolicyEvaluator
from src.evaluation.sensitivity import SensitivityAnalyzer
from src.simulator.product_level import ProductLevelSimulator


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ── 1. Load Data ──
    print("=" * 60)
    print("STEP 1: Loading Dunnhumby data")
    print("=" * 60)
    loader = DunnhumbyLoader(data_dir)
    txn = loader.load_transactions()
    print(f"  Loaded {len(txn):,} transactions")
    print(f"  Products: {txn['PRODUCT_ID'].nunique():,}")
    print(f"  Weeks: {txn['WEEK_NO'].nunique()}")

    # ── 2. Build Panel ──
    print("\n" + "=" * 60)
    print("STEP 2: Building product-week panel (with causal data)")
    print("=" * 60)
    panel = loader.build_product_weekly_panel(txn, min_weeks=30, min_price_cv=0.03,
                                              include_causal=True)
    print(f"  Panel: {len(panel):,} product-week observations")
    print(f"  Eligible products: {panel['PRODUCT_ID'].nunique():,}")
    if "has_display" in panel.columns:
        print(f"  Obs with display: {panel['has_display'].sum():,} ({panel['has_display'].mean()*100:.1f}%)")
        print(f"  Obs with mailer: {panel['has_mailer'].sum():,} ({panel['has_mailer'].mean()*100:.1f}%)")
        print(f"  Obs with coupon product: {panel['has_coupon'].sum():,} ({panel['has_coupon'].mean()*100:.1f}%)")
    if "log_store_coverage" in panel.columns:
        print(f"  Mean store coverage: {panel['n_stores'].mean():.1f} stores")
    if "campaign_active" in panel.columns:
        n_camp = panel["campaign_active"].sum()
        print(f"  Obs with active campaign: {n_camp:,} ({panel['campaign_active'].mean()*100:.1f}%)")
    if "coupon_redemptions" in panel.columns:
        n_red = (panel["coupon_redemptions"] > 0).sum()
        print(f"  Obs with coupon redemptions: {n_red:,}")
    if "DEPARTMENT" in panel.columns:
        print(f"  Departments represented: {panel['DEPARTMENT'].nunique()}")
    if "buyer_income_index" in panel.columns:
        print(f"  Mean buyer income index: {panel['buyer_income_index'].mean():.2f}")

    # ── 3. Compute Product Features & Segment ──
    print("\n" + "=" * 60)
    print("STEP 3: Product segmentation (K=12)")
    print("=" * 60)
    product_features = loader.compute_product_features(txn)

    # Segment ALL eligible products (not just panel products)
    # for better cluster separation with 35K+ products.
    # Representatives will later be restricted to panel-present products.
    seg_eligible = product_features[product_features["n_transactions"] >= 5].copy()
    print(f"  Segmentation-eligible products: {len(seg_eligible):,}")

    segmenter = ProductSegmenter(n_clusters=12, random_state=42)
    seg_mapping = segmenter.fit(seg_eligible)
    print(f"  Segmented {len(seg_mapping):,} products into 12 clusters")

    # ── 4. Select Representative Products ──
    print("\n" + "=" * 60)
    print("STEP 4: Selecting 150 representative products")
    print("=" * 60)
    # Restrict selection to products that exist in the cleaned panel
    panel_product_ids = set(panel["PRODUCT_ID"].unique())
    panel_features = product_features[
        product_features["PRODUCT_ID"].isin(panel_product_ids)
    ].copy()
    selector = RepresentativeSelector(n_products=150, min_weeks=30, min_price_cv=0.03)
    rep_products = selector.select(panel_features, seg_mapping)
    print(f"  Selected {len(rep_products)} products from {rep_products['segment'].nunique()} segments")
    print(f"  Segment allocation: {rep_products['segment'].value_counts().to_dict()}")

    rep_ids = set(rep_products["PRODUCT_ID"])
    rep_panel = panel[panel["PRODUCT_ID"].isin(rep_ids)].copy()

    # Train-test split
    train_panel = rep_panel[rep_panel["WEEK_NO"] <= 82].copy()
    test_panel = rep_panel[rep_panel["WEEK_NO"] > 82].copy()
    print(f"  Train: {len(train_panel):,} obs, Test: {len(test_panel):,} obs")

    # Build cross-product substitution features on both panels
    # (needed for OLS estimation and phi computation)
    rep_seg_mapping_full = seg_mapping[seg_mapping["PRODUCT_ID"].isin(rep_ids)]
    train_panel = LogLogDemandModel._build_segment_discount(train_panel, rep_seg_mapping_full)
    test_panel = LogLogDemandModel._build_segment_discount(test_panel, rep_seg_mapping_full)

    # ── 5. Fit Multiple Demand Models ──
    print("\n" + "=" * 60)
    print("STEP 5: Fitting demand models")
    print("=" * 60)

    model_results = {}

    # 5a. Log-Log OLS (baseline)
    print("\n  [1/4] Log-Log OLS...")
    t0 = time.time()
    loglog = LogLogDemandModel()
    # Pass segment mapping for cross-product substitution estimation
    rep_seg_mapping = seg_mapping[seg_mapping["PRODUCT_ID"].isin(rep_ids)]
    loglog_params = loglog.fit(train_panel, seg_mapping=rep_seg_mapping)
    print(f"    Fitted {len(loglog_params)} products in {time.time()-t0:.1f}s")
    print(f"    Mean R²: {loglog_params['r2'].mean():.3f}, Median: {loglog_params['r2'].median():.3f}")
    print(f"    Significant elasticities (p<0.05): {(loglog_params['elasticity_pval']<0.05).sum()}")
    if "demand_persistence" in loglog_params.columns:
        n_sig_persist = (loglog_params["demand_persistence_pval"] < 0.05).sum()
        print(f"    Significant demand persistence (p<0.05): {n_sig_persist}")
        print(f"    Mean AR(1) coefficient: {loglog_params['demand_persistence'].mean():.3f}")
    if "substitution_effect" in loglog_params.columns:
        n_sig_sub = (loglog_params["substitution_effect_pval"] < 0.05).sum()
        print(f"    Significant substitution effects (p<0.05): {n_sig_sub}")
        print(f"    Mean substitution effect: {loglog_params['substitution_effect'].mean():.3f}")
    if "display_effect" in loglog_params.columns:
        n_sig_disp = (loglog_params["display_pval"] < 0.05).sum()
        print(f"    Significant display effects: {n_sig_disp}")
        print(f"    Mean display effect: {loglog_params['display_effect'].mean():.3f}")
    if "mailer_effect" in loglog_params.columns:
        n_sig_mail = (loglog_params["mailer_pval"] < 0.05).sum()
        print(f"    Significant mailer effects: {n_sig_mail}")
        print(f"    Mean mailer effect: {loglog_params['mailer_effect'].mean():.3f}")
    if "store_coverage_effect" in loglog_params.columns:
        n_sig_sc = (loglog_params["store_coverage_pval"] < 0.05).sum()
        print(f"    Significant store coverage effects: {n_sig_sc}")
        print(f"    Mean store coverage effect: {loglog_params['store_coverage_effect'].mean():.3f}")
    if "campaign_effect" in loglog_params.columns:
        n_sig_camp = (loglog_params["campaign_pval"] < 0.05).sum()
        print(f"    Significant campaign effects: {n_sig_camp}")
        print(f"    Mean campaign effect: {loglog_params['campaign_effect'].mean():.3f}")

    # 5b. Random Forest
    print("\n  [2/4] Random Forest...")
    t0 = time.time()
    rf = RandomForestDemandModel(n_estimators=100, max_depth=8)
    rf_params = rf.fit(train_panel)
    print(f"    Fitted {len(rf_params)} products in {time.time()-t0:.1f}s")
    print(f"    Mean R²: {rf_params['r2'].mean():.3f}, Median: {rf_params['r2'].median():.3f}")

    # 5c. XGBoost
    print("\n  [3/4] XGBoost...")
    t0 = time.time()
    xgb_model = GradientBoostDemandModel(backend="xgboost", n_estimators=100, max_depth=5)
    xgb_params = xgb_model.fit(train_panel)
    print(f"    Fitted {len(xgb_params)} products in {time.time()-t0:.1f}s")
    print(f"    Mean R²: {xgb_params['r2'].mean():.3f}, Median: {xgb_params['r2'].median():.3f}")

    # 5d. LightGBM
    print("\n  [3b/4] LightGBM...")
    t0 = time.time()
    lgbm_model = GradientBoostDemandModel(backend="lightgbm", n_estimators=100, max_depth=5)
    lgbm_params = lgbm_model.fit(train_panel)
    print(f"    Fitted {len(lgbm_params)} products in {time.time()-t0:.1f}s")
    print(f"    Mean R²: {lgbm_params['r2'].mean():.3f}, Median: {lgbm_params['r2'].median():.3f}")

    # 5e. Neural Network
    print("\n  [4/4] Neural Network...")
    t0 = time.time()
    nn_model = NeuralNetDemandModel(hidden_dims=(32, 16), epochs=200, lr=0.005)
    nn_params = nn_model.fit(train_panel)
    print(f"    Fitted {len(nn_params)} products in {time.time()-t0:.1f}s")
    print(f"    Mean R²: {nn_params['r2'].mean():.3f}, Median: {nn_params['r2'].median():.3f}")

    model_results = {
        "Log-Log OLS": loglog_params,
        "Random Forest": rf_params,
        "XGBoost": xgb_params,
        "LightGBM": lgbm_params,
        "Neural Network": nn_params,
    }

    # ── 6. Apply Shrinkage & Calibration to Best Model ──
    print("\n" + "=" * 60)
    print("STEP 6: Shrinkage + calibration for each model")
    print("=" * 60)

    shrinkage = EmpiricalBayesShrinkage(max_weight=0.9)
    calibrated_models = {}

    for name, params in model_results.items():
        pp = shrinkage.shrink(params, seg_mapping[seg_mapping["PRODUCT_ID"].isin(rep_ids)])
        pp = shrinkage.calibrate_intercepts(pp, train_panel)
        phi = shrinkage.compute_phi(pp, train_panel)
        pp["phi"] = phi
        calibrated_models[name] = pp
        print(f"  {name}: phi={phi:.4f}, shrunk elasticity mean={pp['elasticity_shrunk'].mean():.3f}")

    # ── 7. Model Comparison ──
    print("\n" + "=" * 60)
    print("STEP 7: Model comparison on test data")
    print("=" * 60)

    comparison = DemandValidator.compare_models(calibrated_models, test_panel)
    print("\n" + comparison.to_string(index=False))
    comparison.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)

    # Find the best model: prefer OLS for the log-linear simulator
    # since ML models produce noisy elasticity estimates via finite
    # difference. Among equally viable models, prefer lower MAPE.
    # Require phi > 0.5 (calibration not catastrophically broken).
    viable = comparison[comparison["model"].apply(
        lambda m: calibrated_models[m]["phi"].iloc[0] > 0.5
    )]
    if len(viable) > 0:
        # Prefer OLS for structural compatibility with the simulator's
        # log-linear demand equation. Only override if OLS is catastrophically
        # worse (MAPE > 50%) than the best alternative.
        ols_row = viable[viable["model"] == "Log-Log OLS"]
        if len(ols_row) > 0 and ols_row.iloc[0]["weekly_mape_pct"] < 50:
            best_model_name = "Log-Log OLS"
        else:
            best_model_name = viable.sort_values("log_correlation", ascending=False).iloc[0]["model"]
    else:
        best_model_name = "Log-Log OLS"  # fallback
    print(f"\n  *** Best model for simulator: {best_model_name} ***")

    best_params = calibrated_models[best_model_name]
    best_phi = best_params["phi"].iloc[0]

    # ── 8. Build Simulator with Best Model ──
    print("\n" + "=" * 60)
    print("STEP 8: Building simulator with best model")
    print("=" * 60)

    # Compute display/mailer statistics per product from training data
    for col in ["display_mean", "display_std", "mailer_mean", "mailer_std"]:
        if col not in best_params.columns:
            best_params[col] = 0.0
    for idx, row in best_params.iterrows():
        pid = row["PRODUCT_ID"]
        pt = train_panel[train_panel["PRODUCT_ID"] == pid]
        if not pt.empty:
            if "display_pct" in pt.columns:
                best_params.at[idx, "display_mean"] = pt["display_pct"].mean()
                best_params.at[idx, "display_std"] = pt["display_pct"].std()
            elif "has_display" in pt.columns:
                best_params.at[idx, "display_mean"] = pt["has_display"].mean()
                best_params.at[idx, "display_std"] = pt["has_display"].std()
            if "mailer_pct" in pt.columns:
                best_params.at[idx, "mailer_mean"] = pt["mailer_pct"].mean()
                best_params.at[idx, "mailer_std"] = pt["mailer_pct"].std()
            elif "has_mailer" in pt.columns:
                best_params.at[idx, "mailer_mean"] = pt["has_mailer"].mean()
                best_params.at[idx, "mailer_std"] = pt["has_mailer"].std()

    # Compute budget
    budget = 0
    for _, row in best_params.iterrows():
        pid = row["PRODUCT_ID"]
        prod_train = train_panel[train_panel["PRODUCT_ID"] == pid]
        budget += (
            (prod_train["avg_price"] * prod_train["quantity"]).sum()
            - (prod_train["avg_price"] * (1 - prod_train["discount_depth"]) * prod_train["quantity"]).sum()
        )
    # Scale to 12 periods
    n_train_weeks = train_panel["WEEK_NO"].nunique()
    budget = budget / n_train_weeks * 12
    print(f"  Budget: ${budget:,.0f} for 12 periods")

    sim = ProductLevelSimulator(
        product_params=best_params,
        budget=budget,
        horizon=12,
        max_discount=0.45,
        phi=best_phi,
        noise=True,
        start_week=83,
    )
    print(f"  State dim: {sim.state_dim}, Action dim: {sim.action_dim}")

    # Sanity check
    state = sim.reset(seed=42)
    action = np.zeros(sim.action_dim)  # No discount
    _, rev, _, _ = sim.step(action)
    print(f"  Sanity check (no discount, 1 period): revenue=${rev:,.0f}")

    # ── 9. Policy Evaluation ──
    print("\n" + "=" * 60)
    print("STEP 9: Policy rollout evaluation")
    print("=" * 60)

    evaluator = PolicyEvaluator(sim)
    policies = evaluator.build_standard_policies()
    policy_results = evaluator.evaluate_policies(policies, n_runs=50, seed=42)

    print(f"\n  {'Policy':<35} {'Mean Rev ($)':>12} {'Std':>8} {'vs No Disc':>10}")
    print("  " + "-" * 70)
    no_disc_rev = None
    for r in policy_results:
        if r["policy"] == "No Discount":
            no_disc_rev = r["total_revenue"]
    for r in sorted(policy_results, key=lambda x: -x["total_revenue"]):
        vs = f"+{(r['total_revenue']/no_disc_rev - 1)*100:.1f}%" if no_disc_rev and r["policy"] != "No Discount" else "---"
        print(f"  {r['policy']:<35} ${r['total_revenue']:>10,.0f} {r['revenue_std']:>7,.0f} {vs:>10}")

    # Save policy results
    policy_df = pd.DataFrame([{
        "policy": r["policy"],
        "total_revenue": r["total_revenue"],
        "revenue_std": r["revenue_std"],
        "revenue_ci_lo": r["revenue_ci_lo"],
        "revenue_ci_hi": r["revenue_ci_hi"],
        "total_demand": r["total_demand"],
        "total_discount_cost": r["total_discount_cost"],
    } for r in policy_results])
    policy_df.to_csv(os.path.join(results_dir, "policy_results.csv"), index=False)

    # ── 10. Sensitivity Analysis ──
    print("\n" + "=" * 60)
    print("STEP 10: Per-product sensitivity analysis")
    print("=" * 60)

    analyzer = SensitivityAnalyzer(sim)
    sweep = analyzer.per_product_sweep()
    seg_summary = analyzer.segment_summary(sweep)

    n_benefit = sweep["benefits_from_discount"].sum()
    benefit_df = sweep[sweep["benefits_from_discount"]]
    print(f"  Products benefiting from discount: {n_benefit}/{len(sweep)} ({n_benefit/len(sweep)*100:.1f}%)")
    print(f"  Mean optimal discount: {sweep['optimal_discount'].mean()*100:.1f}%")
    print(f"  Median revenue uplift: {benefit_df['uplift_pct'].median():.1f}%")
    print(f"  Mean revenue uplift: {benefit_df['uplift_pct'].mean():.1f}%")
    print(f"  P25-P75 uplift: {benefit_df['uplift_pct'].quantile(0.25):.1f}% - {benefit_df['uplift_pct'].quantile(0.75):.1f}%")

    sweep.to_csv(os.path.join(results_dir, "sensitivity_results.csv"), index=False)
    seg_summary.to_csv(os.path.join(results_dir, "sensitivity_by_segment.csv"), index=False)

    # Save calibrated product-level params for the best model
    best_params.to_csv(os.path.join(results_dir, "product_level_params.csv"), index=False)

    # Save segment-level elasticity summary
    seg_agg_dict = {
            "n_products": ("PRODUCT_ID", "count"),
            "mean_elasticity": ("elasticity_shrunk", "mean"),
            "median_elasticity": ("elasticity_shrunk", "median"),
            "mean_disc_effect": ("disc_effect_shrunk", "mean"),
            "median_disc_effect": ("disc_effect_shrunk", "median"),
            "mean_base_demand": ("base_demand", "mean"),
            "mean_base_price": ("base_price", "mean"),
    }
    if "demand_persistence_shrunk" in best_params.columns:
        seg_agg_dict["mean_persistence"] = ("demand_persistence_shrunk", "mean")
    if "store_coverage_effect_shrunk" in best_params.columns:
        seg_agg_dict["mean_store_cov_eff"] = ("store_coverage_effect_shrunk", "mean")
    seg_elast = (
        best_params.groupby("segment")
        .agg(**seg_agg_dict)
        .reset_index()
    )
    seg_elast.to_csv(os.path.join(results_dir, "segment_elasticities.csv"), index=False)
    print(f"\n  Segment elasticity summary:")
    print(seg_elast.to_string(index=False))

    # ── 11. Generate Comparison Figure ──
    print("\n" + "=" * 60)
    print("STEP 11: Generating model comparison figure")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # R² comparison
    ax = axes[0, 0]
    models = comparison["model"].tolist()
    r2_vals = comparison["median_r2"].tolist()
    ax.barh(models, r2_vals, color="steelblue")
    ax.set_xlabel("Median R²")
    ax.set_title("In-Sample Fit (Median R²)")

    # Weekly MAPE
    ax = axes[0, 1]
    mape_vals = comparison["weekly_mape_pct"].tolist()
    colors = ["green" if v == min(mape_vals) else "steelblue" for v in mape_vals]
    ax.barh(models, mape_vals, color=colors)
    ax.set_xlabel("Weekly Aggregate MAPE (%)")
    ax.set_title("Out-of-Sample Weekly MAPE")

    # Log Correlation
    ax = axes[1, 0]
    corr_vals = comparison["log_correlation"].tolist()
    ax.barh(models, corr_vals, color="steelblue")
    ax.set_xlabel("Log-Space Pearson r")
    ax.set_title("Test-Set Log Correlation")

    # Lift Correlation
    ax = axes[1, 1]
    lift_vals = comparison["lift_correlation"].tolist()
    ax.barh(models, lift_vals, color="steelblue")
    ax.set_xlabel("Discount-Lift Pearson r")
    ax.set_title("Discount-Lift Correlation")

    plt.suptitle(f"Demand Model Comparison (N={len(rep_ids)} products)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "24_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/24_model_comparison.png")

    # ── 12. Summary ──
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    best_row = comparison[comparison["model"] == best_model_name].iloc[0]
    print(f"\n  Best model: {best_model_name}")
    print(f"  Weekly MAPE: {best_row['weekly_mape_pct']:.1f}%")
    print(f"  Log correlation: {best_row['log_correlation']:.3f}")
    print(f"  Lift correlation: {best_row['lift_correlation']:.3f}")
    print(f"  Phi: {best_phi:.4f}")
    print(f"  State dim: {sim.state_dim}, Action dim: {sim.action_dim}")
    print(f"  Best policy: {sorted(policy_results, key=lambda x: -x['total_revenue'])[0]['policy']}")
    print(f"\n  Results saved to: {results_dir}/")
    print(f"  Figures saved to: {fig_dir}/")


if __name__ == "__main__":
    main()
