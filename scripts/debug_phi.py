#!/usr/bin/env python3
"""Debug phi=0 issue for OLS model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.data.loader import DunnhumbyLoader
from src.models.loglog import LogLogDemandModel
from src.models.shrinkage import EmpiricalBayesShrinkage
from src.segmentation.product_seg import ProductSegmenter
from src.segmentation.representative import RepresentativeSelector

loader = DunnhumbyLoader("data")
txn = loader.load_transactions()
panel = loader.build_product_weekly_panel(txn, include_causal=True)
product_features = loader.compute_product_features(txn)

seg_eligible = product_features[product_features["n_transactions"] >= 5]
segmenter = ProductSegmenter(n_clusters=12, random_state=42)
seg_mapping = segmenter.fit(seg_eligible)

panel_ids = set(panel["PRODUCT_ID"].unique())
pf = product_features[product_features["PRODUCT_ID"].isin(panel_ids)]
selector = RepresentativeSelector(n_products=150, min_weeks=30, min_price_cv=0.03)
rep_products = selector.select(pf, seg_mapping)
rep_ids = set(rep_products["PRODUCT_ID"])
rep_panel = panel[panel["PRODUCT_ID"].isin(rep_ids)]
train_panel = rep_panel[rep_panel["WEEK_NO"] <= 82].copy()
rep_seg = seg_mapping[seg_mapping["PRODUCT_ID"].isin(rep_ids)]
train_panel = LogLogDemandModel._build_segment_discount(train_panel, rep_seg)

loglog = LogLogDemandModel()
params = loglog.fit(train_panel, seg_mapping=rep_seg)
print("OLS store/campaign cols:", [c for c in params.columns if "store" in c or "campaign" in c])

shrinkage = EmpiricalBayesShrinkage(max_weight=0.9)
pp = shrinkage.shrink(params, rep_seg)
print("After shrink store/campaign:", [c for c in pp.columns if "store" in c or "campaign" in c])

pp = shrinkage.calibrate_intercepts(pp, train_panel)
print("\nintercept_calibrated:", pp["intercept_calibrated"].describe())

# Check a few predictions manually
pid = pp.iloc[0]["PRODUCT_ID"]
prod_data = train_panel[train_panel["PRODUCT_ID"] == pid].sort_values("WEEK_NO")
row = pp.iloc[0]
log_pred = (
    row["intercept_calibrated"]
    + row["elasticity_shrunk"] * np.log(prod_data["avg_price"].clip(lower=0.01))
    + row["disc_effect_shrunk"] * prod_data["discount_depth"]
)
print(f"\nProduct {pid}: intercept_cal={row['intercept_calibrated']:.3f}")
print(f"  elasticity_shrunk={row['elasticity_shrunk']:.3f}")
print(f"  log_pred range: [{log_pred.min():.1f}, {log_pred.max():.1f}]")
print(f"  exp(log_pred) range: [{np.exp(log_pred.min()):.1f}, {np.exp(log_pred.max()):.1f}]")

# Check store coverage effect
sceff = row.get("store_coverage_effect_shrunk", 0)
print(f"  store_coverage_effect_shrunk={sceff}")
mean_lsc = row.get("mean_log_store_coverage", 0)
print(f"  mean_log_store_coverage={mean_lsc}")
if sceff != 0:
    sc_range = prod_data["log_store_coverage"]
    sc_contrib = sceff * (sc_range - mean_lsc)
    print(f"  store_cov contribution range: [{sc_contrib.min():.2f}, {sc_contrib.max():.2f}]")

# Now compute phi
phi = shrinkage.compute_phi(pp, train_panel)
print(f"\nOLS phi: {phi}")
