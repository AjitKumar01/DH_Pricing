#!/usr/bin/env python3
"""Complete logical consistency review of the pricing simulator pipeline.

Mirrors the EXACT pipeline from run_experiments.py, then validates each
step for logical correctness.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import silhouette_score, silhouette_samples

from src.data.loader import DunnhumbyLoader
from src.segmentation.product_seg import ProductSegmenter
from src.segmentation.representative import RepresentativeSelector
from src.models.loglog import LogLogDemandModel
from src.models.shrinkage import EmpiricalBayesShrinkage
from src.evaluation.validation import DemandValidator

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
loader = DunnhumbyLoader(data_dir)

issues = []

# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: DATA LOADING + CLEANING")
print("=" * 70)
txn = loader.load_transactions()
print(f"Transactions after cleaning: {len(txn):,}")
print(f"Products: {txn['PRODUCT_ID'].nunique():,}")

# Verify no returns or zero-price txns remain
assert (txn["QUANTITY"] > 0).all(), "Returns still present!"
assert (txn["SALES_VALUE"] > 0).all(), "Negative sales still present!"
assert (txn["base_price"] > 0).all(), "Zero base_price still present!"
print("  [OK] No returns, negative sales, or zero base_price")

# Check discount_depth bounds
assert (txn["discount_depth"] >= 0).all(), "Negative discounts!"
assert (txn["discount_depth"] <= 1).all(), "Discounts > 100%!"
print("  [OK] Discount depth in [0, 1]")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: PANEL CONSTRUCTION")
print("=" * 70)
panel = loader.build_product_weekly_panel(txn, min_weeks=30, min_price_cv=0.03, include_causal=True)
print(f"Panel: {len(panel):,} obs, {panel['PRODUCT_ID'].nunique():,} products")

# Verify no zero/negative quantities
assert (panel["quantity"] > 0).all(), "Zero quantities in panel!"
print("  [OK] All quantities > 0")

# Check for extreme quantity outliers
max_qty = panel["quantity"].max()
p999 = panel["quantity"].quantile(0.999)
print(f"  Max quantity: {max_qty:.0f}, P99.9: {p999:.0f}")
if max_qty > 100000:
    issues.append(f"WARN: Max quantity {max_qty:.0f} still very high")
else:
    print("  [OK] No extreme quantity outliers")

# Check log-transformability
assert panel["avg_price"].min() > 0, "Zero prices in panel!"
print("  [OK] All prices > 0 (safe for log transform)")

# Discount variation per product
disc_var = panel.groupby("PRODUCT_ID")["discount_depth"].std()
zero_disc_var = (disc_var < 0.001).sum()
print(f"  Products with zero discount variation: {zero_disc_var}/{panel['PRODUCT_ID'].nunique()}")
if zero_disc_var > panel["PRODUCT_ID"].nunique() * 0.1:
    issues.append(f"INFO: {zero_disc_var} panel products have zero discount variation")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: SEGMENTATION")
print("=" * 70)
product_features = loader.compute_product_features(txn)
seg_eligible = product_features[product_features["n_transactions"] >= 5].copy()
print(f"Segmentation-eligible: {len(seg_eligible):,}")

segmenter = ProductSegmenter(n_clusters=12, random_state=42)
seg_mapping = segmenter.fit(seg_eligible)

# Check cluster quality
X = seg_eligible[ProductSegmenter.FEATURE_COLS].values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
labels = segmenter.kmeans.predict(X_scaled)
sil = silhouette_score(X_scaled, labels)
sil_samples_arr = silhouette_samples(X_scaled, labels)
neg_sil = (sil_samples_arr < 0).sum()

print(f"  Silhouette score: {sil:.3f}")
print(f"  Negative silhouette: {neg_sil} ({neg_sil/len(sil_samples_arr)*100:.1f}%)")

seg_counts = seg_mapping["segment"].value_counts().sort_index()
print(f"  Segment sizes: min={seg_counts.min()}, max={seg_counts.max()}")
print(f"  Segments: {dict(seg_counts)}")

if sil < 0.10:
    issues.append(f"CRITICAL: Very low silhouette ({sil:.3f}) — clusters not meaningful")
elif sil < 0.20:
    issues.append(f"INFO: Low silhouette ({sil:.3f}) — moderate cluster separation")
else:
    print(f"  [OK] Silhouette {sil:.3f} acceptable for shrinkage grouping")

# Stability check
from sklearn.metrics import adjusted_rand_score
ari_scores = []
for seed in [0, 123, 99]:
    km_alt = ProductSegmenter(n_clusters=12, random_state=seed)
    km_alt.fit(seg_eligible)
    alt_labels = km_alt.kmeans.predict(X_scaled)
    ari = adjusted_rand_score(labels, alt_labels)
    ari_scores.append(ari)
print(f"  Stability (ARI): {min(ari_scores):.3f} – {max(ari_scores):.3f}")
if min(ari_scores) < 0.7:
    issues.append("CRITICAL: Unstable clusters across seeds")
else:
    print("  [OK] Clusters stable across seeds")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: REPRESENTATIVE SELECTION")
print("=" * 70)
# Restrict to panel-present products (matching run_experiments.py)
panel_product_ids = set(panel["PRODUCT_ID"].unique())
panel_features = product_features[
    product_features["PRODUCT_ID"].isin(panel_product_ids)
].copy()

selector = RepresentativeSelector(n_products=150, min_weeks=30, min_price_cv=0.03)
rep_products = selector.select(panel_features, seg_mapping)
rep_ids = set(rep_products["PRODUCT_ID"])
print(f"Selected: {len(rep_products)}, from {rep_products['segment'].nunique()} segments")

# Verify ALL selected products are in the panel
rep_panel = panel[panel["PRODUCT_ID"].isin(rep_ids)]
rep_in_panel = rep_panel["PRODUCT_ID"].nunique()
missing = rep_ids - set(rep_panel["PRODUCT_ID"].unique())
print(f"  In panel: {rep_in_panel}/{len(rep_ids)}")
if missing:
    issues.append(f"CRITICAL: {len(missing)} representative products NOT in panel!")
    print(f"  [FAIL] Missing: {missing}")
else:
    print("  [OK] All representatives are in the panel")

# Check minimum per segment
rep_seg_counts = rep_products.groupby("segment")["PRODUCT_ID"].count()
singletons = (rep_seg_counts < 2).sum()
if singletons > 0:
    issues.append(f"CRITICAL: {singletons} segments have < 2 representative products")
else:
    print(f"  [OK] All segments have >= 2 products (min: {rep_seg_counts.min()})")

# Check obs per product in train period
train_panel = rep_panel[rep_panel["WEEK_NO"] <= 82]
test_panel = rep_panel[rep_panel["WEEK_NO"] > 82]
obs_per_prod = train_panel.groupby("PRODUCT_ID").size()
print(f"  Train obs per product: min={obs_per_prod.min()}, median={obs_per_prod.median():.0f}, max={obs_per_prod.max()}")
low_obs = (obs_per_prod < 10).sum()
if low_obs > 0:
    issues.append(f"WARN: {low_obs} products have < 10 train observations")
else:
    print("  [OK] All products have >= 10 train observations")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: OLS DEMAND MODEL")
print("=" * 70)
loglog = LogLogDemandModel()
params = loglog.fit(train_panel)
print(f"Fitted: {len(params)} products")

# Check model quality
print(f"  Mean R²: {params['r2'].mean():.3f}")
print(f"  Median R²: {params['r2'].median():.3f}")

sig_elast = (params["elasticity_pval"] < 0.05).sum()
sig_disc = (params["disc_effect_pval"] < 0.05).sum()
print(f"  Significant elasticities: {sig_elast}/{len(params)} ({sig_elast/len(params)*100:.0f}%)")
print(f"  Significant disc effects: {sig_disc}/{len(params)} ({sig_disc/len(params)*100:.0f}%)")

# Check disc_effect sign (should be mostly positive)
pos_disc = (params["disc_effect"] > 0).sum()
print(f"  Positive disc_effect: {pos_disc}/{len(params)} ({pos_disc/len(params)*100:.0f}%)")
if pos_disc < len(params) * 0.4:
    issues.append(f"WARN: Only {pos_disc/len(params)*100:.0f}% positive disc effects (expect ~50%+)")

# Check for products where disc_effect was unidentified (zero discount var)
rep_disc_var = train_panel.groupby("PRODUCT_ID")["discount_depth"].std()
zero_disc_in_rep = (rep_disc_var < 0.001).sum()
print(f"  Rep products with zero discount variation in train: {zero_disc_in_rep}")
if zero_disc_in_rep > 0:
    issues.append(f"WARN: {zero_disc_in_rep} rep products have zero discount variation — disc_effect unidentified")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: SHRINKAGE + CALIBRATION")
print("=" * 70)
shrinkage = EmpiricalBayesShrinkage(max_weight=0.9)

# Need segment mapping restricted to rep products
rep_seg = seg_mapping[seg_mapping["PRODUCT_ID"].isin(rep_ids)]
pp = shrinkage.shrink(params, rep_seg)
pp = shrinkage.calibrate_intercepts(pp, train_panel)
phi = shrinkage.compute_phi(pp, train_panel)
print(f"  Phi: {phi:.4f}")

if phi < 0.5 or phi > 2.0:
    issues.append(f"CRITICAL: Phi={phi:.4f} far from 1 — calibration broken")
elif phi < 0.8 or phi > 1.2:
    issues.append(f"INFO: Phi={phi:.4f} — moderate bias, correctable")
else:
    print(f"  [OK] Phi near 1")

# Check shrunk parameters
print(f"  Elasticity shrunk: mean={pp['elasticity_shrunk'].mean():.3f}, median={pp['elasticity_shrunk'].median():.3f}")
print(f"  Disc effect shrunk: mean={pp['disc_effect_shrunk'].mean():.3f}, median={pp['disc_effect_shrunk'].median():.3f}")
print(f"  Disc effect shrunk range: [{pp['disc_effect_shrunk'].min():.2f}, {pp['disc_effect_shrunk'].max():.2f}]")

# Check calibrated intercepts
assert pp["intercept_calibrated"].notna().all(), "NaN calibrated intercepts!"
print("  [OK] No NaN in calibrated intercepts")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: VALIDATION")
print("=" * 70)
pp["phi"] = phi
direct = DemandValidator.validate_direct(pp, test_panel, phi=phi)
lift = DemandValidator.validate_lift(pp, test_panel)

print(f"  Weekly MAPE: {direct['weekly_mape_pct']:.1f}%")
print(f"  Log correlation: {direct['log_correlation']:.3f}")
print(f"  Lift correlation: {lift['lift_correlation']:.3f}")
print(f"  Correct direction: {lift['correct_direction_pct']:.1f}%")

if direct["weekly_mape_pct"] > 50:
    issues.append(f"CRITICAL: Weekly MAPE {direct['weekly_mape_pct']:.1f}% too high")
elif direct["weekly_mape_pct"] > 30:
    issues.append(f"WARN: Weekly MAPE {direct['weekly_mape_pct']:.1f}% — moderate accuracy")
else:
    print(f"  [OK] MAPE reasonable")

if lift["lift_correlation"] < 0.3:
    issues.append(f"WARN: Low lift correlation ({lift['lift_correlation']:.3f}) — discount effect predictions weak")
else:
    print(f"  [OK] Lift correlation adequate")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: SIMULATOR CONSISTENCY")
print("=" * 70)
from src.simulator.product_level import ProductLevelSimulator

# Compute budget (same as pipeline)
budget = 0
for _, row in pp.iterrows():
    pid = row["PRODUCT_ID"]
    pt = train_panel[train_panel["PRODUCT_ID"] == pid]
    budget += (pt["avg_price"] * pt["quantity"]).sum() - (pt["avg_price"] * (1 - pt["discount_depth"]) * pt["quantity"]).sum()
n_train_weeks = train_panel["WEEK_NO"].nunique()
budget = budget / n_train_weeks * 12

sim = ProductLevelSimulator(product_params=pp, budget=budget, horizon=12, max_discount=0.45, phi=phi, noise=False)
print(f"  N products: {sim.N}")
print(f"  State dim: {sim.state_dim}, Action dim: {sim.action_dim}")
print(f"  Budget: ${budget:,.0f}")

# Test: no discount should produce reasonable revenue
state = sim.reset(seed=42)
action = np.zeros(sim.action_dim)
_, rev_no_disc, _, info = sim.step(action)
print(f"  No-discount revenue (1 period): ${rev_no_disc:,.0f}")

# Test: max discount should increase demand
action = np.concatenate([np.ones(sim.N), np.full(sim.N, 0.2)])
sim.reset(seed=42)
_, rev_disc, _, info2 = sim.step(action)
print(f"  20% discount revenue (1 period): ${rev_disc:,.0f}")

if rev_disc <= rev_no_disc * 0.5:
    issues.append("CRITICAL: 20% discount destroys revenue — model dynamics inverted")
elif rev_disc <= rev_no_disc:
    print(f"  [INFO] 20% discount decreases revenue — discount effects too weak for some products")
else:
    print(f"  [OK] 20% discount increases revenue by {(rev_disc/rev_no_disc - 1)*100:.1f}%")

# Test demand magnitudes: should be comparable to training data
demands_no_disc = info["per_product_demand"]
train_mean_qty = train_panel.groupby("PRODUCT_ID")["quantity"].mean()
calib_check = []
for i, (_, row) in enumerate(pp.iterrows()):
    pid = row["PRODUCT_ID"]
    if pid in train_mean_qty.index:
        ratio = demands_no_disc[i] / train_mean_qty[pid]
        calib_check.append(ratio)

calib_arr = np.array(calib_check)
print(f"  Demand calibration (sim/observed): mean={calib_arr.mean():.3f}, median={np.median(calib_arr):.3f}")
if np.median(calib_arr) < 0.5 or np.median(calib_arr) > 2.0:
    issues.append(f"CRITICAL: Simulator demand off by {np.median(calib_arr):.1f}x from observed")
else:
    print(f"  [OK] Simulator demand calibrated within 2x of observed")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

critical = [i for i in issues if i.startswith("CRITICAL")]
warns = [i for i in issues if i.startswith("WARN")]
infos = [i for i in issues if i.startswith("INFO")]

if critical:
    print(f"\n  {len(critical)} CRITICAL issues:")
    for i in critical:
        print(f"    - {i}")

if warns:
    print(f"\n  {len(warns)} WARNINGS:")
    for i in warns:
        print(f"    - {i}")

if infos:
    print(f"\n  {len(infos)} INFO items:")
    for i in infos:
        print(f"    - {i}")

if not issues:
    print("\n  ALL CHECKS PASSED. Pipeline is logically consistent.")
else:
    print(f"\n  Total: {len(critical)} critical, {len(warns)} warnings, {len(infos)} info")

if not critical:
    print("\n  VERDICT: Pipeline produces a logically sound simulator.")
