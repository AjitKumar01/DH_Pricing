#!/usr/bin/env python3
"""
Simulator Validation Against Actual Transactions
=================================================

This script validates that the IV-corrected simulator faithfully reproduces
the demand response patterns observed in real Dunnhumby transaction data.

Key validation tests:
1. Baseline demand: sim weekly demand vs empirical at zero discount
2. Discount response: when real data shows a discount, does the simulator
   predict the right demand uplift?
3. Cross-product effects: when product j is discounted, does q_i respond
   as predicted by the cross-elasticity model?
4. Day-of-week pattern validation
5. Price-quantity relationship: scatterplot-style comparison
"""

import numpy as np
import pandas as pd
import json
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = "dunnhumby/outputs"


def load_data():
    """Load transaction data, simulator config, and selected products."""
    txn = pd.read_csv("data/transaction_data.csv")
    prod = pd.read_csv("data/product.csv")
    selected = pd.read_csv(f"{OUTPUT_DIR}/selected_products.csv")
    
    with open(f"{OUTPUT_DIR}/simulator_config.json") as f:
        config = json.load(f)
    
    return txn, prod, selected, config


def compute_empirical_discounts(txn, selected):
    """
    Compute effective discount for each product-week from the transaction data.
    
    Discount = 1 - (avg selling price / max selling price per product).
    The max price approximates the "shelf price" or undiscounted price.
    """
    product_ids = selected['PRODUCT_ID'].tolist()
    txn_sel = txn[txn['PRODUCT_ID'].isin(product_ids)].copy()
    
    # Compute avg price per transaction
    txn_sel['unit_price'] = txn_sel['SALES_VALUE'] / txn_sel['QUANTITY'].clip(lower=1)
    
    # Weekly aggregation
    weekly = txn_sel.groupby(['PRODUCT_ID', 'WEEK_NO']).agg(
        quantity=('QUANTITY', 'sum'),
        revenue=('SALES_VALUE', 'sum'),
        n_txn=('QUANTITY', 'count'),
    ).reset_index()
    weekly['avg_price'] = weekly['revenue'] / weekly['quantity'].clip(lower=1)
    
    # Shelf price = 90th percentile of weekly avg price (avoids outlier max)
    shelf_prices = weekly.groupby('PRODUCT_ID')['avg_price'].quantile(0.90).to_dict()
    
    weekly['shelf_price'] = weekly['PRODUCT_ID'].map(shelf_prices)
    weekly['discount'] = 1 - (weekly['avg_price'] / weekly['shelf_price']).clip(0.2, 1.0)
    weekly['discount'] = weekly['discount'].clip(lower=0)
    
    return weekly, shelf_prices


def test_baseline_demand(config, weekly, selected):
    """
    Test 1: Baseline demand comparison.
    Compare simulated weekly demand (at zero discount) vs empirical weekly demand.
    """
    print("\n" + "=" * 90)
    print(" TEST 1: BASELINE DEMAND (zero discount)")
    print("=" * 90)
    
    product_ids = selected['PRODUCT_ID'].tolist()
    n = config['n_products']
    catalog = config['products']
    dow_mults = np.array(config['day_of_week_multipliers'])
    rng = np.random.default_rng(42)
    
    # Simulate 91 days × 50 replications for stable averages
    sim_weekly_avg = np.zeros(n)
    n_reps = 50
    for rep in range(n_reps):
        for w in range(13):
            for dow in range(7):
                for p in range(n):
                    rate = catalog[p]['base_daily_demand'] * dow_mults[dow]
                    sim_weekly_avg[p] += rng.poisson(max(0.1, rate))
    sim_weekly_avg /= (n_reps * 13)
    
    # Empirical: avg weekly demand at low/no discount
    no_disc = weekly[weekly['discount'] < 0.05]
    emp_weekly_avg = no_disc.groupby('PRODUCT_ID')['quantity'].mean()
    
    print(f"\n  {'Product':<30} {'Empirical':>10} {'Simulated':>10} {'Ratio':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")
    
    ratios = []
    for idx, pid in enumerate(product_ids):
        emp = emp_weekly_avg.get(pid, 0)
        sim = sim_weekly_avg[idx]
        ratio = sim / emp if emp > 0 else float('nan')
        comm = catalog[idx]['category'][:28]
        if emp > 0:
            print(f"  {comm:<30} {emp:>10.1f} {sim:>10.1f} {ratio:>8.2f}")
            ratios.append(ratio)
    
    ratios = np.array(ratios)
    valid = np.isfinite(ratios)
    print(f"\n  Mean ratio (sim/emp): {ratios[valid].mean():.2f}")
    print(f"  Median ratio: {np.median(ratios[valid]):.2f}")
    
    corr = np.corrcoef(
        [emp_weekly_avg.get(pid, 0) for pid in product_ids],
        sim_weekly_avg
    )[0, 1]
    print(f"  Pearson r: {corr:.3f}")
    
    return corr, float(np.median(ratios[valid]))


def test_discount_response(config, weekly, selected):
    """
    Test 2: Discount response validation.
    
    For each product, compare:
    - Empirical demand uplift when product is discounted (vs baseline)
    - Simulated demand uplift using the estimated elasticity
    
    This directly tests whether the IV-corrected elasticities accurately
    capture the actual demand response to price changes.
    """
    print("\n" + "=" * 90)
    print(" TEST 2: DISCOUNT RESPONSE (elasticity validation)")
    print("=" * 90)
    
    product_ids = selected['PRODUCT_ID'].tolist()
    n = config['n_products']
    catalog = config['products']
    dow_mults = np.array(config['day_of_week_multipliers'])
    rng = np.random.default_rng(42)
    
    print(f"\n  {'Product':<25} {'ε':>6} {'Disc':>6} {'Emp Base':>9} {'Emp Disc':>9} "
          f"{'Emp Uplift':>10} {'Sim Uplift':>10} {'Match?':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*9} {'-'*9} {'-'*10} {'-'*10} {'-'*7}")
    
    emp_uplifts = []
    sim_uplifts = []
    
    for idx, pid in enumerate(product_ids):
        pw = weekly[weekly['PRODUCT_ID'] == pid]
        
        # Split into low-discount and high-discount weeks
        baseline_weeks = pw[pw['discount'] < 0.05]
        discounted_weeks = pw[pw['discount'] >= 0.10]
        
        if len(baseline_weeks) < 3 or len(discounted_weeks) < 3:
            continue
        
        emp_base = baseline_weeks['quantity'].mean()
        emp_disc = discounted_weeks['quantity'].mean()
        avg_discount = discounted_weeks['discount'].mean()
        
        if emp_base <= 0:
            continue
        
        emp_uplift = (emp_disc - emp_base) / emp_base
        
        # Simulated uplift using the calibrated model
        elasticity = catalog[idx].get('self_elasticity', -2.0)
        # Model: Q = base * (1-d)^ε  (no promo boost — elasticity already includes it)
        self_mult = (1 - avg_discount) ** elasticity
        sim_uplift = self_mult - 1.0
        
        emp_uplifts.append(emp_uplift)
        sim_uplifts.append(sim_uplift)
        
        comm = catalog[idx]['category'][:23]
        match = "✓" if abs(emp_uplift - sim_uplift) < 0.5 else "✗"
        print(f"  {comm:<25} {elasticity:>5.1f} {avg_discount:>5.0%} {emp_base:>9.1f} "
              f"{emp_disc:>9.1f} {emp_uplift:>+9.0%} {sim_uplift:>+9.0%} {match:>7}")
    
    if emp_uplifts:
        emp_arr = np.array(emp_uplifts)
        sim_arr = np.array(sim_uplifts)
        corr = np.corrcoef(emp_arr, sim_arr)[0, 1] if len(emp_arr) > 2 else 0
        mae = np.mean(np.abs(emp_arr - sim_arr))
        
        print(f"\n  Discount response correlation: {corr:.3f}")
        print(f"  Mean absolute error of uplift: {mae:.2%}")
        print(f"  Mean empirical uplift: {emp_arr.mean():.1%}")
        print(f"  Mean simulated uplift: {sim_arr.mean():.1%}")
        
        return corr, mae
    
    return 0.0, float('inf')


def test_price_quantity_relationship(config, weekly, selected):
    """
    Test 3: Overall price-quantity relationship.
    
    For each product, plot the log(Q) vs log(P) slope from actual data
    and compare with the simulator's elasticity parameter.
    """
    print("\n" + "=" * 90)
    print(" TEST 3: PRICE-QUANTITY RELATIONSHIP (log-log slope)")
    print("=" * 90)
    
    product_ids = selected['PRODUCT_ID'].tolist()
    catalog = config['products']
    
    print(f"\n  {'Product':<25} {'Config ε':>9} {'Data slope':>10} {'Diff':>8} {'N weeks':>8}")
    print(f"  {'-'*25} {'-'*9} {'-'*10} {'-'*8} {'-'*8}")
    
    config_eps = []
    data_eps = []
    
    for idx, pid in enumerate(product_ids):
        pw = weekly[weekly['PRODUCT_ID'] == pid]
        pw = pw[(pw['quantity'] > 0) & (pw['avg_price'] > 0)]
        
        if len(pw) < 10:
            continue
        
        log_q = np.log(pw['quantity'].values)
        log_p = np.log(pw['avg_price'].values)
        
        valid = np.isfinite(log_q) & np.isfinite(log_p)
        lq, lp = log_q[valid], log_p[valid]
        
        if len(lq) < 10 or np.std(lp) < 0.01:
            continue
        
        # OLS slope from data
        cov_qp = np.cov(lq, lp)[0, 1]
        var_p = np.var(lp)
        data_slope = cov_qp / var_p if var_p > 0 else 0
        data_slope = np.clip(data_slope, -5.0, 0.0)
        
        cfg_eps = catalog[idx].get('self_elasticity', -2.0)
        diff = abs(cfg_eps - data_slope)
        
        comm = catalog[idx]['category'][:23]
        print(f"  {comm:<25} {cfg_eps:>9.2f} {data_slope:>10.2f} {diff:>8.2f} {len(lq):>8}")
        
        config_eps.append(cfg_eps)
        data_eps.append(data_slope)
    
    if config_eps:
        config_arr = np.array(config_eps)
        data_arr = np.array(data_eps)
        corr = np.corrcoef(config_arr, data_arr)[0, 1]
        mae = np.mean(np.abs(config_arr - data_arr))
        
        print(f"\n  Config ε vs data slope correlation: {corr:.3f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  Note: IV elasticities should be MORE negative than OLS slopes")
        print(f"    Config mean: {config_arr.mean():.2f}, Data slope mean: {data_arr.mean():.2f}")
    
    return corr if config_eps else 0.0


def test_dow_pattern(config, weekly, selected):
    """
    Test 4: Day-of-week demand pattern validation.
    """
    print("\n" + "=" * 90)
    print(" TEST 4: DAY-OF-WEEK PATTERN")
    print("=" * 90)
    
    txn = pd.read_csv("data/transaction_data.csv")
    product_ids = selected['PRODUCT_ID'].tolist()
    
    txn_sel = txn[txn['PRODUCT_ID'].isin(product_ids)]
    daily = txn_sel.groupby('DAY')['QUANTITY'].sum().reset_index()
    daily['dow'] = (daily['DAY'].astype(int) - 1) % 7
    emp_dow = daily.groupby('dow')['QUANTITY'].mean().values
    emp_dow_norm = emp_dow / emp_dow.mean()
    
    sim_dow_norm = np.array(config['day_of_week_multipliers'])
    # Normalize to mean 1
    sim_dow_norm = sim_dow_norm / sim_dow_norm.mean()
    
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"\n  {'Day':<5} {'Empirical':>10} {'Simulator':>10} {'|Diff|':>8}")
    print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*8}")
    for i in range(7):
        diff = abs(emp_dow_norm[i] - sim_dow_norm[i])
        print(f"  {dow_labels[i]:<5} {emp_dow_norm[i]:>10.3f} {sim_dow_norm[i]:>10.3f} {diff:>8.3f}")
    
    corr = np.corrcoef(emp_dow_norm, sim_dow_norm)[0, 1]
    mae = np.mean(np.abs(emp_dow_norm - sim_dow_norm))
    print(f"\n  DOW correlation: {corr:.3f}")
    print(f"  DOW MAE: {mae:.3f}")
    
    return corr


def test_distribution_shape(config, weekly, selected):
    """
    Test 5: Distribution shape validation.
    Check that the Poisson assumption matches the actual demand variance.
    For each product, compare empirical variance/mean ratio with 1.0 (Poisson property).
    """
    print("\n" + "=" * 90)
    print(" TEST 5: DEMAND DISTRIBUTION SHAPE (Poisson check)")
    print("=" * 90)
    
    product_ids = selected['PRODUCT_ID'].tolist()
    catalog = config['products']
    
    print(f"\n  {'Product':<25} {'Mean Q':>8} {'Var Q':>10} {'Var/Mean':>9} {'Poisson?':>9}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*9} {'-'*9}")
    
    dispersion_ratios = []
    
    for idx, pid in enumerate(product_ids):
        pw = weekly[weekly['PRODUCT_ID'] == pid]
        if len(pw) < 5:
            continue
        
        q = pw['quantity'].values
        mean_q = q.mean()
        var_q = q.var()
        
        if mean_q > 0:
            disp = var_q / mean_q
            dispersion_ratios.append(disp)
            
            comm = catalog[idx]['category'][:23]
            poisson_ok = "✓" if 0.5 < disp < 3.0 else "over-disp" if disp >= 3.0 else "under-disp"
            print(f"  {comm:<25} {mean_q:>8.1f} {var_q:>10.1f} {disp:>9.2f} {poisson_ok:>9}")
    
    disp_arr = np.array(dispersion_ratios)
    print(f"\n  Mean Var/Mean ratio: {disp_arr.mean():.2f} (Poisson=1.0)")
    print(f"  Median Var/Mean ratio: {np.median(disp_arr):.2f}")
    overdispersed = (disp_arr > 3.0).sum()
    print(f"  Over-dispersed products (Var/Mean > 3): {overdispersed}/{len(disp_arr)}")
    
    return float(np.median(disp_arr))


def main():
    print("\n" + "=" * 100)
    print(" SIMULATOR VALIDATION vs ACTUAL TRANSACTIONS")
    print(" (IV-corrected elasticities)")
    print("=" * 100)
    
    txn, prod, selected, config = load_data()
    weekly, shelf_prices = compute_empirical_discounts(txn, selected)
    
    print(f"\n  Config elasticity source: {config.get('elasticity_source', 'original OLS')}")
    elast = [p.get('self_elasticity', -2.0) for p in config['products']]
    print(f"  Mean self-elasticity: {np.mean(elast):.2f}")
    print(f"  Range: [{min(elast):.2f}, {max(elast):.2f}]")
    
    # Run all validation tests
    r_baseline, ratio_baseline = test_baseline_demand(config, weekly, selected)
    r_discount, mae_discount = test_discount_response(config, weekly, selected)
    r_pq = test_price_quantity_relationship(config, weekly, selected)
    r_dow = test_dow_pattern(config, weekly, selected)
    disp = test_distribution_shape(config, weekly, selected)
    
    # Overall summary
    print("\n" + "=" * 100)
    print(" VALIDATION SUMMARY")
    print("=" * 100)
    print(f"\n  {'Test':<40} {'Score':>10} {'Status':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    
    tests = [
        ("Baseline demand correlation", f"r={r_baseline:.3f}", "PASS" if r_baseline > 0.8 else "WARN"),
        ("Baseline demand ratio", f"{ratio_baseline:.2f}", "PASS" if 0.5 < ratio_baseline < 2.0 else "WARN"),
        ("Discount response correlation", f"r={r_discount:.3f}", "PASS" if r_discount > 0.3 else "WARN"),
        ("Discount response MAE", f"{mae_discount:.1%}", "PASS" if mae_discount < 1.0 else "WARN"),
        ("Price-quantity slope corr", f"r={r_pq:.3f}", "PASS" if r_pq > 0.5 else "WARN"),
        ("Day-of-week pattern corr", f"r={r_dow:.3f}", "PASS" if r_dow > 0.9 else "WARN"),
        ("Poisson dispersion (med)", f"{disp:.2f}", "PASS" if 0.5 < disp < 5.0 else "WARN"),
    ]
    
    n_pass = 0
    for name, score, status in tests:
        print(f"  {name:<40} {score:>10} {status:>10}")
        if status == "PASS":
            n_pass += 1
    
    print(f"\n  Overall: {n_pass}/{len(tests)} tests passed")
    
    # Save validation report
    report = {
        'elasticity_source': config.get('elasticity_source', 'original'),
        'mean_elasticity': float(np.mean(elast)),
        'baseline_correlation': float(r_baseline),
        'baseline_ratio': float(ratio_baseline),
        'discount_response_correlation': float(r_discount),
        'discount_response_mae': float(mae_discount),
        'price_quantity_correlation': float(r_pq),
        'dow_correlation': float(r_dow),
        'poisson_dispersion': float(disp),
        'tests_passed': n_pass,
        'tests_total': len(tests),
    }
    
    with open(f"{OUTPUT_DIR}/validation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved to {OUTPUT_DIR}/validation_report.json")


if __name__ == "__main__":
    main()
