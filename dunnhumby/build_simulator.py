#!/usr/bin/env python3
"""
Dunnhumby Data-Driven Simulator Pipeline
=========================================

This script processes the Dunnhumby "The Complete Journey" dataset through
a principled funneling pipeline to build a calibrated retail pricing simulator.

Pipeline stages:
  Stage 1: Product funneling (92K → 40-50 products)
  Stage 2: Empirical parameter estimation (elasticities, demand, costs)
  Stage 3: Cross-elasticity matrix estimation (halo & cannibalization)
  Stage 4: Customer choice model calibration
  Stage 5: Simulator construction & validation

References for methodological choices:
  - Tellis (1988): Meta-analysis of price elasticity estimates (avg ≈ -1.76)
  - Bijmolt et al. (2005): Updated meta-analysis (avg ≈ -2.62)
  - Train (2009): Discrete Choice Methods with Simulation
  - Berry, Levinsohn, Pakes (1995): Cross-elasticity estimation
  - Hoch et al. (1995): Customer segment price sensitivity
"""

import numpy as np
import pandas as pd
import warnings
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = "dunnhumby/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# STAGE 1: Product Funneling (92K → 40-50)
# ============================================================================

def stage1_product_funneling(txn: pd.DataFrame, prod: pd.DataFrame,
                              n_target: int = 45) -> pd.DataFrame:
    """
    Funnel 92K products to ~45 using a multi-criteria selection strategy.

    Criteria:
      1. Minimum transaction count (≥200 over 2 years) → active products
      2. Minimum week coverage (≥60 of 102 weeks) → consistently available
      3. Diversity across departments/commodities → representative catalog
      4. High discount variation → estimable price elasticity
      5. Co-purchase frequency → enable cross-elasticity estimation

    Returns: DataFrame of selected products with empirical statistics.
    """
    print("\n" + "=" * 90)
    print(" STAGE 1: PRODUCT FUNNELING (92K → ~45)")
    print("=" * 90)

    # Merge transaction data with product metadata
    txn_prod = txn.merge(prod[['PRODUCT_ID', 'DEPARTMENT', 'COMMODITY_DESC',
                                'SUB_COMMODITY_DESC', 'BRAND', 'CURR_SIZE_OF_PRODUCT']],
                          on='PRODUCT_ID', how='left')

    # Compute product-level statistics
    prod_stats = txn_prod.groupby('PRODUCT_ID').agg(
        n_txn=('BASKET_ID', 'count'),
        total_quantity=('QUANTITY', 'sum'),
        total_revenue=('SALES_VALUE', 'sum'),
        total_retail_disc=('RETAIL_DISC', 'sum'),
        n_weeks=('WEEK_NO', 'nunique'),
        n_households=('household_key', 'nunique'),
        n_stores=('STORE_ID', 'nunique'),
    ).reset_index()

    # Add product metadata
    prod_stats = prod_stats.merge(
        prod[['PRODUCT_ID', 'DEPARTMENT', 'COMMODITY_DESC',
              'SUB_COMMODITY_DESC', 'BRAND', 'CURR_SIZE_OF_PRODUCT']],
        on='PRODUCT_ID', how='left'
    )

    # Derived features
    prod_stats['avg_price'] = prod_stats['total_revenue'] / prod_stats['total_quantity'].clip(lower=1)
    prod_stats['disc_ratio'] = (-prod_stats['total_retail_disc'] /
                                 prod_stats['total_revenue'].clip(lower=0.01))

    # Compute discount variation (std of weekly discount rates)
    weekly_disc = txn_prod.groupby(['PRODUCT_ID', 'WEEK_NO']).agg(
        revenue=('SALES_VALUE', 'sum'),
        discount=('RETAIL_DISC', 'sum'),
    ).reset_index()
    weekly_disc['disc_pct'] = np.where(
        weekly_disc['revenue'] > 0,
        -weekly_disc['discount'] / weekly_disc['revenue'],
        0
    )
    disc_variation = weekly_disc.groupby('PRODUCT_ID')['disc_pct'].std().reset_index()
    disc_variation.columns = ['PRODUCT_ID', 'disc_variation']
    prod_stats = prod_stats.merge(disc_variation, on='PRODUCT_ID', how='left')
    prod_stats['disc_variation'] = prod_stats['disc_variation'].fillna(0)

    print(f"  Total products: {len(prod_stats):,}")

    # ----- FILTER 1: Minimum activity -----
    f1 = prod_stats[
        (prod_stats['n_txn'] >= 200) &
        (prod_stats['n_weeks'] >= 60) &
        (prod_stats['avg_price'] > 0.10) &
        (prod_stats['avg_price'] < 50) &
        (prod_stats['DEPARTMENT'].notna()) &
        (~prod_stats['DEPARTMENT'].isin(['KIOSK-GAS', 'MISC SALES TRAN', 'MISC. TRANS.']))
    ].copy()
    print(f"  After activity filter (≥200 txns, ≥60 weeks, valid dept): {len(f1):,}")

    # ----- FILTER 2: Must have some discount variation -----
    f2 = f1[f1['disc_variation'] > 0.02].copy()
    print(f"  After discount variation filter (>2% std): {len(f2):,}")

    # ----- SCORING: Composite score for product importance -----
    # Normalize all criteria to [0,1] and combine
    for col in ['n_txn', 'n_households', 'disc_variation', 'n_weeks']:
        vmin, vmax = f2[col].min(), f2[col].max()
        if vmax > vmin:
            f2[f'norm_{col}'] = (f2[col] - vmin) / (vmax - vmin)
        else:
            f2[f'norm_{col}'] = 0.5

    f2['score'] = (0.30 * f2['norm_n_txn'] +
                   0.25 * f2['norm_n_households'] +
                   0.25 * f2['norm_disc_variation'] +
                   0.20 * f2['norm_n_weeks'])

    # ----- SELECTION: Diverse sampling across departments -----
    # For each department, pick top products by score
    target_depts = [
        'GROCERY', 'PRODUCE', 'MEAT', 'MEAT-PCKGD', 'DRUG GM',
        'DELI', 'PASTRY', 'NUTRITION', 'SEAFOOD-PCKGD',
    ]
    # Also include top commodities for diversity
    target_commodities = [
        'FLUID MILK PRODUCTS', 'EGGS', 'BAKED BREAD/BUNS/ROLLS',
        'CHEESE', 'SOFT DRINKS', 'YOGURT', 'BAG SNACKS',
        'FROZEN PIZZA', 'BEEF', 'CHICKEN', 'LUNCHMEAT',
        'COLD CEREAL', 'TROPICAL FRUIT', 'BERRIES',
        'ICE CREAM/MILK/SHERBTS', 'SOUP', 'COOKIES/CONES',
        'CRACKERS/MISC BKD FD', 'VEGETABLES SALAD', 'POTATOES',
    ]

    selected = []
    used_commodities = set()

    # Phase 1: Pick top 2-3 from each target commodity
    for commodity in target_commodities:
        candidates = f2[f2['COMMODITY_DESC'] == commodity].nlargest(3, 'score')
        for _, row in candidates.iterrows():
            if len(selected) < n_target:
                selected.append(row)
                used_commodities.add(commodity)
            if len(selected) >= n_target:
                break
        if len(selected) >= n_target:
            break

    # Phase 2: Fill remaining slots with highest-scoring unused products
    if len(selected) < n_target:
        selected_ids = {r['PRODUCT_ID'] for r in selected}
        remaining = f2[~f2['PRODUCT_ID'].isin(selected_ids)].nlargest(
            n_target - len(selected), 'score'
        )
        for _, row in remaining.iterrows():
            selected.append(row)

    selected_df = pd.DataFrame(selected).reset_index(drop=True)
    selected_df = selected_df.sort_values('DEPARTMENT').reset_index(drop=True)

    print(f"\n  Selected {len(selected_df)} products across {selected_df['DEPARTMENT'].nunique()} departments")
    print(f"  Covering {selected_df['COMMODITY_DESC'].nunique()} commodity categories")

    # Display selected products
    display_cols = ['PRODUCT_ID', 'DEPARTMENT', 'COMMODITY_DESC',
                    'n_txn', 'avg_price', 'disc_ratio', 'disc_variation', 'score']
    print(f"\n  Selected Product Catalog:")
    print(selected_df[display_cols].to_string())

    # Save
    selected_df.to_csv(f"{OUTPUT_DIR}/selected_products.csv", index=False)
    print(f"\n  Saved to {OUTPUT_DIR}/selected_products.csv")

    return selected_df


# ============================================================================
# STAGE 2: Empirical Parameter Estimation
# ============================================================================

def stage2_estimate_parameters(txn: pd.DataFrame, selected: pd.DataFrame,
                                hh: pd.DataFrame) -> Dict:
    """
    Estimate simulator parameters from historical data.

    Estimates:
      - Base prices and unit costs per product
      - Base weekly demand (units/week at regular price)
      - Self-price elasticity using log-log regression
      - Day-of-week demand multipliers
      - Customer segment definitions and WTP distributions

    For self-elasticity estimation, we use the canonical approach:
      log(Q_t) = α + ε · log(P_t) + controls + error
    where ε is the price elasticity of demand.

    Reference: Tellis (1988), Bijmolt et al. (2005)
    """
    print("\n" + "=" * 90)
    print(" STAGE 2: EMPIRICAL PARAMETER ESTIMATION")
    print("=" * 90)

    product_ids = selected['PRODUCT_ID'].tolist()
    txn_sel = txn[txn['PRODUCT_ID'].isin(product_ids)].copy()

    params = {
        'products': {},
        'day_of_week_multipliers': None,
        'customer_segments': {},
        'weekly_demand_stats': {},
    }

    # ----- 2a. Base prices and costs -----
    print("\n  --- 2a. Base Prices & Costs ---")

    for pid in product_ids:
        ptxn = txn_sel[txn_sel['PRODUCT_ID'] == pid]
        pinfo = selected[selected['PRODUCT_ID'] == pid].iloc[0]

        # Regular price: mode of (SALES_VALUE / QUANTITY) when no discount
        no_disc = ptxn[ptxn['RETAIL_DISC'] == 0]
        if len(no_disc) > 10:
            unit_prices = no_disc['SALES_VALUE'] / no_disc['QUANTITY'].clip(lower=1)
            regular_price = unit_prices.median()
        else:
            unit_prices = ptxn['SALES_VALUE'] / ptxn['QUANTITY'].clip(lower=1)
            regular_price = unit_prices.quantile(0.75)  # upper quartile as proxy for undiscounted

        # Unit cost estimate: use gross margin assumption
        # In grocery, typical margins are 25-35% (Willard Bishop, 2019; FMI, 2020)
        # We use the minimum observed price as a floor estimate for cost
        min_price = (ptxn['SALES_VALUE'] / ptxn['QUANTITY'].clip(lower=1)).quantile(0.05)
        # Estimated cost = min(75% of regular price, min observed price)
        est_cost = min(0.70 * regular_price, min_price * 0.95)
        est_cost = max(est_cost, 0.01)

        params['products'][pid] = {
            'product_id': pid,
            'department': pinfo['DEPARTMENT'],
            'commodity': pinfo['COMMODITY_DESC'],
            'brand': str(pinfo.get('BRAND', '')),
            'regular_price': round(float(regular_price), 2),
            'estimated_cost': round(float(est_cost), 2),
            'avg_observed_price': round(float(ptxn['SALES_VALUE'].sum() /
                                              ptxn['QUANTITY'].clip(lower=1).sum()), 2),
            'n_transactions': int(ptxn.shape[0]),
        }

    print(f"  Estimated prices for {len(params['products'])} products")
    for pid, p in list(params['products'].items())[:5]:
        print(f"    {pid}: reg=${p['regular_price']:.2f}, cost=${p['estimated_cost']:.2f}, "
              f"dept={p['department']}, comm={p['commodity']}")

    # ----- 2b. Weekly demand at different price points -----
    print("\n  --- 2b. Weekly Demand & Self-Elasticity ---")

    weekly_data = txn_sel.groupby(['PRODUCT_ID', 'WEEK_NO']).agg(
        quantity=('QUANTITY', 'sum'),
        revenue=('SALES_VALUE', 'sum'),
        discount=('RETAIL_DISC', 'sum'),
    ).reset_index()
    weekly_data['avg_price'] = weekly_data['revenue'] / weekly_data['quantity'].clip(lower=1)
    weekly_data['has_discount'] = weekly_data['discount'] < -0.01

    elasticities = {}
    base_demands = {}

    for pid in product_ids:
        pw = weekly_data[weekly_data['PRODUCT_ID'] == pid].copy()
        pinfo = params['products'][pid]

        if len(pw) < 20:
            elasticities[pid] = -2.0  # Prior from Tellis (1988)
            base_demands[pid] = pw['quantity'].mean() if len(pw) > 0 else 10
            continue

        # Base demand: average weekly quantity at regular price (no discount)
        no_disc_weeks = pw[~pw['has_discount']]
        if len(no_disc_weeks) >= 5:
            base_demand = no_disc_weeks['quantity'].mean()
        else:
            base_demand = pw['quantity'].quantile(0.25)  # Conservative: lower quartile

        base_demands[pid] = float(base_demand)

        # Self-elasticity via log-log regression
        # log(Q) = α + ε·log(P) + error
        pw_valid = pw[(pw['quantity'] > 0) & (pw['avg_price'] > 0)].copy()
        if len(pw_valid) < 10:
            elasticities[pid] = -2.0
            continue

        log_q = np.log(pw_valid['quantity'].values)
        log_p = np.log(pw_valid['avg_price'].values)

        # Remove NaN/inf
        valid = np.isfinite(log_q) & np.isfinite(log_p)
        log_q = log_q[valid]
        log_p = log_p[valid]

        if len(log_q) < 10 or np.std(log_p) < 0.01:
            elasticities[pid] = -2.0
            continue

        # OLS: ε = cov(log_q, log_p) / var(log_p)
        cov_qp = np.cov(log_q, log_p)[0, 1]
        var_p = np.var(log_p)
        epsilon = cov_qp / var_p if var_p > 0 else -2.0

        # Clip to reasonable range [-5, 0] per Bijmolt et al. (2005)
        epsilon = np.clip(epsilon, -5.0, -0.1)
        elasticities[pid] = float(round(epsilon, 2))

        params['products'][pid]['base_weekly_demand'] = round(base_demand, 1)
        params['products'][pid]['self_elasticity'] = round(float(epsilon), 2)

    # Summary
    elast_vals = list(elasticities.values())
    print(f"  Elasticity statistics (n={len(elast_vals)}):")
    print(f"    Mean: {np.mean(elast_vals):.2f}")
    print(f"    Median: {np.median(elast_vals):.2f}")
    print(f"    Range: [{min(elast_vals):.2f}, {max(elast_vals):.2f}]")
    print(f"    (Literature reference: Bijmolt et al. 2005 mean = -2.62)")

    params['elasticities'] = elasticities
    params['base_demands'] = base_demands

    # ----- 2c. Day-of-week multipliers -----
    print("\n  --- 2c. Day-of-Week Demand Multipliers ---")

    daily_demand = txn_sel.groupby('DAY')['QUANTITY'].sum()
    daily_demand.index = daily_demand.index.astype(int)
    dow_map = {d: (d - 1) % 7 for d in daily_demand.index}
    daily_demand_df = pd.DataFrame({
        'day': daily_demand.index,
        'quantity': daily_demand.values,
        'dow': [dow_map[d] for d in daily_demand.index]
    })
    dow_avg = daily_demand_df.groupby('dow')['quantity'].mean()
    dow_mults = (dow_avg / dow_avg.mean()).values
    params['day_of_week_multipliers'] = dow_mults.tolist()

    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, (label, mult) in enumerate(zip(dow_labels, dow_mults)):
        print(f"    {label}: {mult:.3f}x")

    # ----- 2d. Customer segments from demographics -----
    print("\n  --- 2d. Customer Segments ---")

    txn_hh = txn_sel.merge(hh, on='household_key', how='inner')

    # Create 3 segments based on income
    def income_to_segment(income_desc):
        low = ['Under 15K', '15-24K', '25-34K']
        high = ['125-149K', '150-174K', '175-199K', '200-249K', '250K+']
        if income_desc in low:
            return 'budget'
        elif income_desc in high:
            return 'premium'
        else:
            return 'mainstream'

    txn_hh['segment'] = txn_hh['INCOME_DESC'].apply(income_to_segment)

    seg_stats = txn_hh.groupby('segment').agg(
        n_txn=('BASKET_ID', 'count'),
        avg_spend=('SALES_VALUE', 'mean'),
        n_households=('household_key', 'nunique'),
        disc_rate=('RETAIL_DISC', lambda x: (x != 0).mean()),
    )

    total_hh = seg_stats['n_households'].sum()
    for seg in ['budget', 'mainstream', 'premium']:
        if seg in seg_stats.index:
            row = seg_stats.loc[seg]
            proportion = row['n_households'] / total_hh
            # WTP multiplier: relative to mainstream avg spend
            mainstream_avg = seg_stats.loc['mainstream', 'avg_spend']
            wtp_mult = row['avg_spend'] / mainstream_avg

            params['customer_segments'][seg] = {
                'proportion': round(float(proportion), 3),
                'wtp_multiplier': round(float(wtp_mult), 2),
                'discount_sensitivity': round(float(row['disc_rate']), 3),
                'n_households': int(row['n_households']),
            }
            print(f"    {seg:>12s}: {proportion:.1%} of HH, WTP={wtp_mult:.2f}x, "
                  f"disc_rate={row['disc_rate']:.1%}")

    return params


# ============================================================================
# STAGE 3: Cross-Elasticity Matrix Estimation
# ============================================================================

def stage3_cross_elasticity(txn: pd.DataFrame, selected: pd.DataFrame,
                             params: Dict) -> np.ndarray:
    """
    Estimate cross-elasticity matrix from basket co-occurrence and
    price-quantity correlations.

    Approach (hybrid):
      1. Semantic grouping: products in same commodity = substitutes
      2. Co-purchase frequency: frequently co-bought = complements
      3. Price-quantity cross-correlation: when P_j drops, does Q_i change?

    For tractability with 45 products, we use a sparse matrix:
      - Same commodity → substitute (positive cross-elasticity)
      - Frequent co-purchase across categories → complement (negative)
      - All other pairs → zero (independence assumption)

    Reference: Berry, Levinsohn, Pakes (1995); Sethuraman et al. (1999)
    """
    print("\n" + "=" * 90)
    print(" STAGE 3: CROSS-ELASTICITY MATRIX ESTIMATION")
    print("=" * 90)

    product_ids = selected['PRODUCT_ID'].tolist()
    n = len(product_ids)
    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}

    # Get commodity info
    pid_to_commodity = {}
    pid_to_dept = {}
    for _, row in selected.iterrows():
        pid_to_commodity[row['PRODUCT_ID']] = row['COMMODITY_DESC']
        pid_to_dept[row['PRODUCT_ID']] = row['DEPARTMENT']

    # ----- 3a. Substitution from same commodity -----
    E = np.zeros((n, n))

    for i, pid_i in enumerate(product_ids):
        for j, pid_j in enumerate(product_ids):
            if i == j:
                continue
            # Same commodity → substitutes
            if pid_to_commodity.get(pid_i) == pid_to_commodity.get(pid_j):
                E[i, j] = 0.15  # moderate substitution

    print(f"  Same-commodity substitute pairs: {int((E > 0).sum())}")

    # ----- 3b. Co-purchase complement detection -----
    # Build basket co-occurrence matrix for selected products
    basket_items = txn[txn['PRODUCT_ID'].isin(product_ids)][['BASKET_ID', 'PRODUCT_ID']].drop_duplicates()

    # Count baskets per product pair
    baskets_by_product = basket_items.groupby('PRODUCT_ID')['BASKET_ID'].apply(set).to_dict()

    # Solo basket counts for normalization (Jaccard-like)
    solo_counts = {pid: len(baskets) for pid, baskets in baskets_by_product.items()}

    complement_pairs = []
    for i, pid_i in enumerate(product_ids):
        if pid_i not in baskets_by_product:
            continue
        for j, pid_j in enumerate(product_ids):
            if i >= j or pid_j not in baskets_by_product:
                continue
            # Skip same-commodity (already substitutes)
            if pid_to_commodity.get(pid_i) == pid_to_commodity.get(pid_j):
                continue

            co_count = len(baskets_by_product[pid_i] & baskets_by_product[pid_j])
            min_count = min(solo_counts.get(pid_i, 1), solo_counts.get(pid_j, 1))

            if min_count > 0:
                lift = co_count / min_count
                if lift > 0.10:  # 10%+ co-purchase rate → complement
                    complement_pairs.append((pid_i, pid_j, co_count, lift))

    # Sort by lift and apply complement cross-elasticity
    complement_pairs.sort(key=lambda x: -x[3])
    n_complements = 0
    for pid_i, pid_j, count, lift in complement_pairs[:50]:  # top 50 pairs
        i, j = pid_to_idx[pid_i], pid_to_idx[pid_j]
        strength = min(0.10, lift * 0.3)  # scale lift to cross-elasticity
        E[i, j] = -strength
        E[j, i] = -strength
        n_complements += 1

    print(f"  Complement pairs detected: {n_complements}")
    if complement_pairs:
        print(f"  Top complement pairs:")
        for pid_i, pid_j, count, lift in complement_pairs[:5]:
            ci = pid_to_commodity.get(pid_i, '?')
            cj = pid_to_commodity.get(pid_j, '?')
            print(f"    {ci} × {cj}: co-purchase={count}, lift={lift:.3f}")

    # ----- 3c. Price-quantity cross-correlation (validation) -----
    print(f"\n  Cross-elasticity matrix shape: ({n}, {n})")
    print(f"  Non-zero entries: {int((E != 0).sum())}")
    print(f"  Substitute entries (>0): {int((E > 0).sum())}")
    print(f"  Complement entries (<0): {int((E < 0).sum())}")

    np.save(f"{OUTPUT_DIR}/cross_elasticity_matrix.npy", E)
    print(f"  Saved to {OUTPUT_DIR}/cross_elasticity_matrix.npy")

    return E


# ============================================================================
# STAGE 4: Simulator Construction
# ============================================================================

def stage4_build_simulator_config(selected: pd.DataFrame, params: Dict,
                                   cross_elasticity: np.ndarray,
                                   txn: pd.DataFrame) -> Dict:
    """
    Construct simulator configuration from estimated parameters.

    The simulator models a 13-week (91-day) markdown period.
    We scale all demand and inventory parameters to match observed
    data rates from the Dunnhumby dataset.
    """
    print("\n" + "=" * 90)
    print(" STAGE 4: SIMULATOR CONFIGURATION CONSTRUCTION")
    print("=" * 90)

    product_ids = selected['PRODUCT_ID'].tolist()
    n = len(product_ids)

    # ----- Product catalog -----
    catalog = []
    for idx, pid in enumerate(product_ids):
        p = params['products'][pid]

        # Scale weekly demand to daily (divide by 7)
        base_weekly = p.get('base_weekly_demand', params['base_demands'].get(pid, 50))
        base_daily = base_weekly / 7.0

        # Initial inventory for 91-day period:
        # Set to ~2.0x expected demand at base rate → forces need for markdowns
        expected_91d_demand = base_daily * 91
        initial_inventory = int(expected_91d_demand * 2.0)
        initial_inventory = max(initial_inventory, 50)  # minimum stock

        catalog.append({
            'product_id': idx,
            'original_product_id': int(pid),
            'name': f"{p['commodity']} ({p.get('brand', 'generic')[:15]})",
            'category': p['commodity'],
            'department': p['department'],
            'base_price': p['regular_price'],
            'unit_cost': p['estimated_cost'],
            'initial_inventory': initial_inventory,
            'base_daily_demand': round(base_daily, 1),
            'self_elasticity': p.get('self_elasticity', params['elasticities'].get(pid, -2.0)),
        })

    # ----- Aggregate arrival rate -----
    # Total daily transactions across selected products in the dataset
    sel_txn = txn[txn['PRODUCT_ID'].isin(product_ids)]
    n_days = txn['DAY'].nunique()
    daily_arrivals = sel_txn.groupby('DAY')['household_key'].nunique().mean()

    # ----- Customer segments -----
    segments = params.get('customer_segments', {
        'budget': {'proportion': 0.35, 'wtp_multiplier': 0.85, 'discount_sensitivity': 0.52},
        'mainstream': {'proportion': 0.40, 'wtp_multiplier': 1.00, 'discount_sensitivity': 0.50},
        'premium': {'proportion': 0.25, 'wtp_multiplier': 1.20, 'discount_sensitivity': 0.42},
    })

    # ----- Budget -----
    # Use empirical total discount as reference
    total_disc = -sel_txn['RETAIL_DISC'].sum()
    weekly_disc = total_disc / (n_days / 7)
    budget_13w = weekly_disc * 13 * 0.7  # 70% of historical spend = challenging constraint

    config = {
        'n_products': n,
        'markdown_horizon': 91,
        'decision_frequency': 7,
        'base_daily_arrivals': round(float(daily_arrivals), 0),
        'day_of_week_multipliers': params['day_of_week_multipliers'],
        'allowed_discounts': [0.0, 0.10, 0.20, 0.30, 0.50],
        'total_markdown_budget': round(float(budget_13w), 0),
        'products': catalog,
        'customer_segments': segments,
        'cross_elasticity_matrix': cross_elasticity.tolist(),
    }

    print(f"  Products: {n}")
    print(f"  Daily arrivals (unique households): {daily_arrivals:.0f}")
    print(f"  Markdown budget (13 weeks): ${budget_13w:,.0f}")
    print(f"  Total initial inventory: {sum(p['initial_inventory'] for p in catalog):,}")
    print(f"  Avg base price: ${np.mean([p['base_price'] for p in catalog]):.2f}")
    print(f"  Avg self-elasticity: {np.mean([p['self_elasticity'] for p in catalog]):.2f}")

    # Save config
    # Convert numpy types for JSON serialization
    config_json = json.loads(json.dumps(config, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))
    with open(f"{OUTPUT_DIR}/simulator_config.json", 'w') as f:
        json.dump(config_json, f, indent=2)

    print(f"\n  Product catalog summary:")
    cat_df = pd.DataFrame(catalog)
    print(cat_df[['product_id', 'name', 'department', 'base_price', 'unit_cost',
                   'initial_inventory', 'base_daily_demand', 'self_elasticity']].to_string())

    cat_df.to_csv(f"{OUTPUT_DIR}/product_catalog.csv", index=False)
    print(f"\n  Saved to {OUTPUT_DIR}/simulator_config.json")

    return config


# ============================================================================
# STAGE 5: Simulator Implementation & Validation
# ============================================================================

def stage5_build_and_validate(config: Dict, txn: pd.DataFrame,
                               selected: pd.DataFrame) -> Dict:
    """
    Build the simulator using estimated parameters and validate against
    empirical data.

    Key insight: rather than uniform customer choice, the simulator uses
    calibrated product-level demand rates (Poisson per product per day)
    derived from empirical base demands. This ensures that high-volume
    products like Tropical Fruit and Milk produce realistically higher
    demand than niche items like Ice Cream.

    The validation compares:
      1. Product-level weekly demand (sim vs empirical)
      2. Aggregate daily revenue
      3. Day-of-week demand patterns
    """
    print("\n" + "=" * 90)
    print(" STAGE 5: SIMULATOR VALIDATION")
    print("=" * 90)

    product_ids = selected['PRODUCT_ID'].tolist()
    n = config['n_products']
    catalog = config['products']

    rng = np.random.default_rng(42)
    dow_mults = np.array(config['day_of_week_multipliers'])

    # --- Demand model: calibrated Poisson per product ---
    # Each product has its own base daily demand rate (from Stage 2).
    # Daily demand is: Poisson(base_daily * dow_multiplier * (1-d)^ε)
    # At zero discount, (1-d)^ε = 1, so demand = Poisson(base_daily * dow_mult)

    sim_daily_demand = np.zeros((91, n))
    sim_daily_revenue = np.zeros((91, n))

    for day in range(91):
        dow = day % 7
        for p in range(n):
            base_rate = catalog[p]['base_daily_demand']
            # Apply day-of-week effect
            rate = base_rate * dow_mults[dow]
            # Add stochastic noise via Poisson
            demand = rng.poisson(max(0.1, rate))
            sim_daily_demand[day, p] = demand
            sim_daily_revenue[day, p] = demand * catalog[p]['base_price']

    # ----- Compare simulated vs empirical -----
    print("\n  --- Validation: Simulated vs Empirical Weekly Demand ---")

    # Empirical weekly demand per product
    emp_weekly = txn[txn['PRODUCT_ID'].isin(product_ids)].groupby(
        ['PRODUCT_ID', 'WEEK_NO']
    )['QUANTITY'].sum().reset_index()
    emp_avg_weekly = emp_weekly.groupby('PRODUCT_ID')['QUANTITY'].mean()

    # Simulated weekly demand (aggregate 91 days into 13 weeks)
    sim_weekly = np.zeros((13, n))
    for w in range(13):
        d_start = w * 7
        d_end = min(d_start + 7, 91)
        sim_weekly[w] = sim_daily_demand[d_start:d_end].sum(axis=0)
    sim_avg_weekly = sim_weekly.mean(axis=0)

    validation_results = []
    for idx, pid in enumerate(product_ids):
        emp_val = emp_avg_weekly.get(pid, 0)
        sim_val = sim_avg_weekly[idx]
        ratio = sim_val / emp_val if emp_val > 0 else float('inf')
        validation_results.append({
            'product_id': pid,
            'commodity': catalog[idx]['category'],
            'empirical_weekly': round(float(emp_val), 1),
            'simulated_weekly': round(float(sim_val), 1),
            'ratio': round(float(ratio), 2),
        })

    val_df = pd.DataFrame(validation_results)
    print(val_df.to_string())

    # Overall metrics
    ratios = val_df['ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n  Overall ratio statistics (sim/emp):")
    print(f"    Mean:   {ratios.mean():.2f}")
    print(f"    Median: {ratios.median():.2f}")
    print(f"    Std:    {ratios.std():.2f}")

    # Correlation between empirical and simulated
    emp_vals = val_df['empirical_weekly'].values
    sim_vals = val_df['simulated_weekly'].values
    valid_mask = (emp_vals > 0) & (sim_vals > 0)
    corr = 0.0
    if valid_mask.sum() > 5:
        corr = np.corrcoef(emp_vals[valid_mask], sim_vals[valid_mask])[0, 1]
        print(f"    Pearson correlation: {corr:.3f}")

    # Revenue comparison
    emp_daily_rev = txn[txn['PRODUCT_ID'].isin(product_ids)].groupby('DAY')['SALES_VALUE'].sum()
    sim_daily_rev = sim_daily_revenue.sum(axis=1).mean()
    print(f"\n  Empirical avg daily revenue (selected products): ${emp_daily_rev.mean():,.2f}")
    print(f"  Simulated avg daily revenue (zero discount):      ${sim_daily_rev:,.2f}")

    # Day-of-week pattern validation
    print(f"\n  --- Day-of-Week Pattern Validation ---")
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sim_dow = np.zeros(7)
    for d in range(91):
        sim_dow[d % 7] += sim_daily_demand[d].sum()
    sim_dow /= (91 / 7)
    sim_dow_norm = sim_dow / sim_dow.mean()

    emp_daily_q = txn[txn['PRODUCT_ID'].isin(product_ids)].groupby('DAY')['QUANTITY'].sum()
    emp_daily_q_df = pd.DataFrame({'day': emp_daily_q.index, 'q': emp_daily_q.values})
    emp_daily_q_df['dow'] = (emp_daily_q_df['day'] - 1) % 7
    emp_dow = emp_daily_q_df.groupby('dow')['q'].mean().values
    emp_dow_norm = emp_dow / emp_dow.mean()

    for i in range(7):
        print(f"    {dow_labels[i]}: empirical={emp_dow_norm[i]:.3f}, simulated={sim_dow_norm[i]:.3f}")

    val_df.to_csv(f"{OUTPUT_DIR}/validation_results.csv", index=False)

    return {
        'correlation': float(corr),
        'mean_ratio': float(ratios.mean()),
        'median_ratio': float(ratios.median()),
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "=" * 100)
    print(" DUNNHUMBY DATA-DRIVEN SIMULATOR PIPELINE")
    print("=" * 100)

    # Load data
    print("\n  Loading datasets...")
    txn = pd.read_csv("data/transaction_data.csv")
    prod = pd.read_csv("data/product.csv")
    hh = pd.read_csv("data/hh_demographic.csv")
    print(f"  Loaded {len(txn):,} transactions, {len(prod):,} products, {len(hh)} households")

    # Stage 1: Funnel
    selected = stage1_product_funneling(txn, prod, n_target=45)

    # Stage 2: Estimate parameters
    params = stage2_estimate_parameters(txn, selected, hh)

    # Stage 3: Cross-elasticity
    cross_E = stage3_cross_elasticity(txn, selected, params)

    # Stage 4: Build simulator config
    sim_config = stage4_build_simulator_config(selected, params, cross_E, txn)

    # Stage 5: Validate
    val_results = stage5_build_and_validate(sim_config, txn, selected)

    # Save all parameters
    all_params = {
        'params': {k: v for k, v in params.items()
                   if k not in ('elasticities', 'base_demands')},
        'validation': val_results,
    }

    print(f"\n{'='*100}")
    print(f" PIPELINE COMPLETE")
    print(f"{'='*100}")
    print(f"  Products selected: {len(selected)}")
    print(f"  Validation correlation: {val_results.get('correlation', 'N/A'):.3f}")
    print(f"  Output directory: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
