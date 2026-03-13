#!/usr/bin/env python3
"""
Advanced Elasticity Estimation & Cross-Elasticity Calibration
==============================================================

Addresses four identified limitations:
1. Endogeneity in elasticity estimation → IV using display/mailer instruments
2. Time-varying elasticity → seasonal decomposition
3. Data-driven cross-elasticity magnitudes → price-quantity cross-correlations
4. Improved simulator realism

Uses causal_data.csv (display, mailer) as instrumental variables following
the approach in Berry, Levinsohn, Pakes (1995) and Nevo (2001).
"""

import numpy as np
import pandas as pd
import warnings
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = "dunnhumby/outputs"


def run_iv_elasticity(txn, causal, selected):
    """
    Estimate price elasticities using Instrumental Variables (2SLS).
    
    Instruments: display and mailer promotions from causal_data.csv.
    
    The key insight from BLP (1995) is that OLS elasticity estimates are
    biased because of simultaneous equation bias: high demand causes
    high prices (or retailers discount slow movers, biasing ε upward).
    
    Display and mailer are valid instruments because:
    - Relevance: they are correlated with price (promoted items get discounts)
    - Exclusion: they affect demand only through price, not directly
      (this is debatable — displays also affect attention, so we note this caveat)
    
    2SLS procedure:
      Stage 1: log(P_t) = α + β₁·display_t + β₂·mailer_t + v_t
      Stage 2: log(Q_t) = γ + ε·log(P̂_t) + u_t
    where P̂ is the fitted value from Stage 1.
    """
    print("\n" + "=" * 90)
    print(" IV ELASTICITY ESTIMATION (2SLS with display/mailer instruments)")
    print("=" * 90)
    
    product_ids = selected['PRODUCT_ID'].tolist()
    
    # Aggregate causal data to product-week level (across stores)
    causal_sel = causal[causal['PRODUCT_ID'].isin(product_ids)].copy()
    
    # Convert display/mailer to binary
    causal_sel['is_display'] = (causal_sel['display'].astype(str) != '0').astype(int)
    causal_sel['is_mailer'] = (causal_sel['mailer'].astype(str) != '0').astype(int)
    
    # Aggregate to product-week: fraction of stores with display/mailer
    causal_weekly = causal_sel.groupby(['PRODUCT_ID', 'WEEK_NO']).agg(
        display_frac=('is_display', 'mean'),
        mailer_frac=('is_mailer', 'mean'),
    ).reset_index()
    
    # Weekly transaction data
    txn_sel = txn[txn['PRODUCT_ID'].isin(product_ids)]
    weekly_data = txn_sel.groupby(['PRODUCT_ID', 'WEEK_NO']).agg(
        quantity=('QUANTITY', 'sum'),
        revenue=('SALES_VALUE', 'sum'),
    ).reset_index()
    weekly_data['avg_price'] = weekly_data['revenue'] / weekly_data['quantity'].clip(lower=1)
    
    # Merge
    merged = weekly_data.merge(causal_weekly, on=['PRODUCT_ID', 'WEEK_NO'], how='inner')
    
    ols_elasticities = {}
    iv_elasticities = {}
    iv_diagnostics = {}
    
    for pid in product_ids:
        pm = merged[merged['PRODUCT_ID'] == pid].copy()
        
        if len(pm) < 20:
            ols_elasticities[pid] = -2.0
            iv_elasticities[pid] = -2.0
            continue
        
        pm = pm[(pm['quantity'] > 0) & (pm['avg_price'] > 0)].copy()
        if len(pm) < 20:
            ols_elasticities[pid] = -2.0
            iv_elasticities[pid] = -2.0
            continue
        
        log_q = np.log(pm['quantity'].values)
        log_p = np.log(pm['avg_price'].values)
        
        # OLS estimate
        valid = np.isfinite(log_q) & np.isfinite(log_p)
        lq, lp = log_q[valid], log_p[valid]
        
        if len(lq) < 10 or np.std(lp) < 0.01:
            ols_elasticities[pid] = -2.0
            iv_elasticities[pid] = -2.0
            continue
        
        cov_qp = np.cov(lq, lp)[0, 1]
        var_p = np.var(lp)
        ols_eps = np.clip(cov_qp / var_p if var_p > 0 else -2.0, -5.0, -0.1)
        ols_elasticities[pid] = float(round(ols_eps, 2))
        
        # IV (2SLS) estimate
        display = pm['display_frac'].values[valid]
        mailer = pm['mailer_frac'].values[valid]
        
        # Check instrument relevance
        Z = np.column_stack([np.ones(len(display)), display, mailer])
        
        if np.std(display) < 0.01 and np.std(mailer) < 0.01:
            # No instrument variation — fall back to OLS
            iv_elasticities[pid] = ols_elasticities[pid]
            continue
        
        # Stage 1: regress log(P) on instruments
        try:
            ZtZ_inv = np.linalg.inv(Z.T @ Z)
            beta_1 = ZtZ_inv @ Z.T @ lp
            log_p_hat = Z @ beta_1
            
            # First-stage F-statistic (instrument relevance)
            ss_residual_1 = np.sum((lp - log_p_hat) ** 2)
            ss_total_1 = np.sum((lp - np.mean(lp)) ** 2)
            r2_1 = 1 - ss_residual_1 / ss_total_1 if ss_total_1 > 0 else 0
            n = len(lp)
            k = Z.shape[1]
            f_stat = (r2_1 / (k - 1)) / ((1 - r2_1) / (n - k)) if r2_1 < 1 and (n - k) > 0 else 0
            
            # Stage 2: regress log(Q) on log(P_hat)
            X2 = np.column_stack([np.ones(len(log_p_hat)), log_p_hat])
            XtX2_inv = np.linalg.inv(X2.T @ X2)
            beta_2 = XtX2_inv @ X2.T @ lq
            iv_eps = float(np.clip(beta_2[1], -5.0, -0.1))
            
            iv_elasticities[pid] = round(iv_eps, 2)
            iv_diagnostics[pid] = {
                'f_stat': round(float(f_stat), 1),
                'r2_first_stage': round(float(r2_1), 3),
                'n_obs': int(n),
            }
        except np.linalg.LinAlgError:
            iv_elasticities[pid] = ols_elasticities[pid]
    
    # Compare OLS vs IV
    ols_vals = [ols_elasticities[pid] for pid in product_ids]
    iv_vals = [iv_elasticities[pid] for pid in product_ids]
    
    print(f"\n  OLS elasticities:  mean={np.mean(ols_vals):.2f}, median={np.median(ols_vals):.2f}")
    print(f"  IV  elasticities:  mean={np.mean(iv_vals):.2f}, median={np.median(iv_vals):.2f}")
    
    # Hausman test: difference between OLS and IV
    paired = [(ols_elasticities[pid], iv_elasticities[pid]) 
              for pid in product_ids 
              if pid in iv_diagnostics]
    if paired:
        ols_arr = np.array([p[0] for p in paired])
        iv_arr = np.array([p[1] for p in paired])
        diff = iv_arr - ols_arr
        print(f"\n  Endogeneity bias (IV - OLS):")
        print(f"    Mean: {diff.mean():.3f}")
        print(f"    Std:  {diff.std():.3f}")
        bias_dir = "more negative" if diff.mean() < 0 else "less negative"
        print(f"    → IV estimates are {bias_dir} than OLS on average")
        if abs(diff.mean()) > 0.3:
            print(f"    → Meaningful endogeneity detected")
        else:
            print(f"    → Modest endogeneity — OLS may be acceptable")
    
    # F-stat summary (instrument strength)
    f_stats = [d['f_stat'] for d in iv_diagnostics.values()]
    if f_stats:
        print(f"\n  First-stage F-statistics (Stock & Yogo threshold: F > 10):")
        print(f"    Mean: {np.mean(f_stats):.1f}")
        print(f"    Median: {np.median(f_stats):.1f}")
        print(f"    Products with F > 10 (strong instruments): "
              f"{sum(1 for f in f_stats if f > 10)}/{len(f_stats)}")
        print(f"    Products with F > 4 (moderate instruments): "
              f"{sum(1 for f in f_stats if f > 4)}/{len(f_stats)}")
    
    # Show products with largest OLS-IV discrepancy
    print(f"\n  Largest OLS-IV discrepancies (endogeneity bias evidence):")
    diffs = [(pid, ols_elasticities[pid], iv_elasticities[pid],
              abs(iv_elasticities[pid] - ols_elasticities[pid]))
             for pid in product_ids if pid in iv_diagnostics]
    diffs.sort(key=lambda x: -x[3])
    for pid, ols, iv, d in diffs[:10]:
        comm = selected[selected['PRODUCT_ID'] == pid]['COMMODITY_DESC'].iloc[0]
        diag = iv_diagnostics[pid]
        print(f"    {comm[:25]:<25} OLS={ols:>5.2f}  IV={iv:>5.2f}  diff={d:.2f}  F={diag['f_stat']:.1f}")
    
    return ols_elasticities, iv_elasticities, iv_diagnostics


def estimate_seasonal_elasticity(txn, selected):
    """
    Estimate season-dependent elasticity to check if constant elasticity
    assumption holds.
    
    Split the 102 weeks into quarters and estimate elasticity per quarter.
    If elasticities vary significantly across seasons, the constant model
    is a strong simplification.
    """
    print("\n" + "=" * 90)
    print(" SEASONAL ELASTICITY ANALYSIS")
    print("=" * 90)
    
    product_ids = selected['PRODUCT_ID'].tolist()
    txn_sel = txn[txn['PRODUCT_ID'].isin(product_ids)].copy()
    
    # Define quarters (roughly 25 weeks each for 102-week span)
    quarters = {
        'Q1 (wk 1-25)': (1, 25),
        'Q2 (wk 26-50)': (26, 50),
        'Q3 (wk 51-75)': (51, 75),
        'Q4 (wk 76-102)': (76, 102),
    }
    
    weekly_data = txn_sel.groupby(['PRODUCT_ID', 'WEEK_NO']).agg(
        quantity=('QUANTITY', 'sum'),
        revenue=('SALES_VALUE', 'sum'),
    ).reset_index()
    weekly_data['avg_price'] = weekly_data['revenue'] / weekly_data['quantity'].clip(lower=1)
    
    seasonal_results = []
    product_seasonal = {}
    
    for q_name, (w_start, w_end) in quarters.items():
        q_data = weekly_data[(weekly_data['WEEK_NO'] >= w_start) & 
                            (weekly_data['WEEK_NO'] <= w_end)]
        
        elasticities = []
        for pid in product_ids:
            pw = q_data[q_data['PRODUCT_ID'] == pid]
            pw = pw[(pw['quantity'] > 0) & (pw['avg_price'] > 0)]
            
            if len(pw) < 5:
                continue
            
            log_q = np.log(pw['quantity'].values)
            log_p = np.log(pw['avg_price'].values)
            valid = np.isfinite(log_q) & np.isfinite(log_p)
            log_q, log_p = log_q[valid], log_p[valid]
            
            if len(log_q) < 5 or np.std(log_p) < 0.01:
                continue
            
            cov_qp = np.cov(log_q, log_p)[0, 1]
            var_p = np.var(log_p)
            eps = np.clip(cov_qp / var_p if var_p > 0 else -2.0, -5.0, -0.1)
            elasticities.append(eps)
            
            if pid not in product_seasonal:
                product_seasonal[pid] = {}
            product_seasonal[pid][q_name] = eps
        
        if elasticities:
            seasonal_results.append({
                'quarter': q_name,
                'mean_eps': float(np.mean(elasticities)),
                'median_eps': float(np.median(elasticities)),
                'std_eps': float(np.std(elasticities)),
                'n_products': len(elasticities),
            })
    
    print(f"\n  {'Quarter':<20} {'Mean ε':>10} {'Median ε':>10} {'Std':>8} {'N':>5}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*5}")
    for r in seasonal_results:
        print(f"  {r['quarter']:<20} {r['mean_eps']:>10.2f} {r['median_eps']:>10.2f} "
              f"{r['std_eps']:>8.2f} {r['n_products']:>5}")
    
    # Test for significant seasonal variation
    if len(seasonal_results) >= 2:
        means = [r['mean_eps'] for r in seasonal_results]
        seasonal_range = max(means) - min(means)
        print(f"\n  Seasonal range of mean elasticity: {seasonal_range:.2f}")
        if seasonal_range < 0.5:
            print(f"  → Small variation: constant elasticity assumption is reasonable")
        elif seasonal_range < 1.0:
            print(f"  → Moderate variation: constant elasticity is a simplification")
        else:
            print(f"  → Large variation: consider time-varying elasticity model")
    
    # Products with most seasonal variation
    print(f"\n  Products with largest seasonal elasticity variation:")
    pid_ranges = []
    for pid, seasons in product_seasonal.items():
        if len(seasons) >= 3:
            vals = list(seasons.values())
            rng = max(vals) - min(vals)
            comm = selected[selected['PRODUCT_ID'] == pid]['COMMODITY_DESC'].iloc[0]
            pid_ranges.append((pid, comm, rng, vals))
    pid_ranges.sort(key=lambda x: -x[2])
    for pid, comm, rng, vals in pid_ranges[:8]:
        val_str = ", ".join(f"{v:.2f}" for v in vals)
        print(f"    {comm[:25]:<25} range={rng:.2f}  [{val_str}]")
    
    return seasonal_results, product_seasonal


def estimate_cross_elasticity_magnitudes(txn, selected):
    """
    Estimate cross-elasticity magnitudes from price-quantity correlations
    instead of using fixed semantic values.
    
    For each product pair (i,j), compute:
      ε_ij ≈ Cov(log Q_i, log P_j) / Var(log P_j)
    
    This gives data-driven magnitudes for substitution and complementarity.
    """
    print("\n" + "=" * 90)
    print(" DATA-DRIVEN CROSS-ELASTICITY MAGNITUDES")
    print("=" * 90)
    
    product_ids = selected['PRODUCT_ID'].tolist()
    n = len(product_ids)
    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    
    # Get commodity info
    pid_to_commodity = {}
    for _, row in selected.iterrows():
        pid_to_commodity[row['PRODUCT_ID']] = row['COMMODITY_DESC']
    
    # Build weekly price and quantity matrices
    txn_sel = txn[txn['PRODUCT_ID'].isin(product_ids)]
    weekly_data = txn_sel.groupby(['PRODUCT_ID', 'WEEK_NO']).agg(
        quantity=('QUANTITY', 'sum'),
        revenue=('SALES_VALUE', 'sum'),
    ).reset_index()
    weekly_data['avg_price'] = weekly_data['revenue'] / weekly_data['quantity'].clip(lower=1)
    
    # Pivot to matrices
    weeks = sorted(weekly_data['WEEK_NO'].unique())
    week_to_idx = {w: i for i, w in enumerate(weeks)}
    Q = np.full((len(weeks), n), np.nan)
    P = np.full((len(weeks), n), np.nan)
    
    for _, row in weekly_data.iterrows():
        pid = row['PRODUCT_ID']
        if pid not in pid_to_idx:
            continue
        idx = pid_to_idx[pid]
        week_idx = week_to_idx[row['WEEK_NO']]
        Q[week_idx, idx] = row['quantity']
        P[week_idx, idx] = row['avg_price']
    
    # Compute cross-elasticity estimates
    cross_E_data = np.zeros((n, n))
    significance = np.zeros((n, n))
    
    for i in range(n):
        log_qi = np.log(np.where(Q[:, i] > 0, Q[:, i], np.nan))
        valid_i = np.isfinite(log_qi)
        
        for j in range(n):
            if i == j:
                continue
            
            log_pj = np.log(np.where(P[:, j] > 0, P[:, j], np.nan))
            valid_j = np.isfinite(log_pj)
            valid = valid_i & valid_j
            
            if valid.sum() < 15:
                continue
            
            lq = log_qi[valid]
            lp = log_pj[valid]
            
            if np.std(lp) < 0.01:
                continue
            
            cov_qp = np.cov(lq, lp)[0, 1]
            var_p = np.var(lp)
            eps_ij = cov_qp / var_p if var_p > 0 else 0.0
            
            # Clip to reasonable range
            eps_ij = np.clip(eps_ij, -1.0, 1.0)
            cross_E_data[i, j] = eps_ij
            
            # Simple significance: |correlation| > threshold
            if len(lq) > 2:
                corr = np.corrcoef(lq, lp)[0, 1]
                significance[i, j] = abs(corr) if np.isfinite(corr) else 0
    
    # Summary
    nonzero = cross_E_data[cross_E_data != 0]
    print(f"  Estimated {len(nonzero)} non-zero cross-elasticity values")
    if len(nonzero) > 0:
        print(f"  Mean: {nonzero.mean():.3f}")
        print(f"  Range: [{nonzero.min():.3f}, {nonzero.max():.3f}]")
    
    # Significant pairs (|correlation| > 0.3)
    sig_mask = significance > 0.3
    sig_subs = (cross_E_data > 0.05) & sig_mask
    sig_comps = (cross_E_data < -0.05) & sig_mask
    print(f"\n  Significant substitutes (ε > 0.05, |r| > 0.3): {sig_subs.sum()}")
    print(f"  Significant complements (ε < -0.05, |r| > 0.3): {sig_comps.sum()}")
    
    # Show top significant pairs
    print(f"\n  Top data-driven substitutes:")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sig_subs[i, j] or sig_subs[j, i]:
                avg_eps = (cross_E_data[i, j] + cross_E_data[j, i]) / 2
                avg_sig = (significance[i, j] + significance[j, i]) / 2
                ci = pid_to_commodity.get(product_ids[i], '?')
                cj = pid_to_commodity.get(product_ids[j], '?')
                pairs.append((ci, cj, avg_eps, avg_sig))
    pairs.sort(key=lambda x: -x[2])
    for ci, cj, eps, sig in pairs[:10]:
        print(f"    {ci[:20]:<20} × {cj[:20]:<20} ε={eps:+.3f}  |r|={sig:.2f}")
    
    print(f"\n  Top data-driven complements:")
    pairs_c = []
    for i in range(n):
        for j in range(i + 1, n):
            if sig_comps[i, j] or sig_comps[j, i]:
                avg_eps = (cross_E_data[i, j] + cross_E_data[j, i]) / 2
                avg_sig = (significance[i, j] + significance[j, i]) / 2
                ci = pid_to_commodity.get(product_ids[i], '?')
                cj = pid_to_commodity.get(product_ids[j], '?')
                pairs_c.append((ci, cj, avg_eps, avg_sig))
    pairs_c.sort(key=lambda x: x[2])
    for ci, cj, eps, sig in pairs_c[:10]:
        print(f"    {ci[:20]:<20} × {cj[:20]:<20} ε={eps:+.3f}  |r|={sig:.2f}")
    
    # Compare with existing semantic matrix
    try:
        old_matrix = np.load(f"{OUTPUT_DIR}/cross_elasticity_matrix.npy")
        print(f"\n  Comparison with semantic matrix:")
        old_nonzero = old_matrix[old_matrix != 0]
        print(f"    Semantic: {len(old_nonzero)} non-zero, mean={old_nonzero.mean():.3f}")
        print(f"    Data-driven: {len(nonzero)} non-zero, mean={nonzero.mean():.3f}")
        
        # Where do they agree/disagree?
        agree_sub = ((old_matrix > 0) & (cross_E_data > 0) & sig_mask).sum()
        agree_comp = ((old_matrix < 0) & (cross_E_data < 0) & sig_mask).sum()
        disagree = ((old_matrix > 0) & (cross_E_data < -0.05) & sig_mask).sum() + \
                   ((old_matrix < 0) & (cross_E_data > 0.05) & sig_mask).sum()
        print(f"    Agreement (same sign, significant): {agree_sub + agree_comp}")
        print(f"    Disagreement (opposite sign, significant): {disagree}")
    except FileNotFoundError:
        pass
    
    # Save the data-driven matrix
    np.save(f"{OUTPUT_DIR}/cross_elasticity_data_driven.npy", cross_E_data)
    np.save(f"{OUTPUT_DIR}/cross_elasticity_significance.npy", significance)
    
    return cross_E_data, significance


def create_hybrid_elasticities(ols_elast, iv_elast, iv_diag, product_ids):
    """
    Create final elasticity vector using IV where instruments are strong,
    OLS otherwise.
    
    Decision rule:
    - If F > 10: use IV estimate (strong instruments, Stock & Yogo 2005)
    - If 4 < F < 10: average of OLS and IV (moderate instruments)  
    - If F < 4 or no IV: use OLS (weak instruments unreliable)
    """
    print("\n" + "=" * 90)
    print(" HYBRID ELASTICITY CONSTRUCTION")
    print("=" * 90)
    
    hybrid = {}
    source = {}
    
    for pid in product_ids:
        ols_val = ols_elast.get(pid, -2.0)
        iv_val = iv_elast.get(pid, -2.0)
        diag = iv_diag.get(pid, None)
        
        if diag is None:
            hybrid[pid] = ols_val
            source[pid] = 'OLS (no IV)'
        elif diag['f_stat'] > 10:
            hybrid[pid] = iv_val
            source[pid] = f'IV (F={diag["f_stat"]:.0f})'
        elif diag['f_stat'] > 4:
            hybrid[pid] = round((ols_val + iv_val) / 2, 2)
            source[pid] = f'Hybrid (F={diag["f_stat"]:.0f})'
        else:
            hybrid[pid] = ols_val
            source[pid] = f'OLS (weak F={diag["f_stat"]:.0f})'
    
    print(f"\n  Source breakdown:")
    for s_type, count in pd.Series(list(source.values())).value_counts().head(10).items():
        print(f"    {s_type}: {count}")
    
    hybrid_vals = list(hybrid.values())
    print(f"\n  Final hybrid elasticities: mean={np.mean(hybrid_vals):.2f}, median={np.median(hybrid_vals):.2f}")
    
    return hybrid, source


def create_hybrid_cross_elasticity(semantic_matrix, data_matrix, sig_matrix, threshold=0.3):
    """
    Create final cross-elasticity matrix blending semantic knowledge
    with data-driven estimates.
    
    Strategy:
    - Start with semantic structure (substitutes/complements)
    - Where data-driven estimates are significant, use their magnitudes
    - Where data disagrees with semantic, flag but keep semantic (domain knowledge)
    - Where data finds new pairs not in semantic, add if significant
    """
    print("\n" + "=" * 90)
    print(" HYBRID CROSS-ELASTICITY MATRIX")
    print("=" * 90)
    
    n = semantic_matrix.shape[0]
    hybrid = semantic_matrix.copy()
    
    adjusted = 0
    new_pairs = 0
    disagreements = 0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            sem = semantic_matrix[i, j]
            dat = data_matrix[i, j]
            sig = sig_matrix[i, j]
            
            if sig < threshold:
                # Not significant — keep semantic
                continue
            
            if sem != 0 and dat != 0:
                # Both have values — check agreement
                if np.sign(sem) == np.sign(dat):
                    # Same sign — use data magnitude with semantic sign
                    hybrid[i, j] = np.sign(sem) * abs(dat)
                    adjusted += 1
                else:
                    # Disagreement — keep semantic but note
                    disagreements += 1
            elif sem == 0 and abs(dat) > 0.05:
                # New pair found by data — add 
                hybrid[i, j] = dat
                new_pairs += 1
    
    print(f"  Adjusted magnitudes (data-calibrated): {adjusted}")
    print(f"  New pairs from data: {new_pairs}")
    print(f"  Disagreements (kept semantic): {disagreements}")
    
    old_nonzero = (semantic_matrix != 0).sum()
    new_nonzero = (hybrid != 0).sum()
    print(f"  Entries: {old_nonzero} → {new_nonzero}")
    
    # Save
    np.save(f"{OUTPUT_DIR}/cross_elasticity_hybrid.npy", hybrid)
    
    return hybrid


def main():
    print("\n" + "=" * 100)
    print(" ADVANCED ELASTICITY ANALYSIS — ADDRESSING LIMITATIONS")
    print("=" * 100)
    
    # Load data
    print("\n  Loading data...")
    txn = pd.read_csv("data/transaction_data.csv")
    selected = pd.read_csv(f"{OUTPUT_DIR}/selected_products.csv")
    product_ids = selected['PRODUCT_ID'].tolist()
    
    print(f"  Loaded {len(txn):,} transactions, {len(selected)} selected products")
    
    # 1. IV Elasticity
    print("\n  Loading causal data (display/mailer instruments)...")
    causal = pd.read_csv("data/causal_data.csv")
    ols_elast, iv_elast, iv_diag = run_iv_elasticity(txn, causal, selected)
    del causal  # Free memory
    
    # 2. Seasonal elasticity
    seasonal, product_seasonal = estimate_seasonal_elasticity(txn, selected)
    
    # 3. Data-driven cross-elasticity
    cross_E_data, sig = estimate_cross_elasticity_magnitudes(txn, selected)
    
    # 4. Construct hybrid elasticities
    hybrid_elast, hybrid_source = create_hybrid_elasticities(
        ols_elast, iv_elast, iv_diag, product_ids)
    
    # 5. Construct hybrid cross-elasticity matrix
    try:
        semantic_matrix = np.load(f"{OUTPUT_DIR}/cross_elasticity_matrix.npy")
        hybrid_cross = create_hybrid_cross_elasticity(
            semantic_matrix, cross_E_data, sig)
    except FileNotFoundError:
        print("  No semantic matrix found — using data-driven only")
        hybrid_cross = cross_E_data
        np.save(f"{OUTPUT_DIR}/cross_elasticity_hybrid.npy", hybrid_cross)
    
    # 6. Update simulator config with new elasticities
    config_path = f"{OUTPUT_DIR}/simulator_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Save old config as backup
    with open(f"{OUTPUT_DIR}/simulator_config_pre_iv.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Update elasticities in config — both the list and each product entry
    old_elast = [p.get('self_elasticity', -2.0) for p in config['products']]
    new_elast = [hybrid_elast[pid] for pid in product_ids]
    config['elasticities'] = new_elast
    config['elasticity_source'] = 'hybrid_iv_ols'
    
    # Write into each product's self_elasticity field
    for i, pid in enumerate(product_ids):
        config['products'][i]['self_elasticity'] = new_elast[i]
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n  Elasticity update summary:")
    if old_elast:
        old_mean = np.mean(old_elast)
        new_mean = np.mean(new_elast)
        print(f"    Old mean: {old_mean:.2f}")
        print(f"    New mean: {new_mean:.2f}")
        print(f"    Change: {new_mean - old_mean:+.2f}")
    
    # Save comprehensive results
    results = {
        'ols_elasticities': {str(k): v for k, v in ols_elast.items()},
        'iv_elasticities': {str(k): v for k, v in iv_elast.items()},
        'hybrid_elasticities': {str(k): v for k, v in hybrid_elast.items()},
        'hybrid_source': {str(k): v for k, v in hybrid_source.items()},
        'iv_diagnostics': {str(k): v for k, v in iv_diag.items()},
        'seasonal_elasticity': seasonal,
        'n_cross_elast_nonzero': int((hybrid_cross != 0).sum()) if hybrid_cross is not None else 0,
    }
    
    with open(f"{OUTPUT_DIR}/advanced_elasticity_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    
    print(f"\n" + "=" * 100)
    print(f" SUMMARY")
    print(f"=" * 100)
    print(f"\n  1. IV Estimation: {len(iv_diag)} products with valid instruments")
    print(f"  2. Seasonal Analysis: {len(seasonal)} quarters analyzed")
    print(f"  3. Cross-Elasticity: data-driven + semantic hybrid created")
    print(f"  4. Config updated with hybrid elasticities")
    print(f"  5. Results saved to {OUTPUT_DIR}/advanced_elasticity_results.json")
    print(f"\n  Next step: retrain SAC with updated parameters (run train_dunnhumby.py)")


if __name__ == "__main__":
    main()
