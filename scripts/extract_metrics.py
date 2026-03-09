import pandas as pd
import numpy as np

df = pd.read_csv('results/product_level_params.csv')
print("=== Parameter Summaries ===")
print(f"phi: {df['phi'].iloc[0]:.4f}")
print(f"mean_elast_shrunk: {df['elasticity_shrunk'].mean():.3f}")
print(f"median_elast_shrunk: {df['elasticity_shrunk'].median():.3f}")
print(f"neg_elast: {(df['elasticity_shrunk'] < 0).sum()}/{len(df)}")
print(f"mean_disc_shrunk: {df['disc_effect_shrunk'].mean():.3f}")
print(f"median_disc_shrunk: {df['disc_effect_shrunk'].median():.3f}")

if 'demand_persistence_shrunk' in df.columns:
    p = df['demand_persistence_shrunk']
    print(f"mean_persistence_shrunk: {p.mean():.3f}")
    print(f"median_persistence_shrunk: {p.median():.3f}")
    print(f"nonzero_persistence: {(p > 0).sum()}/{len(df)}")
    print(f"p5_persistence: {p.quantile(0.05):.3f}")
    print(f"p95_persistence: {p.quantile(0.95):.3f}")

if 'substitution_effect_shrunk' in df.columns:
    s = df['substitution_effect_shrunk']
    print(f"mean_substitution_shrunk: {s.mean():.3f}")
    print(f"median_substitution_shrunk: {s.median():.3f}")
    print(f"nonzero_substitution: {(s < 0).sum()}/{len(df)}")
    print(f"p5_substitution: {s.quantile(0.05):.3f}")
    print(f"p95_substitution: {s.quantile(0.95):.3f}")

if 'display_effect_shrunk' in df.columns:
    d = df['display_effect_shrunk']
    print(f"mean_display_shrunk: {d.mean():.3f}")
    print(f"nonzero_display: {(d != 0).sum()}/{len(df)}")

if 'mailer_effect_shrunk' in df.columns:
    m = df['mailer_effect_shrunk']
    print(f"mean_mailer_shrunk: {m.mean():.3f}")
    print(f"nonzero_mailer: {(m != 0).sum()}/{len(df)}")

# Raw (unshrunk) stats
print("\n=== Raw (unshrunk) stats ===")
if 'demand_persistence' in df.columns:
    print(f"raw_mean_persistence: {df['demand_persistence'].mean():.3f}")
    n_sig = (df['demand_persistence_pval'] < 0.05).sum() if 'demand_persistence_pval' in df.columns else 'N/A'
    print(f"sig_persistence (p<0.05): {n_sig}")
if 'substitution_effect' in df.columns:
    print(f"raw_mean_substitution: {df['substitution_effect'].mean():.3f}")
    n_sig = (df['substitution_effect_pval'] < 0.05).sum() if 'substitution_effect_pval' in df.columns else 'N/A'
    print(f"sig_substitution (p<0.05): {n_sig}")
if 'display_effect' in df.columns:
    n_sig = (df['display_pval'] < 0.05).sum() if 'display_pval' in df.columns else 'N/A'
    print(f"sig_display (p<0.05): {n_sig}")
    print(f"raw_mean_display: {df['display_effect'].mean():.3f}")
if 'mailer_effect' in df.columns:
    n_sig = (df['mailer_pval'] < 0.05).sum() if 'mailer_pval' in df.columns else 'N/A'
    print(f"sig_mailer (p<0.05): {n_sig}")
    print(f"raw_mean_mailer: {df['mailer_effect'].mean():.3f}")

print(f"\nTrain obs: {df['n_obs'].mean():.0f} mean, {df['n_obs'].median():.0f} median")
print(f"mean_r2: {df['r2'].mean():.3f}")
print(f"median_r2: {df['r2'].median():.3f}")

# Segment summary
print("\n=== Segment Elasticity Summary ===")
seg = pd.read_csv('results/segment_elasticities.csv')
print(seg.to_string(index=False))

# Sensitivity by segment
print("\n=== Sensitivity by Segment ===")
try:
    ss = pd.read_csv('results/sensitivity_by_segment.csv')
    print(ss.to_string(index=False))
except:
    print("Not available")
