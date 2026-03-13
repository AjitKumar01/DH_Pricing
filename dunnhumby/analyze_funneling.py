#!/usr/bin/env python3
"""Analyze the product funneling pipeline in detail."""

import pandas as pd
import numpy as np

txn = pd.read_csv('data/transaction_data.csv')
prod = pd.read_csv('data/product.csv')

txn_prod = txn.merge(prod[['PRODUCT_ID','DEPARTMENT','COMMODITY_DESC']], on='PRODUCT_ID', how='left')

# Product-level stats
prod_stats = txn_prod.groupby('PRODUCT_ID').agg(
    n_txn=('BASKET_ID','count'),
    n_weeks=('WEEK_NO','nunique'),
    n_hh=('household_key','nunique'),
    total_rev=('SALES_VALUE','sum'),
    total_qty=('QUANTITY','sum'),
    total_disc=('RETAIL_DISC','sum'),
).reset_index()
prod_stats = prod_stats.merge(prod[['PRODUCT_ID','DEPARTMENT','COMMODITY_DESC']], on='PRODUCT_ID', how='left')
prod_stats['avg_price'] = prod_stats['total_rev'] / prod_stats['total_qty'].clip(lower=1)
prod_stats['disc_ratio'] = -prod_stats['total_disc'] / prod_stats['total_rev'].clip(lower=0.01)

print(f'Total products: {len(prod_stats):,}')

# Filter 1: Activity
f1 = prod_stats[
    (prod_stats['n_txn'] >= 200) &
    (prod_stats['n_weeks'] >= 60) &
    (prod_stats['avg_price'] > 0.10) &
    (prod_stats['avg_price'] < 50) &
    (prod_stats['DEPARTMENT'].notna()) &
    (~prod_stats['DEPARTMENT'].isin(['KIOSK-GAS', 'MISC SALES TRAN', 'MISC. TRANS.']))
].copy()
print(f'After activity filter: {len(f1):,}')

# Filter 2: Discount variation
weekly_disc = txn_prod.groupby(['PRODUCT_ID','WEEK_NO']).agg(
    revenue=('SALES_VALUE','sum'), discount=('RETAIL_DISC','sum')
).reset_index()
weekly_disc['disc_pct'] = np.where(weekly_disc['revenue']>0, -weekly_disc['discount']/weekly_disc['revenue'], 0)
disc_var = weekly_disc.groupby('PRODUCT_ID')['disc_pct'].std().reset_index()
disc_var.columns = ['PRODUCT_ID','disc_variation']

f1 = f1.merge(disc_var, on='PRODUCT_ID', how='left')
f1['disc_variation'] = f1['disc_variation'].fillna(0)
f2 = f1[f1['disc_variation'] > 0.02].copy()
print(f'After discount variation filter: {len(f2):,}')

# Show departments
print(f'\nDepartment distribution after filtering:')
for dept, count in f2['DEPARTMENT'].value_counts().head(15).items():
    print(f'  {dept}: {count}')

# Show commodity distribution
print(f'\nCommodity categories after filtering (top 30):')
for comm, count in f2['COMMODITY_DESC'].value_counts().head(30).items():
    print(f'  {comm}: {count}')

# Scoring
for col in ['n_txn', 'n_hh', 'disc_variation', 'n_weeks']:
    vmin, vmax = f2[col].min(), f2[col].max()
    if vmax > vmin:
        f2[f'norm_{col}'] = (f2[col] - vmin) / (vmax - vmin)
    else:
        f2[f'norm_{col}'] = 0.5

f2['score'] = (0.30 * f2['norm_n_txn'] +
               0.25 * f2['norm_n_hh'] +
               0.25 * f2['norm_disc_variation'] +
               0.20 * f2['norm_n_weeks'])

print(f'\nScore distribution:')
print(f'  Mean: {f2["score"].mean():.3f}')
print(f'  Std:  {f2["score"].std():.3f}')
print(f'  Min:  {f2["score"].min():.3f}')
print(f'  Max:  {f2["score"].max():.3f}')

# Show how many unique commodities we have
n_commodities = f2['COMMODITY_DESC'].nunique()
print(f'\nTotal unique commodities: {n_commodities}')
print(f'If we pick 3 per commodity from 15 categories = 45 products')
print(f'If we pick 2 per commodity from 22 categories = 44 products')
