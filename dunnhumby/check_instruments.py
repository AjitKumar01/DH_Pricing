#!/usr/bin/env python3
"""Check available data for instrumental variables."""
import pandas as pd

coupon = pd.read_csv('data/coupon.csv')
campaign = pd.read_csv('data/campaign_desc.csv')
causal = pd.read_csv('data/causal_data.csv')
coupon_r = pd.read_csv('data/coupon_redempt.csv')

print('=== COUPON ===')
print(coupon.columns.tolist())
print(coupon.head(3))
print(f'Rows: {len(coupon):,}')

print('\n=== CAMPAIGN ===')
print(campaign.columns.tolist())
print(campaign.head(3))

print('\n=== CAUSAL DATA ===')
print(causal.columns.tolist())
print(causal.head(3))
print(f'Rows: {len(causal):,}')
print(f'Unique products: {causal["PRODUCT_ID"].nunique():,}')

print('\n=== COUPON REDEMPT ===')
print(coupon_r.columns.tolist())
print(coupon_r.head(3))
