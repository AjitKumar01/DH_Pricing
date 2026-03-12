#!/usr/bin/env python3
"""
Deep exploration of the Dunnhumby 'The Complete Journey' dataset.
Goal: Understand structure, identify candidate products for simulator,
compute empirical distributions, and plan the funneling approach.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 140)

def main():
    print("=" * 100)
    print(" DUNNHUMBY 'THE COMPLETE JOURNEY' — DEEP DATA EXPLORATION")
    print("=" * 100)

    # ========================================================================
    # 1. TRANSACTIONS
    # ========================================================================
    txn = pd.read_csv("data/transaction_data.csv")
    print(f"\n{'='*80}")
    print(f" 1. TRANSACTIONS")
    print(f"{'='*80}")
    print(f"Shape: {txn.shape}")
    print(f"Columns: {list(txn.columns)}")
    print(f"Date range (DAY): {txn['DAY'].min()} to {txn['DAY'].max()}")
    print(f"Week range: {txn['WEEK_NO'].min()} to {txn['WEEK_NO'].max()}")
    print(f"Unique households: {txn['household_key'].nunique()}")
    print(f"Unique products: {txn['PRODUCT_ID'].nunique()}")
    print(f"Unique stores: {txn['STORE_ID'].nunique()}")
    print(f"Unique baskets: {txn['BASKET_ID'].nunique()}")
    print(f"Total quantity: {txn['QUANTITY'].sum():,.0f}")
    print(f"Total sales value: ${txn['SALES_VALUE'].sum():,.2f}")
    print(f"Total retail disc: ${txn['RETAIL_DISC'].sum():,.2f}")
    print(f"Total coupon disc: ${txn['COUPON_DISC'].sum():,.2f}")

    print(f"\nQuantity distribution:")
    print(txn['QUANTITY'].describe())
    print(f"\nSales value distribution:")
    print(txn['SALES_VALUE'].describe())

    # Discount analysis
    has_disc = txn['RETAIL_DISC'] != 0
    print(f"\n% of transactions with retail discount: {has_disc.mean()*100:.1f}%")
    disc_nonzero = txn.loc[has_disc, 'RETAIL_DISC']
    print(f"Discount distribution (non-zero, note: values are negative = savings):")
    print(disc_nonzero.describe())

    # Compute implied regular price = SALES_VALUE - RETAIL_DISC (discount is negative)
    txn['implied_regular_price'] = txn['SALES_VALUE'] - txn['RETAIL_DISC']
    txn['discount_pct'] = np.where(
        txn['implied_regular_price'] > 0,
        -txn['RETAIL_DISC'] / txn['implied_regular_price'] * 100,
        0
    )
    disc_txn = txn[has_disc]
    print(f"\nDiscount % distribution (when discounted):")
    print(disc_txn['discount_pct'].describe())
    print(f"\nDiscount % buckets:")
    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    print(pd.cut(disc_txn['discount_pct'], bins=bins).value_counts().sort_index())

    # ========================================================================
    # 2. PRODUCTS
    # ========================================================================
    prod = pd.read_csv("data/product.csv")
    print(f"\n{'='*80}")
    print(f" 2. PRODUCTS")
    print(f"{'='*80}")
    print(f"Shape: {prod.shape}")
    print(f"Unique departments: {prod['DEPARTMENT'].nunique()}")
    print(f"\nTop 15 departments:")
    print(prod['DEPARTMENT'].value_counts().head(15))
    print(f"\nUnique commodity descriptions: {prod['COMMODITY_DESC'].nunique()}")
    print(f"\nTop 20 commodities:")
    print(prod['COMMODITY_DESC'].value_counts().head(20))
    print(f"\nUnique sub-commodities: {prod['SUB_COMMODITY_DESC'].nunique()}")

    # ========================================================================
    # 3. HOUSEHOLDS / DEMOGRAPHICS
    # ========================================================================
    hh = pd.read_csv("data/hh_demographic.csv")
    print(f"\n{'='*80}")
    print(f" 3. HOUSEHOLDS & DEMOGRAPHICS")
    print(f"{'='*80}")
    print(f"Shape: {hh.shape}")
    print(f"\nIncome distribution:")
    print(hh['INCOME_DESC'].value_counts().sort_index())
    print(f"\nAge distribution:")
    print(hh['AGE_DESC'].value_counts().sort_index())
    print(f"\nHousehold size:")
    print(hh['HOUSEHOLD_SIZE_DESC'].value_counts().sort_index())
    print(f"\nMarital status:")
    print(hh['MARITAL_STATUS_CODE'].value_counts())
    print(f"\nKid category:")
    print(hh['KID_CATEGORY_DESC'].value_counts())

    # ========================================================================
    # 4. PRODUCT FUNNELING
    # ========================================================================
    print(f"\n{'='*80}")
    print(f" 4. PRODUCT FUNNELING STRATEGY")
    print(f"{'='*80}")

    # Join transactions with product info
    txn_prod = txn.merge(prod, on='PRODUCT_ID', how='left')

    # Step 1: Focus on top departments by transaction volume
    dept_stats = txn_prod.groupby('DEPARTMENT').agg(
        n_txn=('PRODUCT_ID', 'count'),
        n_products=('PRODUCT_ID', 'nunique'),
        total_revenue=('SALES_VALUE', 'sum'),
        total_quantity=('QUANTITY', 'sum'),
        total_discount=('RETAIL_DISC', 'sum'),
        n_households=('household_key', 'nunique'),
    ).sort_values('n_txn', ascending=False)
    dept_stats['avg_price'] = dept_stats['total_revenue'] / dept_stats['total_quantity']
    dept_stats['disc_ratio'] = -dept_stats['total_discount'] / dept_stats['total_revenue']
    print(f"\nDepartment-level summary (top 15):")
    print(dept_stats.head(15).to_string())

    # Step 2: Focus on top commodities within selected departments
    # Select departments with substantial volume and interesting discount patterns
    top_depts = dept_stats.head(8).index.tolist()
    print(f"\nSelected top departments: {top_depts}")

    comm_stats = txn_prod[txn_prod['DEPARTMENT'].isin(top_depts)].groupby(
        ['DEPARTMENT', 'COMMODITY_DESC']
    ).agg(
        n_txn=('PRODUCT_ID', 'count'),
        n_products=('PRODUCT_ID', 'nunique'),
        total_revenue=('SALES_VALUE', 'sum'),
        total_quantity=('QUANTITY', 'sum'),
        total_discount=('RETAIL_DISC', 'sum'),
    ).sort_values('n_txn', ascending=False)
    comm_stats['avg_price'] = comm_stats['total_revenue'] / comm_stats['total_quantity']
    comm_stats['disc_ratio'] = -comm_stats['total_discount'] / comm_stats['total_revenue']
    print(f"\nTop 30 commodities in selected departments:")
    print(comm_stats.head(30).to_string())

    # Step 3: Top individual products by volume
    prod_stats = txn.groupby('PRODUCT_ID').agg(
        n_txn=('BASKET_ID', 'count'),
        total_quantity=('QUANTITY', 'sum'),
        total_revenue=('SALES_VALUE', 'sum'),
        total_discount=('RETAIL_DISC', 'sum'),
        n_weeks=('WEEK_NO', 'nunique'),
        n_households=('household_key', 'nunique'),
        avg_quantity=('QUANTITY', 'mean'),
    ).sort_values('n_txn', ascending=False)
    prod_stats['avg_price'] = prod_stats['total_revenue'] / prod_stats['total_quantity']
    prod_stats['disc_ratio'] = np.where(
        prod_stats['total_revenue'] > 0,
        -prod_stats['total_discount'] / prod_stats['total_revenue'],
        0
    )
    prod_stats['weeks_active'] = prod_stats['n_weeks']

    # Filter: products active in at least 50 weeks, >100 transactions
    reliable_prods = prod_stats[
        (prod_stats['weeks_active'] >= 50) &
        (prod_stats['n_txn'] >= 100)
    ].copy()
    print(f"\nReliable products (active >= 50 weeks, >= 100 txns): {len(reliable_prods)}")

    # Merge with product info
    reliable_prods = reliable_prods.merge(prod, on='PRODUCT_ID', how='left')
    print(f"\nTop 50 reliable products by transaction count:")
    cols = ['PRODUCT_ID', 'DEPARTMENT', 'COMMODITY_DESC', 'n_txn', 'total_quantity',
            'avg_price', 'disc_ratio', 'n_households', 'weeks_active']
    print(reliable_prods[cols].head(50).to_string())

    # ========================================================================
    # 5. WEEKLY DEMAND PATTERNS
    # ========================================================================
    print(f"\n{'='*80}")
    print(f" 5. WEEKLY DEMAND PATTERNS")
    print(f"{'='*80}")

    weekly = txn.groupby('WEEK_NO').agg(
        n_txn=('BASKET_ID', 'count'),
        total_quantity=('QUANTITY', 'sum'),
        total_revenue=('SALES_VALUE', 'sum'),
        total_discount=('RETAIL_DISC', 'sum'),
        n_households=('household_key', 'nunique'),
        n_baskets=('BASKET_ID', 'nunique'),
    )
    print(f"Weekly statistics (summary):")
    print(weekly.describe())
    print(f"\nWeeks with highest transaction counts (potential holidays):")
    print(weekly.nlargest(10, 'n_txn')[['n_txn', 'total_revenue', 'n_baskets']])

    # ========================================================================
    # 6. DAY-OF-WEEK PATTERNS
    # ========================================================================
    print(f"\n{'='*80}")
    print(f" 6. DAY-LEVEL PATTERNS")
    print(f"{'='*80}")

    daily = txn.groupby('DAY').agg(
        n_txn=('BASKET_ID', 'count'),
        total_quantity=('QUANTITY', 'sum'),
        total_revenue=('SALES_VALUE', 'sum'),
    )
    # DAY is sequential (1=first day, 711=last day)
    # Map to day-of-week: need to figure out what day 1 is
    print(f"Total unique days: {daily.shape[0]}")
    print(f"Day range: {daily.index.min()} to {daily.index.max()}")
    print(f"\nDaily transaction count statistics:")
    print(daily['n_txn'].describe())

    # Check day-of-week pattern using modular arithmetic
    # DAY 1 - need to infer the starting day
    daily['dow'] = (daily.index - 1) % 7  # 0=Mon(?) through 6=Sun(?)
    dow_avg = daily.groupby('dow')['n_txn'].mean()
    print(f"\nAverage transactions by day-of-week offset (0-6):")
    print(dow_avg)
    print(f"\nDay-of-week normalized multipliers:")
    print((dow_avg / dow_avg.mean()).round(3))

    # ========================================================================
    # 7. CROSS-PURCHASE ANALYSIS (for cross-elasticity)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f" 7. CROSS-PURCHASE ANALYSIS (basket co-occurrence)")
    print(f"{'='*80}")

    # Focus on top 20 products for co-occurrence analysis
    top20_ids = reliable_prods['PRODUCT_ID'].head(20).tolist()
    basket_items = txn[txn['PRODUCT_ID'].isin(top20_ids)][['BASKET_ID', 'PRODUCT_ID']].drop_duplicates()

    # Compute co-occurrence matrix
    from itertools import combinations
    co_count = {}
    for bid, grp in basket_items.groupby('BASKET_ID'):
        pids = grp['PRODUCT_ID'].tolist()
        if len(pids) > 1:
            for p1, p2 in combinations(sorted(pids), 2):
                co_count[(p1, p2)] = co_count.get((p1, p2), 0) + 1

    if co_count:
        co_df = pd.DataFrame([
            {'prod1': k[0], 'prod2': k[1], 'co_purchases': v}
            for k, v in co_count.items()
        ]).sort_values('co_purchases', ascending=False)
        print(f"Top 20 co-purchased product pairs (from top 20 products):")
        print(co_df.head(20).to_string())
    else:
        print("No co-purchases found among top 20 products")

    # ========================================================================
    # 8. DISCOUNT ELASTICITY ESTIMATION
    # ========================================================================
    print(f"\n{'='*80}")
    print(f" 8. EMPIRICAL PRICE/DISCOUNT ELASTICITY")
    print(f"{'='*80}")

    # For top 10 products, compare weekly sales with/without discounts
    top10_ids = reliable_prods['PRODUCT_ID'].head(10).tolist()

    for pid in top10_ids:
        ptxn = txn[txn['PRODUCT_ID'] == pid].copy()
        pinfo = prod[prod['PRODUCT_ID'] == pid].iloc[0]

        # Group by week
        weekly_p = ptxn.groupby('WEEK_NO').agg(
            quantity=('QUANTITY', 'sum'),
            revenue=('SALES_VALUE', 'sum'),
            discount=('RETAIL_DISC', 'sum'),
        )
        weekly_p['avg_price'] = weekly_p['revenue'] / weekly_p['quantity']
        weekly_p['has_discount'] = weekly_p['discount'] < -0.01

        disc_weeks = weekly_p[weekly_p['has_discount']]
        no_disc_weeks = weekly_p[~weekly_p['has_discount']]

        if len(disc_weeks) > 0 and len(no_disc_weeks) > 0:
            avg_q_disc = disc_weeks['quantity'].mean()
            avg_q_no = no_disc_weeks['quantity'].mean()
            avg_p_disc = disc_weeks['avg_price'].mean()
            avg_p_no = no_disc_weeks['avg_price'].mean()
            lift = (avg_q_disc - avg_q_no) / avg_q_no * 100 if avg_q_no > 0 else 0

            # Point elasticity
            if avg_p_no > 0 and avg_q_no > 0 and abs(avg_p_disc - avg_p_no) > 0.01:
                elasticity = ((avg_q_disc - avg_q_no)/avg_q_no) / ((avg_p_disc - avg_p_no)/avg_p_no)
            else:
                elasticity = None

            print(f"\n  Product {pid} ({pinfo['COMMODITY_DESC']}):")
            print(f"    Non-discount weeks: {len(no_disc_weeks)}, avg qty: {avg_q_no:.1f}, avg price: ${avg_p_no:.2f}")
            print(f"    Discount weeks:     {len(disc_weeks)}, avg qty: {avg_q_disc:.1f}, avg price: ${avg_p_disc:.2f}")
            print(f"    Volume lift: {lift:+.1f}%")
            if elasticity is not None:
                print(f"    Implied elasticity: {elasticity:.2f}")

    # ========================================================================
    # 9. CUSTOMER SEGMENT ANALYSIS
    # ========================================================================
    print(f"\n{'='*80}")
    print(f" 9. CUSTOMER SEGMENT ANALYSIS")
    print(f"{'='*80}")

    txn_hh = txn.merge(hh, on='household_key', how='inner')
    print(f"Transactions with demographics: {len(txn_hh):,}")

    # Spending by income segment
    income_stats = txn_hh.groupby('INCOME_DESC').agg(
        n_txn=('BASKET_ID', 'count'),
        avg_spend=('SALES_VALUE', 'mean'),
        total_spend=('SALES_VALUE', 'sum'),
        disc_ratio=('RETAIL_DISC', lambda x: -x.sum() / txn_hh.loc[x.index, 'SALES_VALUE'].sum()),
    ).sort_values('total_spend', ascending=False)
    print(f"\nSpending by income level:")
    print(income_stats.to_string())

    # Discount sensitivity by income
    print(f"\nDiscount sensitivity by income:")
    for income, grp in txn_hh.groupby('INCOME_DESC'):
        disc_pct = (grp['RETAIL_DISC'] != 0).mean() * 100
        avg_disc = -grp[grp['RETAIL_DISC'] != 0]['RETAIL_DISC'].mean() if (grp['RETAIL_DISC'] != 0).any() else 0
        print(f"  {income:>20s}: {disc_pct:.1f}% of purchases on discount, avg savings ${avg_disc:.2f}")

    print(f"\n{'='*80}")
    print(f" EXPLORATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
