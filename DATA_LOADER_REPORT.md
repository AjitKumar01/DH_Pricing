# Data Loader Deep Dive — `src/data/loader.py`

## How Raw Data Becomes Model Input

This document traces every transformation from the raw CSV files through the `DunnhumbyLoader` class to the final product-week panel that feeds into the demand models.

---

## 1. Raw Data: Exactly What's in Each File

### `transaction_data.csv` (2,595,732 rows × 12 columns)

Every row is one item scanned at checkout:

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| `household_key` | int | 2375 | Household panel ID |
| `BASKET_ID` | int | 26984851472 | Unique basket (trip) identifier |
| `DAY` | int | 1 | Day number within the study (1–711) |
| `PRODUCT_ID` | int | 1004906 | Product UPC identifier |
| `QUANTITY` | int | 1 | Number of units purchased |
| `SALES_VALUE` | float | 1.39 | **Net** amount paid by the customer (after all discounts) |
| `STORE_ID` | int | 364 | Store where purchase occurred |
| `RETAIL_DISC` | float | -0.60 | Shelf markdown amount (**negative** when discount applied, 0 when no discount) |
| `TRANS_TIME` | int | 1631 | Transaction time (HHMM format) |
| `WEEK_NO` | int | 1 | Week number (1–102) |
| `COUPON_DISC` | float | 0.00 | Manufacturer coupon discount amount (also arrives negative) |
| `COUPON_MATCH_DISC` | float | 0.00 | Retailer match of manufacturer coupon (negative) |

**Critical subtlety**: `SALES_VALUE` is what the customer actually paid. The discount columns tell you how much was discounted from the "full" price. For example, if a $1.99 item is sold for $1.39 with a $0.60 retail discount: `SALES_VALUE=1.39`, `RETAIL_DISC=-0.60`, and the reconstructed base price is `1.39 + 0.60 = 1.99`.

### `product.csv` (92,353 rows × 7 columns)

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| `PRODUCT_ID` | int | 25671 | Links to transactions |
| `MANUFACTURER` | int | 2 | Manufacturer code |
| `DEPARTMENT` | str | GROCERY | Top-level category |
| `BRAND` | str | National | Brand type (National/Private) |
| `COMMODITY_DESC` | str | FRZN ICE | Product category |
| `SUB_COMMODITY_DESC` | str | ICE - CRUSHED/CUBED | Sub-category |
| `CURR_SIZE_OF_PRODUCT` | str | 22 LB | Package size (free text — not parsed) |

### `causal_data.csv` (36,786,524 rows × 5 columns)

Every row is one product in one store in one week:

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| `PRODUCT_ID` | int | 26190 | Links to transactions |
| `STORE_ID` | int | 286 | Store where display/mailer applies |
| `WEEK_NO` | int | 70 | Week of the promotion |
| `display` | str/int | 0 or "A" | 0 = no in-store display, any other value = product is on display |
| `mailer` | str | "A" or other | "A" = product appears in promotional mailer, anything else = no |

**Scale context**: 36.8M rows because it's every product × every store × every week. Most entries are `display=0, mailer=` (no promotion). This file is ~100× larger than the transaction file.

### `coupon.csv` (124,548 rows × 3 columns)

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| `COUPON_UPC` | int | 10000089061 | Coupon barcode identifier |
| `PRODUCT_ID` | int | 27160 | Product eligible for this coupon |
| `CAMPAIGN` | int | 4 | Campaign this coupon belongs to |

### `coupon_redempt.csv` (2,318 rows × 4 columns)

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| `household_key` | int | 1 | Which household redeemed |
| `DAY` | int | 421 | When redeemed |
| `COUPON_UPC` | int | 10000085364 | Which coupon |
| `CAMPAIGN` | int | 8 | Campaign |

### `campaign_desc.csv` (30 rows × 4 columns)

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| `DESCRIPTION` | str | TypeB | Campaign type (A, B, or C) |
| `CAMPAIGN` | int | 24 | Campaign ID |
| `START_DAY` | int | 659 | Start day number |
| `END_DAY` | int | 719 | End day number |

### `campaign_table.csv` (7,208 rows × 3 columns)

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| `DESCRIPTION` | str | TypeA | Campaign type |
| `household_key` | int | 17 | Targeted household |
| `CAMPAIGN` | int | 26 | Campaign ID |

### `hh_demographic.csv` (801 rows × 8 columns)

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| `AGE_DESC` | str | 65+ | Age bracket |
| `MARITAL_STATUS_CODE` | str | A | Marital status |
| `INCOME_DESC` | str | 35-49K | Income bracket |
| `HOMEOWNER_DESC` | str | Homeowner | Homeownership |
| `HH_COMP_DESC` | str | 2 Adults No Kids | Household composition |
| `HOUSEHOLD_SIZE_DESC` | str | 2 | Household size |
| `KID_CATEGORY_DESC` | str | None/Unknown | Child presence |
| `household_key` | int | 1 | Household ID |

---

## 2. The `DunnhumbyLoader` Class — Method by Method

### 2.1 `__init__(self, data_dir: str)`

```python
loader = DunnhumbyLoader("data/")
```

- Stores the `data_dir` path
- Calls `_validate_data_dir()` which checks that all 8 required CSV files exist
- If any file is missing, raises `FileNotFoundError` with the list of missing files
- **Nothing is loaded yet** — data is loaded lazily on demand

### 2.2 `load_transactions() → pd.DataFrame`

This is the first and most important method. Here's exactly what happens:

**Step 1: Read raw CSV**
```python
txn = pd.read_csv("data/transaction_data.csv")
```
Result: 2,595,732 rows with columns as described above.

**Step 2: Ensure integer types**
```python
txn["DAY"] = txn["DAY"].astype(int)
txn["WEEK_NO"] = txn["WEEK_NO"].astype(int)
```
Safety — ensures these are integer type even if CSV parser guesses float.

**Step 3: Make discount columns positive**
```python
for col in ["RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"]:
    txn[col] = txn[col].abs()
```

The raw data stores discounts as negative numbers (e.g., `RETAIL_DISC = -0.60` means a $0.60 discount). This converts them to positive amounts for cleaner arithmetic. After this step, `RETAIL_DISC = 0.60`.

**Step 4: Compute derived price features**

```python
txn["total_discount"] = txn["RETAIL_DISC"] + txn["COUPON_DISC"] + txn["COUPON_MATCH_DISC"]
```
Sum of all three discount channels. For a product with a $0.60 shelf discount and $0.20 coupon: `total_discount = 0.80`.

```python
txn["base_price"] = txn["SALES_VALUE"] + txn["total_discount"]
```
The "full" price before any discounts. For our example: `1.39 + 0.80 = 2.19`. This is the price the customer *would have paid* without any promotional discounts.

```python
txn["avg_unit_price"] = txn["base_price"] / txn["QUANTITY"].clip(lower=1)
```
Per-unit base price. `clip(lower=1)` prevents division by zero if QUANTITY somehow equals 0. For a single item: `2.19 / 1 = 2.19`.

```python
txn["discount_depth"] = txn["total_discount"] / txn["base_price"].clip(lower=0.01)
```
Discount as a fraction of the base price (0 to 1). For our example: `0.80 / 2.19 = 0.365` (36.5% discount). `clip(lower=0.01)` prevents division by zero.

```python
txn["discount_flag"] = (txn["discount_depth"] > 0.001).astype(int)
```
Binary: 1 if there's any meaningful discount (>0.1%), 0 otherwise. The 0.001 threshold filters out floating-point noise.

**Step 5: Filter out bad data**
```python
txn = txn[(txn["QUANTITY"] > 0) & (txn["SALES_VALUE"] > 0) & (txn["base_price"] > 0)].copy()
```
Removes:
- Returns/refunds (negative quantities)
- Zero-sale transactions
- Transactions where base_price ≤ 0 (unreliable data, e.g., when discount exceeds sale value due to data errors)

This removes ~1% of rows (~25,000 transactions). The `copy()` prevents SettingWithCopyWarning.

**Output**: DataFrame with ~2,570,000 rows, original 12 columns plus 5 new columns: `total_discount`, `base_price`, `avg_unit_price`, `discount_depth`, `discount_flag`.

### 2.3 `_build_causal_features(product_ids=None) → pd.DataFrame`

This method transforms the massive 36.8M-row causal file into a manageable product-week summary.

**Step 1: Read and filter**
```python
causal = pd.read_csv("data/causal_data.csv")
if product_ids is not None:
    causal = causal[causal["PRODUCT_ID"].isin(product_ids)]
```
When called from `build_product_weekly_panel()`, `product_ids` is the set of products remaining in the panel (~4,000). This reduces 36.8M rows to a few hundred thousand.

**Step 2: Parse display/mailer flags**
```python
causal["on_display"] = (causal["display"].fillna("0").astype(str) != "0").astype(int)
causal["on_mailer"] = (causal["mailer"].fillna("").astype(str) == "A").astype(int)
```

The `display` column has mixed types:
- `0` (int or str) = no display → `on_display = 0`
- Any other value (1, 2, 3, 4, 5, 6, 7, "A", etc.) = on display → `on_display = 1`
- NaN (missing) = treated as no display → `on_display = 0`

The `mailer` column:
- `"A"` = on mailer → `on_mailer = 1`
- Anything else (NaN, other letters) = not on mailer → `on_mailer = 0`

**Step 3: Aggregate to product-week**
```python
causal_pw = causal.groupby(["PRODUCT_ID", "WEEK_NO"]).agg(
    display_pct=("on_display", "mean"),    # fraction of stores
    mailer_pct=("on_mailer", "mean"),       # fraction of stores
    n_stores_display=("on_display", "sum"), # count of stores
    n_stores_mailer=("on_mailer", "sum"),   # count of stores
)
```

**This is the key aggregation**: The raw data tells us, for each store, whether this product was displayed. By taking the **mean of the binary flag across stores**, we get the **fraction of stores** where the product was on display.

Example: Product 26190 in week 70 is on display in stores 286, 288, 290 but not in stores 292, 294. If 5 stores carry it, `display_pct = 3/5 = 0.60` (60% display coverage).

**Why fractions, not binary**: If we just used a binary "was displayed anywhere," we'd lose the distinction between a product displayed in 1 of 300 stores vs 250 of 300 stores. The fraction preserves the *intensity* of the promotional support.

Also computes binary convenience flags:
```python
causal_pw["has_display"] = (causal_pw["display_pct"] > 0).astype(int)
causal_pw["has_mailer"] = (causal_pw["mailer_pct"] > 0).astype(int)
```

**Output**: DataFrame with columns `[PRODUCT_ID, WEEK_NO, display_pct, mailer_pct, n_stores_display, n_stores_mailer, has_display, has_mailer]`.

### 2.4 `_build_coupon_features() → pd.DataFrame`

```python
coupons = self.load_coupons()  # reads coupon.csv
coupon_products = coupons["PRODUCT_ID"].unique()
return pd.DataFrame({"PRODUCT_ID": coupon_products, "has_coupon": 1})
```

Creates a simple lookup: which products have any associated coupon in the dataset. Returns a 2-column DataFrame. Any product not in this list implicitly has `has_coupon = 0` (applied later via left merge + fillna).

### 2.5 `build_product_weekly_panel(txn, ...) → pd.DataFrame`

**This is the central method**. It takes the raw transaction-level data and produces the **product-week panel** that feeds directly into the demand model.

#### Parameters:
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `txn` | required | Transaction DataFrame from `load_transactions()` |
| `min_weeks` | 30 | Product must appear in ≥30 weeks |
| `min_price_cv` | 0.03 | Product must have price CV ≥ 3% |
| `stable_start` | 18 | First week of stable period |
| `stable_end` | 102 | Last week |
| `include_causal` | True | Whether to merge display/mailer/coupon data |

#### Step-by-step walkthrough:

**Step 1: Restrict to stable period (weeks 18–102)**
```python
stable = txn[(txn["WEEK_NO"] >= 18) & (txn["WEEK_NO"] <= 102)].copy()
```
Weeks 1–17 are excluded because they represent the panel formation period — households are being recruited, so transaction volumes are artificially low and non-representative. The analysis window is 85 weeks.

**Step 2: Aggregate transactions to product-week level**
```python
panel = stable.groupby(["PRODUCT_ID", "WEEK_NO"]).agg(
    quantity=("QUANTITY", "sum"),
    revenue=("SALES_VALUE", "sum"),
    total_base_value=("base_price", "sum"),
    discount_depth=("discount_depth", "mean"),
    discount_flag=("discount_flag", "max"),
    n_transactions=("QUANTITY", "count"),
)
```

For each product-week combination, this computes:

| Output Column | Aggregation | Meaning |
|---------------|-------------|---------|
| `quantity` | SUM of QUANTITY | **Total units sold** across all households and stores that week. This is the **dependent variable** in the demand model (`ln(quantity)` on the left side of the regression). |
| `revenue` | SUM of SALES_VALUE | Total revenue (net of discounts). Used for summaries, not directly in the demand model. |
| `total_base_value` | SUM of base_price | Sum of all pre-discount prices paid × units. Used temporarily to compute `avg_price`. |
| `discount_depth` | MEAN of per-transaction discount_depth | Average discount intensity. **Critical**: this is the model's `d_it` — the regressor capturing promotional depth. If 3 transactions have depths 0.10, 0.20, 0.15, the average is 0.15. |
| `discount_flag` | MAX | 1 if any transaction that week was discounted. Not used in the model (replaced by continuous depth), retained for compatibility. |
| `n_transactions` | COUNT | Number of individual transactions. Data quality indicator — more transactions = more reliable data. |

**Step 3: Compute quantity-weighted average price**
```python
panel["avg_price"] = panel["total_base_value"] / panel["quantity"].clip(lower=1)
```

This is the **quantity-weighted average base (pre-discount) price** for the product that week. It represents the "shelf price" before markdowns.

Why quantity-weighted? If a product is sold at multiple price points (e.g., different pack sizes sold under the same PRODUCT_ID, or mid-week price changes), weighting by quantity gives the effective price the average buyer paid.

This `avg_price` becomes the $P_{it}$ in the demand model: `ln(P_it)` appears as a regressor (the log of this weighted average price).

**Step 4: Remove zero/negative quantity weeks**
```python
panel = panel[panel["quantity"] > 0].copy()
```

**Step 5: Outlier removal — per-product z-score**
```python
log_qty = np.log(panel["quantity"].clip(lower=0.5))
product_stats = log_qty.groupby(panel["PRODUCT_ID"]).agg(["mean", "std"])
# merge stats, compute z-score, and drop outlier rows
z_score = (log_qty_merged - panel["lq_mean"]) / panel["lq_std"].clip(lower=1e-6)
panel = panel[z_score.abs() <= 4].copy()
```
Computes mean and std of log(quantity) for each product, then **drops** product-weeks whose z-score exceeds ±4. These are data anomalies — e.g., a product that normally sells 5 units/week suddenly shows 50,000 units due to a data error or bulk institutional purchase. The `clip(lower=1e-6)` on std prevents division by zero for constant-quantity products.

**Step 6: Outlier removal — global extreme products**
```python
product_med_lq = log_qty.groupby(panel["PRODUCT_ID"]).median()
global_cap = product_med_lq.quantile(0.995)
extreme_products = set(product_med_lq[product_med_lq > global_cap].index)
panel = panel[~panel["PRODUCT_ID"].isin(extreme_products)].copy()
```

Removes the top 0.5% of products by median log-quantity. These are weight/volume products (bulk flour, salt, sugar) where "quantity" is measured in ounces or grams rather than individual items. Their quantities are orders of magnitude larger and would dominate the regression.

**Step 7: Temporal coverage filter**
```python
weeks_per_product = panel.groupby("PRODUCT_ID")["WEEK_NO"].nunique()
eligible_products = weeks_per_product[weeks_per_product >= min_weeks].index
```

A product must have sales in ≥30 of the 85 weeks. Products with fewer weeks have too little data for reliable OLS estimation (the model has 9+ parameters; with fewer than 30 observations, the estimates are extremely noisy).

**Step 8: Price variation filter**
```python
price_cv = panel.groupby("PRODUCT_ID")["avg_price"].agg(
    lambda x: x.std() / x.mean() if x.mean() > 0 else 0
)
price_vary_products = price_cv[price_cv >= min_price_cv].index
```

Coefficient of variation (std/mean) must be ≥ 0.03. If a product always sells at exactly $3.49 with zero price variation, we cannot estimate its price elasticity. We need *some* price movement. The 0.03 threshold is very low — it means the product's price must vary by at least 3% of its mean.

**Step 9: Merge causal data (display/mailer)**

After filtering to eligible products, the causal data is merged:
```python
causal_pw = self._build_causal_features(product_ids=set(panel["PRODUCT_ID"].unique()))
panel = panel.merge(causal_pw[...], on=["PRODUCT_ID", "WEEK_NO"], how="left")
panel["display_pct"] = panel["display_pct"].fillna(0)
panel["mailer_pct"] = panel["mailer_pct"].fillna(0)
```

Left merge: if a product-week has no causal data entry, display and mailer fractions default to 0 (no promotional support).

These `display_pct` and `mailer_pct` columns become the $D_{it}$ and $M_{it}$ regressors in the demand model.

**Step 10: Merge coupon flags**
```python
coupon_df = self._build_coupon_features()
panel = panel.merge(coupon_df, on="PRODUCT_ID", how="left")
panel["has_coupon"] = panel["has_coupon"].fillna(0).astype(int)
```

Binary flag: 1 if the product has any historical coupon association. Not currently used as a regressor in the final model, but available for future analysis.

**Final output**: DataFrame with ~50,000–80,000 product-week rows for ~4,000 eligible products, with columns:

| Column | Type | Model Role |
|--------|------|-----------|
| `PRODUCT_ID` | int | Product identifier (grouping variable for per-product OLS) |
| `WEEK_NO` | int | Time index (used for train/test split, seasonality, AR(1) lag) |
| `quantity` | float | **Dependent variable**: ln(quantity) is the left-hand side |
| `revenue` | float | Summaries only |
| `avg_price` | float | **Regressor**: ln(avg_price) enters as ε_i × ln(P_it) |
| `discount_depth` | float | **Regressor**: γ_i × d_it (continuous discount intensity) |
| `discount_flag` | int | Not used in final model (replaced by continuous depth) |
| `n_transactions` | int | Data quality indicator |
| `display_pct` | float | **Regressor**: δ_1i × display_pct (promotional support) |
| `mailer_pct` | float | **Regressor**: δ_2i × mailer_pct (promotional support) |
| `has_display` | int | Binary convenience flag |
| `has_mailer` | int | Binary convenience flag |
| `has_coupon` | int | Available but not used as regressor |

### 2.6 `compute_product_features(txn) → pd.DataFrame`

This method computes **per-product summary statistics** used for K-Means segmentation. It operates on the raw transaction data (not the panel).

```python
stable = txn[(txn["WEEK_NO"] >= 18) & (txn["WEEK_NO"] <= 102)].copy()
n_weeks_total = stable["WEEK_NO"].nunique()  # 85 weeks

feats = stable.groupby("PRODUCT_ID").agg(
    total_revenue=("SALES_VALUE", "sum"),
    total_qty=("QUANTITY", "sum"),
    n_transactions=("QUANTITY", "count"),
    mean_price=("avg_unit_price", "mean"),
    std_price=("avg_unit_price", "std"),
    pct_on_disc=("discount_flag", "mean"),
    mean_discount_depth=("discount_depth", "mean"),
    n_weeks=("WEEK_NO", "nunique"),
)

feats["weekly_velocity"] = feats["total_qty"] / n_weeks_total
feats["price_cv"] = feats["std_price"] / feats["mean_price"].clip(lower=0.01)
feats["log_price"] = np.log(feats["mean_price"].clip(lower=0.01))
feats["log_velocity"] = np.log(feats["weekly_velocity"].clip(lower=0.01))
```

**Output columns and their use:**

| Feature | How Computed | Used For |
|---------|-------------|----------|
| `total_revenue` | Sum of SALES_VALUE | Revenue-proportional allocation in representative selection |
| `total_qty` | Sum of QUANTITY | Summary |
| `n_transactions` | Count of sales rows | Filter: ≥5 for segmentation eligibility |
| `mean_price` | Mean avg_unit_price | Summary |
| `std_price` | Std avg_unit_price | Summary |
| `pct_on_disc` | Mean discount_flag (0–1) | **Clustering feature**: promotion frequency |
| `mean_discount_depth` | Mean discount_depth | **Clustering feature**: typical markdown depth |
| `n_weeks` | # unique weeks | Data quality filter |
| `weekly_velocity` | total_qty / 85 | Derived |
| `price_cv` | std_price / mean_price | **Clustering feature** + data quality filter |
| `log_price` | ln(mean_price) | **Clustering feature**: log-price tier |
| `log_velocity` | ln(weekly_velocity) | **Clustering feature**: log sales rate |

---

## 3. How This Data Feeds Into the Demand Model

### 3.1 Data Flow Overview

```
transaction_data.csv  →  load_transactions()  →  build_product_weekly_panel()  →  panel
                                                                                    │
                     compute_product_features()  →  segmenter.fit()  →  seg_mapping │
                                                         │                          │
                     selector.select(features, seg_map)  →  rep_ids (150 products)  │
                                                                                    │
                     panel[panel.PRODUCT_ID.isin(rep_ids)]  →  rep_panel            │
                                                                                    │
                     rep_panel[WEEK_NO ≤ 82]  →  train_panel  ─────────────────────→ OLS
                     rep_panel[WEEK_NO > 82]  →  test_panel  ──────────────────────→ Validation
```

### 3.2 What Each Panel Column Becomes in the OLS Regression

The demand model equation:

$$\ln(Q_{it}) = \alpha_i + \epsilon_i \ln(P_{it}) + \gamma_i d_{it} + \delta_{1i} \text{disp}_{it} + \delta_{2i} \text{mail}_{it} + \rho_i \ln(Q_{i,t-1}) + \lambda_i \text{sdo}_{it} + \beta_1 \cos(.) + \beta_2 \sin(.) + u_{it}$$

| Model Term | Panel Column | Transformation | What It Captures |
|------------|-------------|----------------|------------------|
| $\ln(Q_{it})$ (Y) | `quantity` | `np.log(quantity.clip(lower=0.5))` | Demand (clipped to avoid log(0)) |
| $\ln(P_{it})$ | `avg_price` | `np.log(avg_price.clip(lower=0.01))` | Shelf price effect |
| $d_{it}$ | `discount_depth` | Used directly (0–1 scale) | Promotional lift from discounting |
| $\text{disp}_{it}$ | `display_pct` | Used directly (0–1 fraction) | In-store display effect |
| $\text{mail}_{it}$ | `mailer_pct` | Used directly (0–1 fraction) | Mailer ad effect |
| $\ln(Q_{i,t-1})$ | `quantity` | `np.log(quantity.clip(0.5)).shift(1)` after sorting by WEEK_NO | AR(1) demand persistence |
| $\text{sdo}_{it}$ | Built from `discount_depth` + segment mapping | `_build_segment_discount()` | Within-segment substitution/cannibalization |
| $\cos(.)$, $\sin(.)$ | `WEEK_NO` | `cos(2π × w / 52)`, `sin(2π × w / 52)` | Annual seasonality |

### 3.3 How `build_segment_discount` Creates the Substitution Feature

This feature is NOT in the panel output from `loader.py`. It's constructed later by `LogLogDemandModel._build_segment_discount()` using the panel + segment mapping:

```python
# For each segment-week, compute total discount mass and count
seg_week = merged.groupby(["segment", "WEEK_NO"]).agg(
    seg_total_disc=("discount_depth", "sum"),
    seg_count=("discount_depth", "count"),
)
# Mean discount of OTHER products = (total - own) / (count - 1)
merged["seg_disc_other"] = (seg_total_disc - own_discount) / (seg_count - 1)
```

Example: If segment 4 has 31 products, and in week 50, the total discount depth across all 31 products sums to 3.1, then for product X with discount_depth=0.15:
- `seg_disc_other = (3.1 - 0.15) / (31 - 1) = 2.95 / 30 = 0.098`

Product X "sees" an average 9.8% discount on its competitors this week. If λ_X < 0, higher competitor discounts reduce X's demand (cannibalization).

### 3.4 Train/Test Split

```python
train_panel = rep_panel[rep_panel["WEEK_NO"] <= 82].copy()   # 7,473 obs
test_panel  = rep_panel[rep_panel["WEEK_NO"] > 82].copy()    # 2,359 obs
```

Temporal split: weeks 18–82 for training (65 weeks), weeks 83–102 for testing (20 weeks). This is a proper out-of-time split — no information leakage.

### 3.5 What Happens After `loader.py`: The Downstream Pipeline

1. **`loader.py` outputs** → `panel` (product-week observations) + `product_features` (per-product summaries)
2. **`product_seg.py`** → Takes `product_features`, clusters into 12 segments using 5 features
3. **`representative.py`** → Takes `product_features` + `segment_mapping`, selects 150 products via stratified tercile sampling
4. **`run_experiments.py`** → Builds `seg_disc_other` on both train and test panels, fits OLS per product
5. **`shrinkage.py`** → Takes OLS coefficients, shrinks toward segment medians, calibrates intercepts, computes φ
6. **`product_level.py`** → Takes calibrated parameters, builds MDP simulator

### 3.6 Exactly Which Files Are Used vs. Not Used in the Final Pipeline

| File | Used in Pipeline? | How |
|------|-------------------|-----|
| `transaction_data.csv` | **YES** — core input | All demand estimation and feature engineering |
| `product.csv` | **NO** — explored in notebook only | Department/brand info for notebook analysis only |
| `hh_demographic.csv` | **NO** — explored in notebook only | Customer segmentation in notebook only |
| `causal_data.csv` | **YES** — merged into panel | Display/mailer fractions as regressors |
| `coupon.csv` | **YES** — merged into panel | Binary has_coupon flag (available but not a regressor) |
| `coupon_redempt.csv` | **NO** | Not used in any pipeline step |
| `campaign_desc.csv` | **NO** | Not used in any pipeline step |
| `campaign_table.csv` | **NO** | Not used in any pipeline step |

### 3.7 Data Quality Metrics at Each Stage

| Stage | Products | Product-Weeks | Notes |
|-------|----------|--------------|-------|
| Raw transactions | 91,905 | N/A (transaction-level, not product-week) | Raw data |
| After panel aggregation (weeks 18–102) | ~35,000+ | ~500,000+ | Before any quality filters |
| After outlier removal | ~34,000+ | ~490,000+ | Removed extreme-quantity products |
| After temporal coverage (≥30 weeks) | ~4,000 | ~250,000+ | Most products drop out — sparse purchase histories |
| After price variation (CV ≥ 0.03) | ~4,000 | ~250,000+ | Removes fixed-price products |
| After representative selection | **150** | ~10,000 | The working dataset |
| After AR(1) lag (drop first obs/product) | 150 | ~9,800 | First obs per product dropped |
| Train (weeks 18–82) | 150 | **7,473** | Model estimation sample |
| Test (weeks 83–102) | 150 | **2,359** | Out-of-sample validation |
