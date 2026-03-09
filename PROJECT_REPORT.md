# Product-Level Markdown Pricing Simulator — Comprehensive Project Report

## Table of Contents

1. [Raw Data: What Was Present](#1-raw-data-what-was-present)
2. [Data Extraction & Loading](#2-data-extraction--loading)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Figures: What, Why, and How](#4-figures-what-why-and-how)
5. [Feature Engineering](#5-feature-engineering)
6. [Product Segmentation (Clustering)](#6-product-segmentation-clustering)
7. [Representative Product Selection](#7-representative-product-selection)
8. [Demand Model Estimation](#8-demand-model-estimation)
9. [Empirical Bayes Shrinkage & Calibration](#9-empirical-bayes-shrinkage--calibration)
10. [Simulator Construction](#10-simulator-construction)
11. [Policy Evaluation & Sensitivity Analysis](#11-policy-evaluation--sensitivity-analysis)
12. [Validation](#12-validation)
13. [Key Results](#13-key-results)
14. [Project Structure](#14-project-structure)

---

## 1. Raw Data: What Was Present

The project uses the **Dunnhumby "The Complete Journey"** dataset — a publicly available, real-world retail transaction dataset covering **2 years** (102 weeks) of grocery purchases from approximately **2,500 households** across multiple stores.

### 1.1 Data Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `transaction_data.csv` | 2,595,732 | 12 | Every household−basket−product purchase: household_key, BASKET_ID, DAY, PRODUCT_ID, QUANTITY, SALES_VALUE, STORE_ID, RETAIL_DISC, TRANS_TIME, WEEK_NO, COUPON_DISC, COUPON_MATCH_DISC |
| `product.csv` | 92,353 | 7 | Product master: PRODUCT_ID, MANUFACTURER, DEPARTMENT, BRAND, COMMODITY_DESC, SUB_COMMODITY_DESC, CURR_SIZE_OF_PRODUCT |
| `hh_demographic.csv` | 801 | 8 | Panelist demographics: AGE_DESC, MARITAL_STATUS_CODE, INCOME_DESC, HOMEOWNER_DESC, HH_COMP_DESC, HOUSEHOLD_SIZE_DESC, KID_CATEGORY_DESC, household_key |
| `causal_data.csv` | 36,786,524 | 5 | Store-product-week promotional flags: PRODUCT_ID, STORE_ID, WEEK_NO, display (0/A/other codes), mailer (A = on mailer) |
| `coupon.csv` | 124,548 | 3 | Coupon−product−campaign associations: COUPON_UPC, PRODUCT_ID, CAMPAIGN |
| `coupon_redempt.csv` | 2,318 | 4 | Actual coupon redemptions: household_key, DAY, COUPON_UPC, CAMPAIGN |
| `campaign_desc.csv` | 30 | 4 | Campaign metadata: DESCRIPTION (TypeA/B/C), CAMPAIGN, START_DAY, END_DAY |
| `campaign_table.csv` | 7,208 | 3 | Household−campaign targeting: DESCRIPTION, household_key, CAMPAIGN |

### 1.2 Key Characteristics of the Raw Data

- **Price reconstruction challenge**: The raw `SALES_VALUE` is the *net* amount paid. To get the *base* (pre-discount) price, we must reverse-engineer: `base_price = SALES_VALUE + |RETAIL_DISC| + |COUPON_DISC| + |COUPON_MATCH_DISC|`. The discount columns come as negative values representing the discount amount.
- **Three discount channels**: Retail shelf markdowns (`RETAIL_DISC`), manufacturer coupons (`COUPON_DISC`), and retailer coupon-match programs (`COUPON_MATCH_DISC`). All three are combined into a unified `total_discount` and `discount_depth` (as a fraction of base price).
- **Promotional support data**: The causal_data file (36.8M rows) records, for every product−store−week combination, whether the product was on in-store display (`display` field, 0 = no, anything else = yes) and whether it appeared in a promotional mailer (`mailer` field, `A` = yes). This is at store granularity — a product can be on display in some stores but not others in the same week.
- **Scale**: 91,905 unique products, but the distribution is extremely long-tailed — the top 10% of products by revenue account for the vast majority of sales. The bottom half of products have very sparse purchase histories, making individual demand estimation unreliable for most SKUs.
- **Temporal structure**: 102 weeks, with week numbers 1−102. Weeks 1−17 show ramp-up effects (panel formation), so the stable analysis period is weeks 18−102 (85 weeks).
- **Demographics**: Only 801 of the ~2,500 households have demographic data. Demographics include income brackets (Under 15K to 200K+), age brackets (19−24 through 65+), household composition, and homeownership status.


## 2. Data Extraction & Loading

### 2.1 Data Loader (`src/data/loader.py`)

The `DunnhumbyLoader` class handles all data loading and preprocessing:

**What it does:**

1. **Validates data directory** — checks all 8 required CSV files exist before proceeding.

2. **`load_transactions()`** — Reads `transaction_data.csv` and:
   - Makes discount columns positive (they arrive as negative values)
   - Reconstructs `total_discount = |RETAIL_DISC| + |COUPON_DISC| + |COUPON_MATCH_DISC|`
   - Computes `base_price = SALES_VALUE + total_discount`
   - Computes `avg_unit_price = base_price / QUANTITY`
   - Computes `discount_depth = total_discount / base_price` (0 to 1 scale)
   - Creates a binary `discount_flag` (1 if discount_depth > 0.1%)
   - Filters out returns/refunds (negative quantities or sales) and zero-base-price records (~1% of data)

3. **`_build_causal_features()`** — Processes causal_data.csv:
   - Converts display codes: `display = 0` → not displayed, any other value → displayed
   - Converts mailer codes: `mailer = 'A'` → on mailer, anything else → not
   - **Aggregates from store-product-week to product-week**: computes `display_pct` (fraction of stores where product was displayed) and `mailer_pct` (fraction of stores with mailer exposure). This is important — the raw data is at the store level, but our demand model operates at the product-week level (aggregated across stores), so we need the fraction, not a binary flag.

4. **`_build_coupon_features()`** — Identifies which products have any coupon association.

5. **`build_product_weekly_panel()`** — The core panel construction:
   - Restricts to the stable period (weeks 18−102)
   - Aggregates transactions to product-week level: total `quantity`, `revenue`, weighted `discount_depth`, max `discount_flag`, transaction count
   - Computes quantity-weighted `avg_price = total_base_value / quantity`
   - **Outlier removal**: Products whose log-quantity exceeds 4 standard deviations from their own mean are flagged. Products whose median log-quantity exceeds the 99.5th percentile globally are removed entirely (catches weight/volume products like bulk items with millions of "units").
   - **Temporal coverage filter**: Requires each product to appear in ≥30 of the 85 stable weeks
   - **Price variation filter**: Requires coefficient of variation in price ≥ 0.03 (need *some* price variation to estimate elasticity)
   - Merges in causal features (display_pct, mailer_pct) and coupon flags
   - Result: ~4,000 eligible products with adequate data quality for demand estimation

6. **`compute_product_features()`** — Per-product summary statistics for segmentation:
   - Total revenue, total quantity, transaction count
   - Mean price, price standard deviation, price coefficient of variation
   - Promotion frequency (`pct_on_disc`), mean discount depth
   - Weekly velocity (quantity per week)
   - Log transformations of price and velocity (for better-behaved distributions in clustering)

### 2.2 What Data Was Extracted vs. Ignored

**Used:**
- All transaction fields (prices, quantities, discounts)
- Product hierarchy (DEPARTMENT, COMMODITY, for initial notebook exploration)
- All three discount channels (combined into a single `discount_depth`)
- Causal display/mailer data (aggregated to product-week fractions)
- Coupon product associations (as a binary flag)

**Not used in the final pipeline** (explored in notebook only):
- Household demographics (used for customer segmentation analysis in the notebook, but the final product-level simulator does not condition on household-level features)
- Campaign targeting data (campaign_table, campaign_desc — these describe which households were targeted by which campaigns, which would be relevant for a household-level choice model but not the aggregate product-level demand model)
- Coupon redemption details (only coupon-product *existence* is used as a flag)
- Store-level variation (aggregated away when building the product-week panel)
- Basket-level associations (the simulator models each product independently with within-segment substitution; basket-level complementarities are not modeled)


## 3. Exploratory Data Analysis (EDA)

Two notebooks contain EDA:

### 3.1 `01_data_exploration.ipynb` — Brazilian E-Commerce (Olist)

This notebook was an **early exploration** on a different dataset (Brazilian Olist e-commerce data). It covers:
- Loading 9 Olist CSV files and building a master merged dataframe
- Orders/revenue time series with moving averages
- Payment type distribution and Pareto analysis of category revenue
- Product price distributions across top 12 categories
- Cross-seller price variation as a "natural experiment" for elasticity
- Temporal patterns (day-of-week, hour-of-day, seasonality via STL decomposition)
- Customer RFM (Recency-Frequency-Monetary) analysis
- Product segmentation via K-Means (K=15)
- Segment-level elasticity estimation with economic priors

This notebook was abandoned in favor of the much richer Dunnhumby dataset. It remains in the project for reference but does not feed into the final pipeline.

### 3.2 `dunnhumby_exploration.ipynb` — The Main EDA Notebook

This is the **primary analysis notebook** with 47 code cells spanning 19 sections. It progresses from raw data exploration through a fully validated product-level simulator.

#### Section 1–2: Data Loading and Initial Statistics
- Loads all 7 Dunnhumby CSV files
- Reports shapes: 2.6M transactions, 92K products, 801 demographics, 36.8M causal records
- Computes base_price, unit_price, discount_depth from raw transaction fields
- Unique counts: ~91,905 products, ~2,500 households, ~364 stores, ~44 departments
- Discount statistics: ~26% of transactions have some discount, mean depth ~5.4%

#### Section 3: Transaction Volume and Revenue Over Time
- Aggregates weekly baskets, units, revenue, discount intensity
- Identifies seasonal patterns (holiday spikes, promotional peaks)
- Calculates discount cost as fraction of revenue

#### Section 4: Price Distribution and Discount Patterns  
- Per-product statistics: mean price, discount depth, transaction count, price CV
- Pareto analysis: top 10% of products capture ~75% of revenue (classic long-tail)
- Department-level discount intensity comparison

#### Section 5: Demand Elasticity Analysis
- Builds a product-week panel for weeks 18−102
- Filters to products with ≥30 weeks and price CV > 0.05
- Runs per-product log-log OLS: `ln(Q) ~ ln(P) + has_discount`
- **Key finding**: Most raw elasticities cluster near zero (~57% are near-zero or positive), indicating massive omitted variable bias from not controlling for display, mailer, and seasonality. This motivated the full multivariate model in the `src/` codebase.
- Plots elasticity distribution: all products, significant-only (p<0.05), and elasticity vs R²

#### Section 6: Customer Purchase Behavior
- Per-household features: total spend, basket count, visit frequency, discount affinity, category diversity
- PCA + K-Means customer segmentation (optimal K=3−4)
- Customer segments visualized in PCA space with z-scored feature heatmap
- Spend distribution by income bracket
- **Conclusion**: Customer segments exist but the product-level simulator abstracts away household heterogeneity.

#### Section 7: Correlation Analysis
- Correlation matrix of price, discount, demand, revenue variables
- VIF analysis for multicollinearity checking
- Key correlations: discount_depth ↔ quantity (positive, as expected), log_price ↔ log_quantity (weak negative, suggesting endogeneity issues)

#### Section 8: Product Segmentation
- Clustering features: log_price, log_velocity, pct_on_disc, mean_discount_depth, price_cv
- Evaluates K=5−25 via silhouette score and inertia (elbow method)
- Selects K=12 (best balance of interpretability and separation)
- Segment profiles visualized as z-scored heatmap + revenue share pie chart
- Saves `product_segments.csv`
- **Key insight**: Elasticity was initially included as a clustering feature but removed because the bivariate elasticity is too noisy (dominated by endogeneity). The final clustering uses only price/velocity/discount behavior variables.

#### Section 9: Segment-Level Elasticity Models
- Pools all product-weeks within each segment
- Fits per-segment log-log OLS: `ln(Q) ~ ln(P) + discount_flag + week_FE`
- Train/test split: weeks 18−82 / 83−102
- Reports segment-level elasticities with 95% CIs
- Computes out-of-sample MAPE per segment

#### Section 10: Multinomial Logit (MNL) Choice Model
- Constructs choice data: each basket is a "choice occasion"
- Chosen segment = segment with highest household spend in that basket
- Estimates MNL via maximum likelihood (L-BFGS-B optimization)
- Parameters: β_price (price sensitivity), β_discount (discount response), per-segment ASCs
- McFadden pseudo-R²
- **Purpose**: MNL provides cross-segment substitution elasticities for the simulator

#### Section 11: Nested Logit Extension (Attempted)
- Maps segments to dominant departments as nests
- Implements nested logit log-likelihood with dissimilarity parameters
- **Outcome**: Computational complexity too high for the data size; MNL is retained as the working model with the note that segment-level aggregation partially mitigates IIA concerns

#### Sections 12–15: Initial Simulator Construction
- Segment-level MDP simulator combining log-log demand with MNL substitution
- State: selection × discount × demand × budget × time
- Budget constraint enforcement (scales discounts if overspent)
- Instantiated with calibrated segment parameters

#### Section 16: Simulator Rollout (4 Policies)
- No discount, uniform 10%, greedy elastic 20%, declining 30%→5%
- 50 Monte Carlo rollouts each, T=12 periods
- Plots weekly revenue, cumulative revenue, budget, demand trajectories

#### Sections 17–20: Aggregate Validation, Elasticity Recovery, Policy Comparison, Sensitivity
- Segment-level validation: out-of-sample MAPE, correlation, KS tests
- Elasticity recovery: verifies simulator recovers input elasticities under controlled experiments
- 5-policy Monte Carlo comparison
- Per-segment discount sensitivity sweeps (0−50%)

#### Sections 21–27: Product-Level Pipeline (the foundation for the `src/` codebase)
- Representative product selection (150 products, revenue-proportional)
- Per-product log-log OLS demand estimation
- Empirical Bayes shrinkage toward segment medians
- Product-level simulator construction (6 iterative versions V1−V3 + calibrated + final)
- Extensive validation: direct prediction, simulator replay, matched comparison, Duan smearing correction
- Elasticity recovery at product level
- 7-policy Monte Carlo comparison
- Per-product sensitivity analysis (discount sweep 0−45%)


## 4. Figures: What, Why, and How

### 4.1 EDA Figures (Generated in Notebook)

| Figure | File | What | Why | How |
|--------|------|------|-----|-----|
| 01 | `01_time_series_overview.png` | 6-panel weekly time series: baskets, actual vs full-price revenue, units, discount %/depth, discount cost, active products/households | Understand temporal structure — seasonality, trends, promotional rhythm | Aggregate transactions by WEEK_NO, compute weekly sums/means, plot with matplotlib |
| 02 | `02_price_discount_patterns.png` | 6-panel distributions: base price histogram, discount depth, % on discount, Pareto rank, price-vs-discount scatter, department discount bars | Understand the price/discount landscape — is there enough variation? How concentrated? | Per-product aggregation, histograms, scatter, Pareto cumulative sum |
| 03 | `03_elasticity_distribution.png` | 3-panel: all elasticities, significant-only (p<0.05), elasticity vs R² scatter | **Critical**: Shows that bivariate elasticities are mostly near zero — motivates the multivariate model with promotional controls | Per-product OLS `ln(Q) ~ ln(P) + disc_flag`, collect estimated ε, histogram/scatter |
| 04 | `04_customer_behavior.png` | 6-panel customer behavior: spend, visits, discount %, basket size, dept diversity, spend-by-income | Understand household heterogeneity — do customer segments matter? | Per-household aggregation from transactions, merge demographics, histograms |
| 05 | `05_customer_segments.png` | PCA-projected customer clusters (K-Means) + z-scored feature heatmap | Visualize natural customer groupings | PCA on standardized customer features, K-Means with silhouette selection, scatter+heatmap |
| 06 | `06_correlation_analysis.png` | Correlation heatmap + discount-vs-demand scatter with fitted line | Check multicollinearity, verify discount-demand relationship direction | Pearson correlation matrix, seaborn heatmap, scatter with np.polyfit |
| 07 | `07_kmeans_evaluation.png` | Silhouette score and inertia (elbow) curves for K=5−25 | Choose optimal number of product clusters | K-Means with varying K, compute silhouette_score per K |
| 08 | `08_product_segments.png` | Segment profile heatmap (z-scored means) + revenue share pie chart | Characterize the 12 product segments — which are premium vs economy, high vs low velocity | Z-score segment means on clustering features, matplotlib heatmap + pie |
| 09 | `09_segment_elasticities.png` | Per-segment elasticity bars with 95% CI, discount effects, R²/MAPE | Show cross-segment demand responsiveness heterogeneity | Pooled per-segment OLS, extract coefficients ± 1.96×SE, bar charts |

### 4.2 Model & Simulation Figures (Generated in Notebook)

| Figure | File | What | Why | How |
|--------|------|------|-----|-----|
| 10 | `10_simulator_rollout.png` | 4-panel policy rollout: weekly revenue, cumulative revenue, budget remaining, total demand | Compare initial segment-level policies | 50 MC rollouts per policy, plot mean±std trajectories |
| 11 | `11_demand_validation.png` | 6-panel validation: scatter, weekly aggregate, per-segment MAPE, Q-Q, residuals, segment comparison | Validate segment-level simulator against test data | Replay test-period data through simulator, compute error metrics |
| 12 | `12_aggregate_validation.png` | 6-panel: aggregate simulator scatter, time series, MAPE bars, Q-Q, residuals, top-segment | Validate revised aggregate simulator | Same as above with aggregate model |
| 13 | `13_elasticity_recovery.png` | Elasticity recovery + cross-elasticity matrix | Verify the simulator faithfully recovers inputted elasticities | Sweep discount 0−40% per segment, regress to recover implied elasticity |
| 14 | `14_policy_rollout.png` | 5-policy comparison on aggregate simulator: cumulative revenue, budget, demand, box plots | Compare policies at aggregate level | 100 MC rollouts per policy |
| 15 | `15_sensitivity_analysis.png` | Per-segment discount sensitivity curves + ROI | Identify which segments benefit from discounting and optimal depth | Sweep discount 0−50% per segment, 100 runs each |
| 16 | `16_product_validation.png` | 6-panel product-level validation V1 | Initial validation of per-product simulator | Direct prediction + replay on test weeks |
| 17 | `17_product_validation_revised.png` | 6-panel product-level validation V1b (with fixes) | Diagnose and fix high-MAPE outliers | Purchase probability adjustment, trimmed MAPE |
| 18a | `18_product_validation_v2.png` | Product-level validation V2 (constrained elasticity) | Constrain elasticities ≤ 0, re-shrink | Constrained OLS, V2 simulator with purchase probability |
| 18b | `18_product_validation_final.png` | 6-panel final validation V3 (matched comparison) | Fair matched comparison: only count demand for products with actual sales | Match simulated to actual by product-week presence |
| 19 | `19_final_validation.png` | 6-panel bias-corrected validation with Duan smearing (φ correction) | Calibrated simulator with heteroscedasticity-consistent retransformation | Compute φ = Σobserved / Σpredicted, apply globally |
| 20 | `20_production_validation.png` | Production-ready simulator validation | Final full validation with purchase probability + φ correction | Full 150-product replay |
| 21 | `21_elasticity_recovery_product.png` | Product-level elasticity recovery (one product per segment) | Verify simulator's demand function recovers expected elasticity patterns | Sweep discount 0−45% per representative product, compute arc elasticity |
| 22 | `22_policy_rollout_product.png` | 7-policy comparison: revenue bars, revenue-vs-cost scatter, budget traces | Compare product-level policies (the final comparison) | 30 MC rollouts per policy over T=12 |
| 23 | `23_sensitivity_product.png` | 6-panel product-level sensitivity: elasticity vs optimal discount, uplift histogram, segment heatmap, disc_effect vs uplift, top-20 products, best/worst curves | Per-product optimal discount analysis | Sweep 0−45% per product, 10 MC runs each |

### 4.3 Production Pipeline Figure (Generated in `scripts/run_experiments.py`)

| Figure | File | What | Why | How |
|--------|------|------|-----|-----|
| 24 | `24_model_comparison.png` | 4-panel model comparison: R², MAPE, log correlation, lift correlation for 5 models | Compare OLS vs RF vs XGBoost vs LightGBM vs Neural Network | `DemandValidator.compare_models()` on test data |


## 5. Feature Engineering

Feature engineering happens at two levels: **transaction-to-panel aggregation** and **per-product features for segmentation**.

### 5.1 Transaction → Product-Week Panel

The raw data is at the **household−basket−product** level (each row is one item in one basket). The demand model needs **product−week** data.

**Aggregation (in `loader.build_product_weekly_panel()`):**

| Raw Field(s) | Aggregated Feature | Aggregation | Purpose |
|-------------|-------------------|-------------|---------|
| QUANTITY | `quantity` | SUM per product-week | Demand (dependent variable) |
| SALES_VALUE | `revenue` | SUM per product-week | Revenue computation |
| base_price | `total_base_value` → `avg_price` | SUM, then `total_base_value / quantity` | Quantity-weighted average unit price (price regressor) |
| discount_depth | `discount_depth` | MEAN per product-week | Average discount intensity (regressor) |
| discount_flag | `discount_flag` | MAX per product-week | Whether any transaction was discounted |
| QUANTITY | `n_transactions` | COUNT per product-week | Data quality indicator |
| causal.display | `display_pct` | MEAN(on_display) across stores | Fraction of stores with in-store display (continuous 0−1) |
| causal.mailer | `mailer_pct` | MEAN(on_mailer) across stores | Fraction of stores with mailer exposure (continuous 0−1) |
| coupon | `has_coupon` | Binary flag from coupon table | Whether product has any associated coupon |

**Key design choices:**
- **Continuous display/mailer fractions** rather than binary flags — if a product is displayed in 30% of stores vs 90% of stores, the promotional impact differs. Using fractions preserves this variation.
- **Quantity-weighted average price** rather than simple mean price — a $2 item bought 10 times and a $5 item bought once should be weighted toward $2.
- **Combined discount depth** across all three channels (retail, coupon, coupon-match) — the consumer sees the net price regardless of how the discount is funded.

### 5.2 Outlier Removal

Two filters prevent extreme products from distorting the demand models:

1. **Product-level z-score filter**: Flag product-weeks where log(quantity) exceeds 4σ from the product's own mean. This catches data-entry errors and unusual bulk purchases.

2. **Global cap filter**: Remove products whose median log-quantity exceeds the 99.5th percentile of all product medians. This catches weight/volume-based products (e.g., product 6534178) whose "quantity" is in ounces or grams rather than units, producing millions of "units" that distort regressions.

### 5.3 Per-Product Features for Segmentation

Computed in `loader.compute_product_features()`:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `total_revenue` | Σ SALES_VALUE | Product importance ranking |
| `total_qty` | Σ QUANTITY | Volume indicator |
| `n_transactions` | COUNT | Data richness |
| `mean_price` | Mean(avg_unit_price) | Price tier |
| `std_price` | Std(avg_unit_price) | Price variability |
| `pct_on_disc` | Mean(discount_flag) | Promotion frequency |
| `mean_discount_depth` | Mean(discount_depth) | Typical markdown depth |
| `n_weeks` | # unique weeks with sales | Temporal coverage |
| `weekly_velocity` | total_qty / n_weeks | Demand level |
| `price_cv` | std_price / mean_price | Price variation coefficient |
| `log_price` | ln(mean_price) | Log-transformed price (for clustering) |
| `log_velocity` | ln(weekly_velocity) | Log-transformed velocity (for clustering) |

### 5.4 Cross-Product Substitution Feature (AR(1) + Substitution Enhancement)

Added in the AR(1)/substitution enhancement phase (`LogLogDemandModel._build_segment_discount()`):

For each product $i$ in week $t$, compute the mean discount depth of **other** products in the same segment:

$$\text{seg\_disc\_other}_{it} = \frac{\sum_{j \in S(i), j \neq i} d_{jt}}{|S(i)| - 1}$$

where $S(i)$ is the set of products in product $i$'s segment and $d_{jt}$ is the discount depth of product $j$ in week $t$.

This feature captures within-segment cannibalization: when competitors in the same product segment are heavily discounted, own demand decreases.

### 5.5 Lagged Demand Feature

Also added in the AR(1) enhancement:

$$\text{lag\_log\_qty}_{it} = \ln(Q_{i,t-1})$$

This is the log-quantity of the same product in the previous week. The first observation per product is dropped (no lag available), reducing the training set from ~8,630 to ~7,473 observations.


## 6. Product Segmentation (Clustering)

### 6.1 Why Segment?

With 91,905 products, individual estimation for every SKU is impractical — most have too little data. Segmentation serves two purposes:
1. **Group similar products** so we can pool information via shrinkage (borrow strength from segment neighbors)
2. **Define "segments" for substitution** — products in the same segment compete with each other when discounted

### 6.2 Segmentation Features

The `ProductSegmenter` class (`src/segmentation/product_seg.py`) uses K-Means on 5 standardized features:

| Feature | Why Included |
|---------|-------------|
| `log_price` | Price tier (premium vs economy) — determines willingness-to-pay |
| `log_velocity` | Sales velocity — high-volume vs niche products |
| `pct_on_disc` | Promotion frequency — always-on-sale vs rarely-discounted |
| `mean_discount_depth` | Typical markdown depth — shallow vs deep discounters |
| `price_cv` | Price variability — stable-price vs volatile-price products |

**Why elasticity is NOT a segmentation feature**: The bivariate (uncontrolled) elasticity estimated in the EDA is extremely noisy — 57% of products have near-zero or positive elasticity due to omitted variable bias (not controlling for display, mailer, seasonality). Including this noisy estimate would contaminate the clustering without adding signal.

### 6.3 Clustering Procedure

1. **Eligible products**: Products with ≥5 transactions (35,000+ products)
2. **Standardization**: StandardScaler on all 5 features (zero mean, unit variance)
3. **K-Means**: K=12, 10 random initializations, 300 max iterations, random_state=42
4. **K selection**: Evaluated K=5−25 via silhouette score; K=12 selected (best silhouette while maintaining interpretable cluster sizes)

### 6.4 Segment Characteristics (from Latest Run)

The 12 segments (labeled 0−11) roughly correspond to different price-velocity-discount profiles. Some segments have as few as 2 representative products while the largest have 30+, reflecting the proportional revenue allocation.


## 7. Representative Product Selection

### 7.1 Why Select Representatives?

Even after filtering to ~4,000 panel-eligible products, fitting individual demand models for all of them is (a) computationally expensive and (b) yields many unreliable estimates for low-data products. We select 150 representative products that capture the diversity of the assortment.

### 7.2 Stratified Revenue-Tercile Sampling

The `RepresentativeSelector` class (`src/segmentation/representative.py`) implements stratified sampling:

1. **Revenue-proportional allocation**: Each segment gets a share of the 150 product budget proportional to its total revenue, with a minimum of 2 per segment.

2. **Stratified tercile sampling within each segment**:
   - Sort products by total_revenue (descending)
   - Split into 3 tiers: top third, middle third, bottom third
   - Allocate **50%** from the top tier, **30%** from the middle tier, **20%** from the bottom tier
   - Select the highest-revenue products within each tier allocation

3. **Edge case handling**: If a segment has fewer products than its allocation, take all available. Redistribute shortfalls to tiers with surplus.

### 7.3 Why Stratified? (Addressing Review Finding #5)

The original implementation selected only the top-revenue products per segment, creating **head-SKU bias** — overweighting stable, high-velocity items while ignoring mid-range and long-tail products where promotional response may differ. The stratified 50/30/20 split ensures that the 150 representatives span the full revenue distribution within each segment, producing results more representative of the entire assortment.

### 7.4 Data Quality Filters

Products must pass:
- **Temporal coverage**: ≥30 weeks with positive sales (out of 85 stable weeks)
- **Price variation**: Coefficient of variation ≥ 0.03 (need some price movement to estimate elasticity)

These filters are applied **before** stratified selection to ensure every selected product has sufficient data for reliable OLS estimation.


## 8. Demand Model Estimation

### 8.1 The Log-Log OLS Model (`src/models/loglog.py`)

The primary demand model is a per-product log-linear specification:

$$\ln(Q_{it}) = \alpha_i + \epsilon_i \ln(P_{it}) + \gamma_i \cdot d_{it} + \delta_{1i} \cdot \text{display}_{it} + \delta_{2i} \cdot \text{mailer}_{it} + \rho_i \cdot \ln(Q_{i,t-1}) + \lambda_i \cdot \text{seg\_disc\_other}_{it} + \beta_{1i} \cos\!\left(\frac{2\pi \cdot w}{52}\right) + \beta_{2i} \sin\!\left(\frac{2\pi \cdot w}{52}\right) + u_{it}$$

where:
- $\alpha_i$ = product-specific intercept (log of base demand)
- $\epsilon_i$ = own-price elasticity (expected ≤ 0)
- $\gamma_i$ = discount effect (expected ≥ 0, direct lift from discounting)
- $\delta_{1i}, \delta_{2i}$ = display and mailer promotional effects
- $\rho_i$ = AR(1) demand persistence (captures demand "stickiness")
- $\lambda_i$ = within-segment substitution (expected ≤ 0, cannibalization)
- $\beta_{1i}, \beta_{2i}$ = Fourier seasonality (annual cycle)
- $P_{it}$ = quantity-weighted average base price
- $d_{it}$ = continuous discount depth [0, 1]

**Key design features:**

1. **Continuous discount depth** (not binary) — captures the intensity of discounting, not just its presence. A 5% markdown has a different effect than a 40% clearance.

2. **Separate shelf-price elasticity and discount effect** — `log_price` captures changes in the regular shelf price; `discount_depth` captures the promotional effect. This avoids double-counting the price reduction that comes with a discount.

3. **HC1 robust standard errors** — heteroscedasticity-consistent (White) standard errors, since demand variance often increases with price or during promotions.

4. **AR(1) lagged demand** — `lag_log_qty` = ln(Q_{t-1}) enters as a regressor. The first observation per product is dropped. This makes the demand transition **genuinely dynamic** (not memoryless). Min 10 valid observations required after lag construction.

5. **Cross-product substitution** — `seg_disc_other` = mean discount depth of other products in the same segment. Captures cannibalization: when segment competitors are deeply discounted, own demand may decrease.

6. **Fourier seasonality** — cos/sin with period 52 weeks captures the annual demand cycle without using 51 week dummy variables.

### 8.2 Estimation Procedure

For each of the 150 representative products:
1. Sort observations by WEEK_NO
2. Construct lagged log-quantity (drop first observation)
3. Construct seg_disc_other from segment membership
4. Require ≥10 valid observations after lagging
5. Fit OLS with `statsmodels.OLS(...).fit(cov_type='HC1')`
6. Extract: intercept, all 8 coefficients, p-values, standard errors, R², adjusted R², residual std, AIC, BIC

### 8.3 Benchmark Models

Four ML benchmarks are fitted to the same training data for comparison:

| Model | Implementation | Key Settings | Elasticity Extraction |
|-------|---------------|-------------|----------------------|
| **Random Forest** | sklearn `RandomForestRegressor` | 100 trees, max_depth=8 | Finite difference: ΔŷΔ(ln P), evaluated at +1% log price |
| **XGBoost** | `xgboost.XGBRegressor` | 100 estimators, max_depth=5, lr=0.1 | Finite difference |
| **LightGBM** | `lightgbm.LGBMRegressor` | 100 estimators, max_depth=5, lr=0.1 | Finite difference |
| **Neural Network** | Custom PyTorch 2-layer MLP (32→16) | 200 epochs, lr=0.005, dropout=0.1 | Finite difference on standardized inputs |

**Why OLS is preferred for the simulator**: The tree-based and neural network models are trained as flexible predictors and then collapsed to two-parameter log-linear approximations via finite differences. This approximation is necessary because the simulator requires analytic demand functions, not opaque predict() calls. The finite-difference elasticities from ML models are noisy and may not satisfy economic sign constraints. OLS directly estimates the structural parameters that the simulator uses.

### 8.4 Results from Latest Pipeline Run

| Statistic | Value |
|-----------|-------|
| Products fitted | 150 |
| Mean R² | 0.363 |
| Median R² | 0.318 |
| Significant elasticities (p<0.05) | varies by run |
| Significant persistence (p<0.05) | 21/150 |
| Significant substitution (p<0.05) | 17/150 |
| Significant display (p<0.05) | 37/150 |
| Significant mailer (p<0.05) | 29/150 |
| Mean AR(1) coefficient | ~0.039 (after shrinkage) |
| Mean substitution effect | ~−0.852 (after shrinkage) |


## 9. Empirical Bayes Shrinkage & Calibration

### 9.1 Why Shrinkage?

Per-product OLS estimates are noisy for products with few observations or low R². Some will have extreme elasticities (e.g., +5 or −20) purely from sampling noise. Empirical Bayes shrinkage pulls noisy estimates toward the segment median, producing more reliable parameters for simulation.

### 9.2 James-Stein Shrinkage (`src/models/shrinkage.py`)

For each parameter $\theta_i$ (elasticity, disc_effect, display_effect, mailer_effect, demand_persistence, substitution_effect):

$$\hat{\theta}_i^{\text{shrunk}} = w_i \cdot \hat{\theta}_i^{\text{OLS}} + (1 - w_i) \cdot \tilde{\theta}_{\text{seg}}$$

where:
- $\tilde{\theta}_{\text{seg}}$ = median of parameter within the product's segment
- $w_i$ = James-Stein weight = $\frac{\sigma_{\text{seg}}^2}{\sigma_{\text{seg}}^2 + \text{SE}_i^2}$
- $\sigma_{\text{seg}}^2$ = within-segment variance of the parameter
- $\text{SE}_i^2$ = squared standard error of product $i$'s OLS estimate

Products with large standard errors (poor estimation) → small $w_i$ → heavily shrunk toward segment median.
Products with small standard errors (precise estimation) → $w_i$ up to max_weight=0.9 → retain most of their individual estimate.

### 9.3 Post-Shrinkage Constraints

After shrinkage, structural economic constraints are enforced:

| Parameter | Constraint | Rationale |
|-----------|-----------|-----------|
| **Elasticity** | Winsorize at [P2.5, P97.5], then clip ≤ 0 | Law of demand: higher price → lower demand. Positive estimates arise from endogeneity. |
| **Discount effect** | Winsorize at [P2.5, P97.5] | Remove extreme outliers |
| **Display effect** | Winsorize at [P2.5, P97.5] | Remove extreme outliers |
| **Mailer effect** | Winsorize at [P2.5, P97.5] | Remove extreme outliers |
| **Demand persistence (ρ)** | Winsorize, then clip to [0, 0.95] | Stationarity: ρ≥1 → explosive demand. ρ<0 → negative persistence (economically implausible). |
| **Substitution (λ)** | Winsorize, then clip ≤ 0 | Cannibalization: competitors' discounts should reduce (not increase) own demand. |

### 9.4 Intercept Calibration

The post-shrinkage intercept is recalibrated so that at training-period **mean** values of all covariates, the model predicts the **observed mean demand**:

$$\alpha_i^{\text{cal}} = \ln(\bar{Q}_i) - \hat{\epsilon}_i^s \ln(\bar{P}_i) - \hat{\gamma}_i^s \bar{d}_i - \hat{\delta}_{1i}^s \bar{D}_i - \hat{\delta}_{2i}^s \bar{M}_i - \hat{\lambda}_i^s \overline{\text{seg\_disc\_other}}_i$$

The AR(1) persistence term does **not** appear in this equation because the mean deviation of lagged log-demand from its own mean is zero by construction over the training period.

### 9.5 Duan Smearing Correction (φ)

The log-linear model predicts $\ln(Q)$, but the simulator needs $Q$. Naive exponentiation $\exp(\hat{\ln Q})$ systematically **underpredicts** because $E[\exp(X)] > \exp(E[X])$ for any non-degenerate random variable (Jensen's inequality).

The Duan (1983) smearing correction computes:

$$\phi = \frac{\sum_i \sum_t Q_{it}^{\text{obs}}}{\sum_i \sum_t \exp(\widehat{\ln Q}_{it})}$$

This φ includes the effects of AR(1) deviation and substitution deviation terms in the predicted log-demand. The correction is applied multiplicatively: $\hat{Q} = \phi \cdot \exp(\widehat{\ln Q})$.

**Latest value**: φ = 0.698

### 9.6 Shrinkage Results (Latest Run)

| Parameter | Mean Shrunk | Nonzero Count | Interpretation |
|-----------|-------------|---------------|----------------|
| Elasticity | −0.773 | All 150 (91 negative) | Most products show weak but negative price sensitivity |
| Discount effect | 2.263 | Most nonzero | Strong positive lift from discounting |
| Display effect | 0.518 | 148/150 | Moderate display uplift |
| Mailer effect | 0.217 | 135/150 | Smaller mailer uplift |
| Demand persistence | 0.039 | 64/150 | Weak but present AR(1) — demand has some "memory" |
| Substitution | −0.852 | 80/150 | Moderate cannibalization when segment competitors are discounted |


## 10. Simulator Construction

### 10.1 MDP Formulation (`src/simulator/product_level.py`)

The simulator is a finite-horizon Markov Decision Process (MDP) with:

**State space** ($4N + 2$ dimensions, where $N = 150$):
- **selection** ($N$): Binary vector — which products are currently discounted
- **discount** ($N$): Discount depth per product (0 to max_discount)
- **demand** ($N$): Most recent demand realization per product (feeds into AR(1))
- **budget_consumed** ($N$): Cumulative discount cost per product
- **budget_remaining** (1): Remaining total discount budget
- **time_remaining** (1): Remaining periods

Total state dimension: $4 \times 150 + 2 = 602$

**Action space** ($2N$ dimensions):
- **new_selection** ($N$): Binary — which products to discount next period
- **new_discount** ($N$): Discount depth per product for next period

Total action dimension: $2 \times 150 = 300$

### 10.2 Demand Transition

The demand transition is the heart of the simulator:

$$\ln(\hat{Q}_{it}) = \alpha_i^{\text{cal}} + \epsilon_i^s \ln(P_i^{\text{base}}) + \gamma_i^s \cdot (\text{sel}_i \cdot d_i)$$
$$\quad + \rho_i^s \cdot (\ln(Q_{i,t-1}) - \overline{\ln Q_i})$$
$$\quad + \lambda_i^s \cdot (\text{seg\_disc\_other}_{it} - \overline{\text{seg\_disc\_other}}_i)$$
$$\quad + \beta_{1i} (\cos_w - \overline{\cos}) + \beta_{2i} (\sin_w - \overline{\sin})$$
$$\quad + \delta_{1i}^s (D_{it} - \bar{D}_i) + \delta_{2i}^s (M_{it} - \bar{M}_i) + \varepsilon_{it}$$

$$\hat{Q}_{it} = \phi \cdot \exp(\ln(\hat{Q}_{it}))$$

**Key points:**
- All non-intercept terms enter as **deviations from training-period means** — the intercept calibration absorbed the means, so the simulator only needs to model deviations.
- **AR(1) persistence** ($\rho_i$): The previous period's demand ($Q_{i,t-1}$) feeds forward. This makes the MDP state genuinely **non-memoryless** — the demand in period $t$ depends on the demand realization in period $t-1$.
- **Cross-product substitution** ($\lambda_i$): The mean discount depth of other products in the same segment is computed **in real time** during simulation. If the policy discounts multiple products in the same segment, they cannibalize each other.
- **Stochastic promotional events**: Display and mailer draws from truncated normal distributions $D_{it} \sim \text{TN}(0,1;\mu_D,\sigma_D)$ clipped to $[0,1]$.
- **Multiplicative noise**: $\varepsilon_{it} \sim N(0, \sigma_i^2)$ applied as $\exp(\varepsilon)$ to the demand level.

### 10.3 Budget Enforcement

Each period:
1. Compute discount cost: $\text{cost}_{it} = P_i^{\text{base}} \cdot \text{sel}_i \cdot d_i \cdot Q_{it}$
2. If total cost exceeds remaining budget, **scale all discounts proportionally**: $d_i \leftarrow d_i \cdot (B^{\text{rem}} / \text{total\_cost})$
3. Recompute demand with scaled discounts
4. Update $B^{\text{rem}} \leftarrow B^{\text{rem}} - \text{total\_cost}$

### 10.4 Budget Computation

The total budget equals the actual historical discount expenditure (at observed discount depths and quantities) scaled to the simulator's 12-period horizon:

$$B = \frac{\sum_{i,t \in \text{train}} P_i \cdot d_{it} \cdot Q_{it}}{|\text{train weeks}|} \times 12$$

**Latest value**: $9,068

### 10.5 Horizon and Timing

- **Horizon**: $T = 12$ periods (roughly one quarter)
- **Start week**: 83 (beginning of test period)
- **Calendar-aware seasonality**: Cosine/sine terms use the actual calendar week


## 11. Policy Evaluation & Sensitivity Analysis

### 11.1 Heuristic Policies (`src/evaluation/policy.py`)

Seven policies are evaluated via Monte Carlo rollouts (50 runs each):

| Policy | Description |
|--------|-------------|
| **No Discount** | Never discount any product (baseline) |
| **Uniform 10%** | Discount all 150 products at 10% |
| **Uniform 20%** | Discount all 150 products at 20% |
| **Target Top-30 (elasticity)** | Discount only the 30 most price-elastic products at 20% |
| **Target Top-30 (disc effect)** | Discount only the 30 products with highest discount effect at 20% |
| **Front-Load** | Start at 30% discount on all products, decline linearly to 5% |
| **Target Top-30 (ROI)** | Discount the 30 products with best return-on-investment at 20% |

### 11.2 Policy Results (Latest Run)

| Policy | Mean Revenue ($) | vs No Discount |
|--------|-----------------|----------------|
| Target Top-30 (disc effect) | 68,755 | **+41.3%** |
| Uniform 20% | 55,902 | +14.9% |
| Uniform 10% | 51,369 | +5.6% |
| Target Top-30 (elasticity) | 50,117 | +3.0% |
| Front-Load | varies | varies |
| Target Top-30 (ROI) | varies | varies |
| No Discount | 48,653 | --- |

**Key finding**: Targeting the top-30 products by **discount effect** (the direct promotional lift coefficient $\gamma_i$) is far superior to targeting by elasticity, ROI, or applying uniform discounts. This is because the discount effect coefficient directly measures the demand response to promotional markdowns, while the shelf-price elasticity measures response to permanent price changes — a different mechanism.

### 11.3 Per-Product Sensitivity Analysis (`src/evaluation/sensitivity.py`)

For each of the 150 products:
1. Sweep discount depth from 0% to 45% in 5% increments
2. For each depth, run 20 Monte Carlo simulations (varying display/mailer/demand noise)
3. Compute expected revenue at each discount level
4. Identify **optimal discount** and **revenue uplift** vs no-discount baseline

**Results (Latest Run):**
- Products benefiting from discounting: **97/150 (64.7%)**
- Mean optimal discount among benefiting products: **26.3%**
- Median revenue uplift: **105.1%**
- Mean revenue uplift: **503.8%** (skewed by a few products with very high lifts)
- IQR uplift: **36.8% — 288.8%**

### 11.4 Per-Segment Sensitivity Summary

Results are aggregated by segment in `results/sensitivity_by_segment.csv`, showing which segments are most responsive to discounting and the typical optimal discount depth per segment.


## 12. Validation

### 12.1 Validation Metrics (`src/evaluation/validation.py`)

The `DemandValidator` class computes:

1. **Log-space correlation**: Pearson correlation between predicted and observed log-demand on test data. Measures ranking accuracy.
2. **Weekly aggregate MAPE**: Mean Absolute Percentage Error of weekly *aggregate* (sum across all products) demand predictions. Measures level accuracy.
3. **Lift correlation**: For each product, compare observed demand lift from discounting (mean discounted demand / mean non-discounted demand) vs predicted lift (from the model's elasticity + discount effect). Measures promotional response accuracy.
4. **Correct direction percentage**: What fraction of products have the *sign* of the lift correct (both observed and predicted > 1, or both ≤ 1).

### 12.2 Validation Results (Latest Run)

| Metric | OLS (Simulator) | Random Forest | XGBoost | LightGBM | Neural Net |
|--------|-----------------|---------------|---------|----------|------------|
| Mean R² | 0.363 | 0.551 | 0.606 | 0.156 | varies |
| Median R² | 0.318 | 0.586 | 0.641 | 0.056 | varies |
| Log correlation | 0.621 | 0.715 | 0.704 | 0.694 | varies |
| Weekly MAPE | 46.1% | 18.0% | 22.4% | 21.2% | varies |
| Lift correlation | **0.694** | 0.503 | 0.480 | 0.171 | varies |

**Why OLS is selected despite higher MAPE**: The OLS model has the **highest lift correlation** (0.694), which is the most important metric for a discount optimization simulator. Lift correlation measures how well the model predicts the *directional and proportional response* to discounting — exactly what the simulator needs to rank and optimize discount strategies. The higher MAPE (46.1% vs 18.0% for RF) is largely driven by level accuracy, which the Duan φ correction partially addresses.

### 12.3 Validation Interpretation

The MAPE increased from ~21.6% (before AR(1) + substitution) to ~46.1% (after). This tradeoff occurs because:
1. **Dropping the first observation** per product for AR(1) lag reduces the training sample
2. **Adding two more noisy regressors** (persistence and substitution) can increase variance of predictions
3. However, the **lift correlation is preserved** at 0.694, confirming that the model's promotional response estimates remain reliable — which is what matters for discount optimization

The median elasticity improved from −0.123 to −0.215, closer to (though still well below) the expected range from the literature, suggesting that the AR(1) and substitution terms absorb some omitted-variable bias that previously attenuated the elasticity estimate.


## 13. Key Results

### 13.1 Pipeline Metrics Summary

| Metric | Value |
|--------|-------|
| Dataset | 2,576,815 transactions, 91,905 products, 102 weeks |
| Eligible products (panel) | ~4,000 |
| Product segments | 12 (K-Means) |
| Representative products | 150 (stratified tercile sampling) |
| Training observations | 7,473 |
| Test observations | 2,359 |
| Parameters shrunk per product | 6 (elasticity, disc effect, display, mailer, persistence, substitution) |
| Duan correction (φ) | 0.698 |
| Simulator budget | $9,068 (12 periods) |
| Simulator state dimension | 602 |
| Simulator action dimension | 300 |

### 13.2 Demand Model Performance

| Metric | Value |
|--------|-------|
| Mean R² (OLS) | 0.363 |
| Log correlation (test) | 0.621 |
| Weekly MAPE (test) | 46.1% |
| Lift correlation (test) | 0.694 |
| Correct lift direction | 77.8% |
| Median shrunk elasticity | −0.215 |
| Median shrunk discount effect | 1.982 |
| Mean shrunk persistence (ρ) | 0.039 |
| Mean shrunk substitution (λ) | −0.852 |

### 13.3 Best Policy

**Target Top-30 (discount effect)** at 20% discount depth:
- Revenue: $68,755 (vs $48,653 no-discount baseline)
- Improvement: **+41.3%**
- Budget utilization: within $9,068 constraint

### 13.4 Products That Benefit from Discounting

- **97/150 (64.7%)** of products show positive revenue uplift at some discount level
- **53/150 (35.3%)** should optimally receive no discount (demand response doesn't offset price reduction)
- Mean optimal discount (among benefiting products): **26.3%**
- Median revenue uplift: **105.1%** (the typical benefiting product more than doubles its revenue)


## 14. Project Structure

```
/Users/ajit/Documents/Pricing/
│
├── data/                           # Dunnhumby Complete Journey dataset
│   ├── transaction_data.csv        # 2.6M transactions
│   ├── product.csv                 # 92K products
│   ├── hh_demographic.csv          # 801 household demographics
│   ├── causal_data.csv             # 36.8M store-product-week display/mailer flags
│   ├── coupon.csv                  # 124K coupon-product associations
│   ├── coupon_redempt.csv          # 2.3K coupon redemptions
│   ├── campaign_desc.csv           # 30 campaign descriptions
│   └── campaign_table.csv          # 7.2K household-campaign targeting
│
├── src/                            # Core library (production code)
│   ├── data/
│   │   └── loader.py               # DunnhumbyLoader: load, clean, aggregate, build panel
│   ├── segmentation/
│   │   ├── product_seg.py          # ProductSegmenter: K-Means on 5 features (K=12)
│   │   └── representative.py       # RepresentativeSelector: stratified tercile sampling
│   ├── models/
│   │   ├── loglog.py               # LogLogDemandModel: per-product OLS with AR(1)+substitution
│   │   ├── shrinkage.py            # EmpiricalBayesShrinkage: J-S shrinkage + calibration + φ
│   │   ├── random_forest.py        # RandomForestDemandModel: benchmark
│   │   ├── gradient_boost.py       # GradientBoostDemandModel: XGBoost/LightGBM benchmark
│   │   └── neural_net.py           # NeuralNetDemandModel: PyTorch MLP benchmark
│   ├── simulator/
│   │   └── product_level.py        # ProductLevelSimulator: 602-state MDP with AR(1)+substitution
│   └── evaluation/
│       ├── validation.py           # DemandValidator: log-corr, MAPE, lift correlation
│       ├── policy.py               # PolicyEvaluator: 7 heuristic policies, MC rollouts
│       └── sensitivity.py          # SensitivityAnalyzer: per-product discount sweep
│
├── scripts/
│   ├── run_experiments.py          # Main pipeline: data → models → shrinkage → simulator → policies
│   ├── extract_metrics.py          # Helper: extract key metrics from CSVs
│   └── critical_review_v2.py       # Logical consistency validator
│
├── notebooks/                      # (empty — notebooks at project root)
│
├── dunnhumby_exploration.ipynb     # Main EDA notebook (47 cells, 19 sections)
├── 01_data_exploration.ipynb       # Earlier Olist exploration (abandoned)
│
├── figures/                        # Generated visualizations (24 figures)
│   ├── 01_time_series_overview.png
│   ├── 02_price_discount_patterns.png
│   ├── 03_elasticity_distribution.png
│   ├── 04_customer_behavior.png
│   ├── 05_customer_segments.png
│   ├── 06_correlation_analysis.png
│   ├── 07_kmeans_evaluation.png
│   ├── 08_product_segments.png
│   ├── 09_segment_elasticities.png
│   ├── 10_simulator_rollout.png
│   ├── 11_demand_validation.png
│   ├── 12_aggregate_validation.png
│   ├── 13_elasticity_recovery.png
│   ├── 14_policy_rollout.png
│   ├── 15_sensitivity_analysis.png
│   ├── 16−20: product_validation iterations
│   ├── 21_elasticity_recovery_product.png
│   ├── 22_policy_rollout_product.png
│   ├── 23_sensitivity_product.png
│   └── 24_model_comparison.png
│
├── results/                        # Saved CSV outputs
│   ├── model_comparison.csv        # 5-model comparison metrics
│   ├── product_level_params.csv    # 150 products × 40+ columns of parameters
│   ├── product_segments.csv        # Product → segment mapping
│   ├── segment_elasticities.csv    # Per-segment demand summaries
│   ├── aggregate_segment_elasticities.csv
│   ├── policy_results.csv          # 7 policies × revenue/demand/cost
│   ├── sensitivity_results.csv     # 150 products × optimal discount/uplift
│   └── sensitivity_by_segment.csv  # Per-segment sensitivity summary
│
├── report_final.tex                # LaTeX report (comprehensive)
├── report_final.pdf                # Compiled PDF
├── PROMOTION_DISCOUNTING_REVIEW.md # Critical review with 7 findings
├── README.md                       # Project overview
└── pricing_env/                    # Python virtual environment
```

### 14.1 Pipeline Execution Flow

Running `python scripts/run_experiments.py` executes the complete pipeline:

1. **Load data** → `DunnhumbyLoader` reads CSVs, reconstructs prices, filters returns
2. **Build panel** → Aggregate to product-week, merge causal data, apply quality filters
3. **Segment** → K-Means on 35K+ products → 12 clusters
4. **Select representatives** → Stratified tercile sampling → 150 products
5. **Train/test split** → Weeks 18−82 (train) / 83−102 (test)
6. **Build substitution features** → seg_disc_other on both train and test panels
7. **Fit 5 demand models** → OLS + RF + XGBoost + LightGBM + Neural Net
8. **Shrink + calibrate** → James-Stein shrinkage, intercept calibration, φ computation
9. **Compare models** → Test-set validation metrics → select best (OLS for structural reasons)
10. **Build simulator** → ProductLevelSimulator with calibrated parameters
11. **Evaluate policies** → 7 heuristic policies, 50 MC rollouts each
12. **Sensitivity analysis** → Per-product discount sweep (0−45%, 20 runs each)
13. **Save results** → CSVs + figure + summary statistics

Total runtime: approximately 10−15 minutes on a modern laptop.
