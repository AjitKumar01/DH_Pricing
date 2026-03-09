# Product-Level Markdown Pricing Simulator

A multi-model demand estimation and MDP-based discount optimization framework applied to the Dunnhumby Complete Journey dataset.

## Overview

This project builds a **product-level markdown pricing simulator** that:

1. **Estimates individual demand models** for 150 representative products using log-log OLS with promotional controls (display, mailer) and Fourier seasonality
2. **Segments 35,000+ products** into 12 clusters via K-Means on 5 behavioral features, then selects representatives via revenue-proportional allocation
3. **Applies James–Stein empirical Bayes shrinkage** with winsorization and economic sign constraints to regularize per-product parameters
4. **Simulates a finite-horizon MDP** (602-dimensional state, 300-dimensional action) with budget-constrained discount decisions and stochastic promotional events
5. **Evaluates discount policies** via Monte Carlo rollouts, finding that targeted discounting improves revenue by **42.5%** over the no-discount baseline

## Results

| Metric | Value |
|--------|-------|
| Out-of-sample MAPE | 12.0% |
| Log correlation | 0.663 |
| Lift correlation | 0.651 |
| Best policy uplift | +42.5% |
| Products benefiting | 70.0% |
| Median revenue uplift | 218.2% |

## Project Structure

```
├── src/                        # Core library
│   ├── data/
│   │   └── loader.py           # DunnhumbyLoader: load, clean, build panel
│   ├── models/
│   │   ├── loglog.py           # Log-log OLS demand model
│   │   ├── random_forest.py    # Random Forest benchmark
│   │   ├── gradient_boost.py   # XGBoost / LightGBM benchmarks
│   │   ├── neural_net.py       # Neural network benchmark
│   │   └── shrinkage.py        # James-Stein empirical Bayes shrinkage
│   ├── segmentation/
│   │   ├── product_seg.py      # K-Means product segmentation
│   │   └── representative.py   # Revenue-proportional representative selection
│   ├── simulator/
│   │   └── product_level.py    # Product-level MDP simulator
│   └── evaluation/
│       ├── validation.py       # Out-of-sample validation metrics
│       ├── sensitivity.py      # Per-product discount sensitivity analysis
│       └── policy.py           # Heuristic policy evaluation
├── scripts/
│   ├── run_experiments.py      # Main pipeline: data → model → simulator → policies
│   └── critical_review_v2.py   # End-to-end logical consistency validator
├── data/                       # Dunnhumby dataset (not tracked)
├── figures/                    # Generated plots
├── results/                    # Saved model parameters and results
├── report_final.tex            # LaTeX report
└── README.md
```

## Setup

```bash
python3 -m venv pricing_env
source pricing_env/bin/activate
pip install pandas numpy scikit-learn statsmodels xgboost lightgbm torch matplotlib seaborn
```

## Data

Download the [Dunnhumby Complete Journey](https://www.dunnhumby.com/source-files/) dataset and place the CSV files in `data/`:

- `transaction_data.csv` (2.6M transactions)
- `product.csv` (92K products)
- `causal_data.csv` (36.8M store-product-week promotional indicators)
- `hh_demographic.csv`, `coupon.csv`, `coupon_redempt.csv`, `campaign_table.csv`, `campaign_desc.csv`

## Usage

Run the full pipeline:

```bash
python scripts/run_experiments.py
```

This executes all steps: data loading → panel construction → segmentation → representative selection → model fitting (5 models) → shrinkage → validation → simulator construction → policy evaluation → sensitivity analysis.

Validate pipeline consistency:

```bash
python scripts/critical_review_v2.py
```

## Methodology

### Demand Model

$$\ln Q_{it} = \alpha_i + \varepsilon_i \ln p_{it}^{\text{shelf}} + \gamma_i d_{it} + \delta_i^D D_{it} + \delta_i^M M_{it} + \beta_{i,1} \cos(2\pi w_t/52) + \beta_{i,2} \sin(2\pi w_t/52) + u_{it}$$

The shelf price captures structural pricing; the discount depth captures promotional response. The simulator holds shelf price **fixed** and varies only discount depth — avoiding the common double-counting error where discounts are applied to both the price elasticity and the discount effect.

### Data Quality

Systematic data quality review removed returns (QUANTITY ≤ 0), zero-price transactions, and extreme volume outliers. Bivariate elasticity (57% near-zero) was dropped from clustering features. Feature computation was aligned to the stable period (weeks 18–102).

### Shrinkage

James–Stein weights: $w_i = \sigma_s^2 / (\sigma_s^2 + \sigma_i^2/n_i)$, capped at 0.9. Post-shrinkage winsorization at P2.5/P97.5. Economic sign constraint: $\varepsilon_i \leq 0$.

### Simulator

The MDP simulator generates stochastic promotional events (display/mailer) each period based on per-product historical frequencies, adding demand effects as deviations from the mean to preserve calibration.

## Report

Compile the LaTeX report:

```bash
pdflatex report_final.tex
pdflatex report_final.tex  # twice for references
```
