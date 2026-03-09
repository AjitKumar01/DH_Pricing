# Promotion-Based Discounting Review

## Scope

This review covers the end-to-end promotion-discounting pipeline in the current codebase and the report:

- `src/data/loader.py`
- `src/segmentation/product_seg.py`
- `src/segmentation/representative.py`
- `src/models/loglog.py`
- `src/models/random_forest.py`
- `src/models/gradient_boost.py`
- `src/models/neural_net.py`
- `src/models/shrinkage.py`
- `src/simulator/product_level.py`
- `src/evaluation/validation.py`
- `src/evaluation/policy.py`
- `src/evaluation/sensitivity.py`
- `scripts/run_experiments.py`
- `scripts/critical_review_v2.py`
- `report_final.tex`

The focus is narrow and explicit: the theory and implementation of promotion-based discounting, whether the simulator logically follows from the estimated demand model, and whether the report accurately describes what the code is doing.

## Executive Verdict

The current system is a credible research prototype for product-level promotion response under a discount budget, but it is not yet a simulator that fully captures retail market dynamics.

What is solid:

- The code now correctly separates shelf-price effects from discount-depth effects in the simulator demand equation.
- Promotional covariates from `causal_data.csv` are incorporated into estimation instead of being omitted entirely.
- The pipeline uses a temporal split, shrinkage, intercept calibration, and Duan-style level correction, which is directionally sound.
- The simulator state and action spaces do operate at the product level rather than the segment level.

What remains materially incomplete:

- The simulator treats products as conditionally independent demand generators; it does not model substitution, cannibalization, assortment competition, or household choice.
- Several state variables are present in the MDP but do not actually influence future demand transitions.
- Some promotion accounting is still incomplete at the data layer.
- Important evaluation artifacts and report tables remain internally inconsistent.

The net assessment is: strong engineering progress, but the current simulator is still a reduced-form demand-response engine, not a fully behavioral retail market simulator.

## Findings

### 1. Coupon-match discounts are normalized but then excluded from price reconstruction

Severity: High

Evidence:

- `src/data/loader.py:45` converts `COUPON_MATCH_DISC` to absolute value.
- `src/data/loader.py:50-56` reconstructs `base_price` and `discount_depth` using only `RETAIL_DISC` and `COUPON_DISC`.

Why this matters:

- The code acknowledges coupon-match discounts as a legitimate discount component, but then drops them from both the reconstructed shelf price and the promotion depth.
- That understates promotional intensity for transactions where coupon matching matters.
- It also biases both the price regressor and the promotion regressor in opposite directions: shelf price is too low and discount depth is too low.

Implication:

- Estimated `disc_effect` and any downstream simulator response to discounting are biased for a nontrivial subset of transactions.

Assessment:

- This is a real implementation inconsistency, not a stylistic preference.

### 2. The simulator’s transition function barely uses the state it exposes

Severity: High

Evidence:

- The declared state is `[selection, discount, demand, budget_consumed, budget_remaining, time_remaining]` in `src/simulator/product_level.py:4-5`.
- Demand is computed in `_compute_demand()` from static per-product parameters plus the current action only, `src/simulator/product_level.py:126-170`.
- Previous demand, previous selection, and cumulative budget consumption do not enter the demand equation.
- `time_remaining` only affects episode termination in `src/simulator/product_level.py:221-222`; it does not affect demand.

Why this matters:

- The state includes demand and prior promotion variables, but the next demand realization is essentially memoryless given the action.
- That means the MDP is much thinner than advertised. It is closer to repeated static response with a budget stock than to a dynamic retail system.

Implication:

- The simulator does not capture intertemporal promotion dynamics such as stockpiling, purchase acceleration, post-promotion dips, habituation, or reference-price adaptation.
- The presence of `demand` in the state is mostly observational, not causal.

Assessment:

- This is the biggest theoretical gap between the stated ambition and the actual simulator.

### 3. There is no customer choice or substitution mechanism across products

Severity: High

Evidence:

- Product demand is computed independently for each product in `src/simulator/product_level.py:146-170`.
- No demand term depends on competitor prices, substitute discounts, category totals, household choice sets, or assortment-wide promotional pressure.

Why this matters:

- Promotion-based discounting in retail is not just own-item lift. It also reallocates demand across substitute items and brands.
- Without substitution, the simulator can overstate the total category lift from broad promotions because all promoted products can lift simultaneously without stealing share from each other.

Implication:

- The simulator is not a market-clearing or category-share model.
- It cannot distinguish between true category expansion and within-category cannibalization.

Assessment:

- This is acceptable for a first reduced-form prototype, but it is not consistent with the stronger report language around “market dynamics”.

### 4. Seasonality is estimated but intentionally not simulated

Severity: High

Evidence:

- OLS includes `cos_season` and `sin_season` in `src/models/loglog.py:60-63`.
- Intercept calibration in `src/models/shrinkage.py:128-132` absorbs average conditions into the intercept.
- The simulator demand equation in `src/simulator/product_level.py:146-170` does not include any time-varying seasonal term.
- The report explicitly says the simulator represents “average” seasonal conditions in `report_final.tex:955-958`.

Why this matters:

- Weekly promotion policy over a 12-period horizon should be influenced by calendar position if seasonality is a meaningful driver.
- The current simulator ignores any week-specific demand environment except random display/mailer deviations.

Implication:

- The simulator cannot evaluate whether the same promotion should be deployed differently early vs late in a seasonal cycle.
- `time_remaining` has operational meaning only for budget exhaustion and horizon, not for demand evolution.

Assessment:

- This is a deliberate simplification, but it materially weakens the claim that the simulator tracks market dynamics over time.

### 5. Representative selection is not truly representative; it is head-SKU biased

Severity: Medium-High

Evidence:

- `RepresentativeSelector.select()` allocates by segment revenue and then selects the top-revenue products within each segment in `src/segmentation/representative.py:35-50`.

Why this matters:

- High-revenue products are important, but they are not the same thing as representative products.
- This procedure overweights stable, well-supported, high-velocity items and underweights the long tail, intermittent movers, and low-base products where promotion behavior can differ.

Implication:

- The fitted 150-product universe is better interpreted as a “commercially important subset” than as a representative sample of the retailer’s assortment.
- General statements in the report should reflect that narrower scope.

Assessment:

- The current approach is operationally defensible, but the terminology in the report overstates representativeness.

### 6. Panel price aggregation uses an unweighted average of transaction unit prices

Severity: Medium

Evidence:

- `build_product_weekly_panel()` aggregates `avg_price=("avg_unit_price", "mean")` in `src/data/loader.py:149-156`.

Why this matters:

- If a product sells at multiple prices within a week and transaction sizes differ, the simple mean of transaction unit prices is not the same as revenue divided by quantity.
- For promotion analysis, the quantity-weighted effective unit price is often the more relevant weekly regressor.

Implication:

- Price elasticity and discount-depth estimates can be biased when a week contains mixed-price transactions with heterogeneous basket sizes.

Assessment:

- This is not fatal, but it is a real measurement-choice issue that should be acknowledged.

### 7. Causal promotional features are heavily simplified before estimation

Severity: Medium

Evidence:

- Store-level causal data is reduced to product-week fractions and then binary indicators in `src/data/loader.py:89-115`.
- `has_display` and `has_mailer` are used in the models, not `display_pct` and `mailer_pct`, see `src/models/loglog.py:56-59` and the ML feature builders.

Why this matters:

- A product displayed in 1 out of 100 stores is treated the same as a product displayed in 100 out of 100 stores.
- That collapses intensity into incidence and throws away potentially important identifying variation.

Implication:

- The estimated display and mailer effects are likely attenuated or distorted relative to an exposure-intensity specification.

Assessment:

- The code uses causal data, which is a major improvement, but it does so in a coarse way.

### 8. The shrinkage variance proxy is not parameter-specific

Severity: Medium

Evidence:

- Shrinkage weight uses `residual_std^2 / n_obs` for both elasticity and discount effect in `src/models/shrinkage.py:56-79`.

Why this matters:

- Residual variance is not the same thing as the standard error of the price coefficient or the discount coefficient.
- Two products can have the same residual variance but very different uncertainty on elasticity depending on their price variation.

Implication:

- The James-Stein style pooling is heuristic rather than statistically aligned with the actual coefficient uncertainty.

Assessment:

- This is acceptable as regularization engineering, but the report should not oversell it as precise empirical Bayes inference on coefficient posteriors.

### 9. Post-shrinkage winsorization and sign constraints make the simulator policy-safe, but less empirically faithful

Severity: Medium

Evidence:

- Winsorization at the 2.5th/97.5th percentiles is applied in `src/models/shrinkage.py:82-89`.
- Elasticities are then clipped to be non-positive in `src/models/shrinkage.py:91-98`.

Why this matters:

- These steps are reasonable for preventing pathological policies and enforcing the law of demand.
- But they are ex post structural edits to the empirical estimates, not direct outputs of the data.

Implication:

- The simulator is more stable for optimization, but less faithful as a pure measurement device.
- The distinction between “estimated from data” and “constrained for simulation” should remain explicit.

Assessment:

- This is a legitimate modeling choice, but it is a choice, not a discovered empirical fact.

### 10. The ML benchmark models are not structurally comparable to the simulator they are supposed to feed

Severity: Medium

Evidence:

- For RF/XGBoost/LightGBM/NN, elasticity and discount effect are extracted via finite differences in `src/models/random_forest.py:64-74`, `src/models/gradient_boost.py:76-84`, and `src/models/neural_net.py:98-112`.

Why this matters:

- Those models are trained as flexible predictors, then collapsed back to a two-parameter log-linear approximation for simulation.
- This is not equivalent to simulating the original ML model.

Implication:

- The ML benchmarks are useful as predictive comparators, but not as directly interpretable simulator engines.
- Choosing OLS for the simulator is sensible, but the report should present this as a structural compatibility decision, not just a performance win.

Assessment:

- The current setup is reasonable, but the theoretical comparison across model classes is not one-to-one.

### 11. The policy evaluator stores per-period revenue from only the last Monte Carlo run

Severity: Medium

Evidence:

- In `src/evaluation/policy.py:69-90`, aggregate statistics are averaged over `n_runs`, but `per_period_revenue` is set from `history` after the loop, which corresponds only to the final replication.

Why this matters:

- The headline revenue means and confidence intervals are computed correctly.
- But any downstream figure or report that interprets `per_period_revenue` as an average trajectory would be wrong.

Implication:

- The policy-level temporal profile is not Monte Carlo averaged even though the table suggests replicated evaluation.

Assessment:

- This is a contained implementation defect.

### 12. The sensitivity analysis bypasses key simulator mechanics

Severity: Medium

Evidence:

- `SensitivityAnalyzer.per_product_sweep()` computes closed-form revenue from intercept, elasticity, and discount effect only in `src/evaluation/sensitivity.py:30-46`.
- It does not use stochastic display/mailer deviations, budget coupling, or multi-product interaction.

Why this matters:

- The sensitivity output is not a simulator-based marginal value analysis; it is a deterministic single-product response calculation.

Implication:

- The sensitivity section should be interpreted as an analytical approximation, not as a full policy simulation.

Assessment:

- Acceptable if labeled correctly; misleading if presented as full-environment sensitivity.

### 13. The internal “critical review” script is a pipeline audit, not an independent validation

Severity: Medium

Evidence:

- `scripts/critical_review_v2.py` rebuilds the same pipeline and checks thresholds chosen inside the project itself.

Why this matters:

- This is useful for regression testing and consistency checks.
- But it is not external validation, nor does it challenge the structural assumptions of the simulator.

Implication:

- Statements like “zero critical issues” should be interpreted narrowly: no critical issues under the project’s current assumptions and thresholds.

Assessment:

- Good internal QA, not a substitute for methodological review.

### 14. The report contains stale or contradictory segmentation results

Severity: High

Evidence:

- The report says products were segmented into 12 clusters in `report_final.tex:500`.
- The segment allocation table at `report_final.tex:512-532` lists only 11 segment rows.
- The same table reports positive median elasticities for several segments, while the report later states “All products have non-positive elasticity” at `report_final.tex:814`.

Why this matters:

- These are direct internal inconsistencies in the current report.
- A reader cannot tell whether the sign constraint was applied before or after the segment summary table was generated.

Implication:

- The report cannot currently be treated as a faithful frozen snapshot of the code outputs.

Assessment:

- This is the most important report-level issue.

### 15. The report overstates “market dynamics” relative to what is implemented

Severity: High

Evidence:

- The simulator section in `report_final.tex:840-1115` presents a dynamic MDP.
- But the implemented demand process has no lagged demand, no household stockpiling, no post-promotion dip, no substitution, and no inventory/stock constraints.

Why this matters:

- The implemented system captures static own-product promotional response under budget constraints.
- It does not capture the broader retail dynamics usually implied by “market dynamics”.

Implication:

- The report should narrow its claims to “calibrated product-level promotion response simulator” unless those omitted dynamics are explicitly added.

Assessment:

- This is primarily a claims-management issue, but an important one.

## Positive Checks

The review also found several places where the code is now theoretically consistent and better than earlier iterations.

### A. Shelf price and discount depth are correctly separated in the simulator

Evidence:

- `src/simulator/product_level.py:132-149` keeps `log(shelf_price)` fixed and applies only `discount_depth` through `disc_effect`.

Why this is good:

- It avoids the earlier double-counting bug where discounting changed both the price regressor and the discount regressor.

### B. The chosen simulator model family is structurally aligned with the simulator equation

Evidence:

- `scripts/run_experiments.py:197-208` selects the best simulator model from the calibrated candidates and effectively prefers the structurally coherent OLS family.

Why this is good:

- The simulator needs interpretable coefficients and stable level correction, which the OLS model provides better than the finite-difference ML surrogates.

### C. Temporal splitting is correct and avoids leakage

Evidence:

- `scripts/run_experiments.py:96-99` trains on weeks `<= 82` and tests on later weeks.

Why this is good:

- This is the right way to evaluate forecasting-style pricing models with temporal dependence.

## Overall Assessment by Layer

### Data layer

Status: Mostly sound, but not fully promotion-complete.

- Good: return filtering, zero-price filtering, stable-period restriction, causal merge.
- Weak: coupon-match omission, binary treatment of causal intensity, unweighted weekly price aggregation.

### Segmentation layer

Status: Operationally useful, not behaviorally deep.

- Good: stable K-Means feature set, removal of noisy elasticity feature.
- Weak: clusters group commercial profiles, not substitution sets; representative selection is head-biased.

### Demand estimation layer

Status: Best part of the system.

- Good: own-price vs discount separation, display/mailer controls, seasonality controls, temporal validation, calibration.
- Weak: no dynamic promotion effects, heuristic shrinkage variance, constrained post-processing.

### Simulation layer

Status: Usable for policy experiments, not a full retail market simulator.

- Good: product-level state/action, budget enforcement, stochastic promo shocks.
- Weak: no substitution, no dynamics from prior state, no seasonality path, no inventory/stockpiling.

### Evaluation/reporting layer

Status: Useful but partially inconsistent.

- Good: direct validation, lift validation, heuristic policy comparison.
- Weak: sensitivity is not full-simulator based, `per_period_revenue` bug, stale report tables.

## What I Would Fix First

1. Include `COUPON_MATCH_DISC` in `base_price` and `discount_depth` reconstruction.
2. Repair the stale segmentation table and any other cached report numbers so the report matches the current pipeline exactly.
3. Add a time-varying seasonal demand term to the simulator, since the model already estimates it.
4. Make the transition use prior demand or a reference-price state if the state is meant to be dynamic.
5. Replace binary display/mailer indicators with exposure intensity (`display_pct`, `mailer_pct`) in at least one robustness run.
6. Clarify in the report that the simulator is a reduced-form own-demand response simulator, not a substitution-aware market simulator.
7. Fix `per_period_revenue` to aggregate across Monte Carlo runs rather than keeping only the last trajectory.

## Bottom Line

The current project is substantially better engineered than a notebook-only prototype and the promotion-discounting logic is directionally coherent. But the strongest claim that can be defended today is:

> This is a calibrated, product-level, reduced-form promotion response simulator with budget constraints.

The stronger claim that it “captures the market dynamics” end-to-end is not yet supported, mainly because the simulator omits substitution, dynamic demand carryover, and time-varying seasonal structure, while the report still contains stale internal contradictions.