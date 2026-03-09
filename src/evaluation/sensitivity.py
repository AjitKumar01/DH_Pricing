"""Per-product sensitivity analysis."""

import numpy as np
import pandas as pd
from ..simulator.product_level import ProductLevelSimulator


class SensitivityAnalyzer:
    """Sweep discount depths per product to find optimal discounts."""

    def __init__(self, simulator: ProductLevelSimulator):
        self.sim = simulator

    def per_product_sweep(
        self, discount_range: np.ndarray = None, n_runs: int = 20
    ) -> pd.DataFrame:
        """For each product, sweep discount 0-45% and compute expected revenue.

        Uses simulator rollouts with a single-product discount policy to
        capture stochastic display/mailer deviations and budget interaction.

        Returns DataFrame with PRODUCT_ID, optimal_discount, base_revenue,
        optimal_revenue, uplift_pct.
        """
        if discount_range is None:
            discount_range = np.arange(0, 0.50, 0.05)

        results = []
        N = self.sim.N
        pp = self.sim.pp

        for i in range(N):
            pid = pp.iloc[i]["PRODUCT_ID"]
            seg = pp.iloc[i]["segment"]
            base_price = self.sim._base_prices[i]
            intercept = self.sim._intercepts[i]
            elast = self.sim._elasticities[i]
            disc_eff = self.sim._disc_effects[i]
            log_base = np.log(max(base_price, 0.01))

            revenues = []
            for d in discount_range:
                # Monte Carlo over noise seeds for robustness
                rev_runs = []
                for run in range(n_runs):
                    # Use the simulator's demand computation for one product
                    sel = np.zeros(N)
                    disc = np.zeros(N)
                    if d > 0:
                        sel[i] = 1.0
                        disc[i] = d
                    self.sim.reset(seed=42 + run)
                    demand = self.sim._compute_demand(sel, disc)
                    q = demand[i]
                    rev = base_price * (1 - d) * q
                    rev_runs.append(rev)
                revenues.append(np.mean(rev_runs))

            revenues = np.array(revenues)
            best_idx = np.argmax(revenues)
            base_rev = revenues[0]
            opt_rev = revenues[best_idx]
            uplift = (opt_rev / base_rev - 1) * 100 if base_rev > 0 else 0

            results.append({
                "PRODUCT_ID": pid,
                "segment": seg,
                "base_revenue": base_rev,
                "optimal_discount": discount_range[best_idx],
                "optimal_revenue": opt_rev,
                "uplift_pct": uplift,
                "benefits_from_discount": best_idx > 0,
            })

        return pd.DataFrame(results)

    def segment_summary(self, sweep_results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sensitivity results by segment."""
        return (
            sweep_results.groupby("segment")
            .agg(
                n_products=("PRODUCT_ID", "count"),
                pct_benefit=("benefits_from_discount", "mean"),
                mean_optimal_discount=("optimal_discount", "mean"),
                mean_uplift_pct=("uplift_pct", "mean"),
                total_base_revenue=("base_revenue", "sum"),
                total_optimal_revenue=("optimal_revenue", "sum"),
            )
            .reset_index()
        )
