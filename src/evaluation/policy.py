"""Heuristic policy evaluation for the pricing simulator."""

import numpy as np
from typing import Dict, List, Callable

from ..simulator.product_level import ProductLevelSimulator


def no_discount_policy(N: int) -> Callable:
    """Policy that never discounts."""
    def policy(state):
        return np.zeros(2 * N)
    return policy


def uniform_discount_policy(N: int, depth: float = 0.20) -> Callable:
    """Policy that applies uniform discount to all products."""
    def policy(state):
        action = np.ones(2 * N)
        action[N:] = depth
        return action
    return policy


def target_top_k_policy(
    N: int, scores: np.ndarray, k: int = 30, depth: float = 0.20
) -> Callable:
    """Discount only top-k products by a score (e.g., elasticity, ROI)."""
    top_k = np.argsort(scores)[-k:]
    def policy(state):
        action = np.zeros(2 * N)
        action[top_k] = 1
        action[N + top_k] = depth
        return action
    return policy


def front_load_policy(
    N: int, horizon: int = 12, max_depth: float = 0.30, min_depth: float = 0.05,
    scores: np.ndarray = None, k: int = 30,
) -> Callable:
    """Declining discount schedule on top-k products (by score).

    Discounts only the top-k products by score (default: all), starting
    high and decreasing linearly over the horizon.
    """
    if scores is not None:
        top_k = np.argsort(scores)[-k:]
    else:
        top_k = np.arange(N)

    def policy(state):
        time_remaining = state[-1]
        frac = time_remaining / horizon
        depth = min_depth + (max_depth - min_depth) * frac
        action = np.zeros(2 * N)
        action[top_k] = 1
        action[N + top_k] = depth
        return action
    return policy


class PolicyEvaluator:
    """Evaluate multiple discount policies on the simulator."""

    def __init__(self, simulator: ProductLevelSimulator):
        self.sim = simulator

    def evaluate_policies(
        self, policies: Dict[str, Callable], n_runs: int = 50, seed: int = 42
    ) -> List[Dict]:
        """Run each policy with multiple seeds and collect results.

        Uses Monte Carlo averaging over n_runs to get robust revenue
        estimates with confidence intervals.
        """
        results = []

        for name, policy_fn in policies.items():
            run_revenues = []
            run_demands = []
            run_costs = []
            all_period_revenues = []
            for r in range(n_runs):
                total_rev, history = self.sim.rollout(policy_fn, seed=seed + r)
                run_revenues.append(total_rev)
                run_demands.append(sum(h["total_demand"] for h in history))
                run_costs.append(sum(h["total_discount_cost"] for h in history))
                all_period_revenues.append([h["total_revenue"] for h in history])

            rev_arr = np.array(run_revenues)
            # Average per-period revenue across all MC runs
            mean_period_revenue = np.mean(all_period_revenues, axis=0).tolist()
            results.append({
                "policy": name,
                "total_revenue": rev_arr.mean(),
                "revenue_std": rev_arr.std(),
                "revenue_ci_lo": np.percentile(rev_arr, 2.5),
                "revenue_ci_hi": np.percentile(rev_arr, 97.5),
                "total_demand": np.mean(run_demands),
                "total_discount_cost": np.mean(run_costs),
                "budget_used_pct": (
                    np.mean(run_costs) / self.sim.budget * 100 if self.sim.budget > 0 else 0
                ),
                "per_period_revenue": mean_period_revenue,
            })

        return results

    def build_standard_policies(self) -> Dict[str, Callable]:
        """Build the standard set of heuristic policies for comparison."""
        N = self.sim.N
        pp = self.sim.pp

        policies = {
            "No Discount": no_discount_policy(N),
            "Uniform 10%": uniform_discount_policy(N, 0.10),
            "Uniform 20%": uniform_discount_policy(N, 0.20),
        }

        # Target by elasticity (most negative = most responsive)
        elasticities = pp["elasticity_shrunk"].values
        policies["Target Top-30 (elasticity)"] = target_top_k_policy(
            N, -elasticities, k=30, depth=0.20
        )

        # Target by discount effect
        disc_effects = pp["disc_effect_shrunk"].values
        policies["Target Top-30 (disc effect)"] = target_top_k_policy(
            N, disc_effects, k=30, depth=0.20
        )

        # Front-load (declining schedule on most responsive products)
        policies["Front-Load"] = front_load_policy(
            N, self.sim.horizon, scores=disc_effects, k=30
        )

        # Target by ROI (demand uplift per dollar of discount cost)
        base_demand = pp["base_demand"].values
        base_price = pp["base_price"].values
        roi = disc_effects * base_demand / (base_price * 0.20 + 1e-6)
        policies["Target Top-30 (ROI)"] = target_top_k_policy(
            N, roi, k=30, depth=0.20
        )

        return policies
