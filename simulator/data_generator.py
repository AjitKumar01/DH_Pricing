"""
Synthetic data generation: runs the simulator with a heuristic policy
and exports CSV datasets for offline pre-training.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional

from simulator.config import SimulatorConfig
from simulator.engine import RetailSimulator


class DataGenerator:
    """
    Runs the simulator under a specified policy and exports:
      - product_metadata.csv
      - daily_state_transitions.csv
      - transaction_log.csv
    """

    def __init__(self, config: Optional[SimulatorConfig] = None,
                 output_dir: str = "outputs"):
        self.config = config or SimulatorConfig()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def random_policy(self, inventory: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Random discount policy: pick a random allowed tier for each product."""
        n = self.config.n_products
        tiers = self.config.allowed_discounts
        indices = rng.integers(0, len(tiers), size=n)
        discounts = np.array([tiers[i] for i in indices])
        # Zero-out discounts for out-of-stock
        discounts = np.where(inventory > 0, discounts, 0.0)
        return discounts

    def heuristic_policy(self, inventory: np.ndarray, day: int,
                         budget_remaining: float) -> np.ndarray:
        """
        Simple heuristic: apply moderate discounts weighted by remaining
        inventory and time pressure.
        """
        n = self.config.n_products
        horizon = self.config.markdown_horizon
        tiers = np.array(self.config.allowed_discounts)
        initial_inv = np.array([p.initial_inventory for p in self.config.products])

        # Time pressure: increases as we approach end
        time_pressure = (day + 1) / horizon  # 0→1

        # Inventory pressure: high if lots of stock remains
        inv_frac = inventory / (initial_inv + 1e-8)

        # For each product, pick a discount tier based on pressure
        discounts = np.zeros(n)
        for p in range(n):
            if inventory[p] <= 0:
                discounts[p] = 0.0
                continue
            pressure = 0.5 * time_pressure + 0.5 * inv_frac[p]
            # Map pressure to a tier index
            tier_idx = min(int(pressure * len(tiers)), len(tiers) - 1)

            # Budget check: don't overspend
            est_cost = self.config.products[p].base_price * tiers[tier_idx] * 5
            if budget_remaining - est_cost < 0:
                tier_idx = max(tier_idx - 1, 0)

            discounts[p] = tiers[tier_idx]

        return discounts

    def generate(self, policy: str = "heuristic", seed: Optional[int] = None) -> dict:
        """
        Run a full episode and export CSV files.

        Args:
            policy: "random" or "heuristic"
            seed: random seed override

        Returns:
            Dictionary with DataFrames: 'metadata', 'transitions', 'transactions'
        """
        if seed is not None:
            self.config.seed = seed
        sim = RetailSimulator(self.config)
        rng = np.random.default_rng(self.config.seed)

        # ===== Product Metadata =====
        metadata_rows = []
        for p in self.config.products:
            metadata_rows.append({
                "product_id": p.product_id,
                "name": p.name,
                "category": p.category,
                "base_price": p.base_price,
                "unit_cost": p.unit_cost,
                "initial_inventory": p.initial_inventory,
            })
        df_metadata = pd.DataFrame(metadata_rows)

        # ===== Run episode =====
        transition_rows = []
        all_transactions = []
        n = self.config.n_products

        prev_discounts = np.zeros(n)
        prev_demand = np.zeros(n)

        while not sim.done:
            day = sim.day
            dow = (self.config.start_day_of_week + day) % 7

            # Current state BEFORE action
            state_before = {
                "d_prev": prev_discounts.tolist(),
                "q_prev": prev_demand.tolist(),
                "inventory": sim.inventory.tolist(),
                "cumul_discount_cost": sim.cumulative_discount_cost.tolist(),
                "budget_remaining": sim.budget_remaining,
                "time_remaining": sim.time_remaining,
                "day_of_week": dow,
            }

            # Select action
            if policy == "random":
                discounts = self.random_policy(sim.inventory, rng)
            else:
                discounts = self.heuristic_policy(
                    sim.inventory, day, sim.budget_remaining
                )

            # Step
            result = sim.step(discounts)

            # State AFTER action (next state)
            dow_next = (self.config.start_day_of_week + sim.day) % 7
            state_after = {
                "inventory_next": sim.inventory.tolist(),
                "cumul_discount_cost_next": sim.cumulative_discount_cost.tolist(),
                "budget_remaining_next": sim.budget_remaining,
                "time_remaining_next": sim.time_remaining,
                "day_of_week_next": dow_next,
            }

            # Reward (simple margin-based for data generation)
            reward = result.total_margin

            # Log transition
            transition_rows.append({
                "day": day,
                "day_of_week": dow,
                **{f"discount_p{i}": float(discounts[i]) for i in range(n)},
                **{f"units_sold_p{i}": int(result.units_sold[i]) for i in range(n)},
                **{f"inventory_p{i}": int(sim.inventory[i]) for i in range(n)},
                **{f"revenue_p{i}": float(result.revenue[i]) for i in range(n)},
                **{f"discount_cost_p{i}": float(result.discount_cost[i]) for i in range(n)},
                "total_revenue": result.total_revenue,
                "total_margin": result.total_margin,
                "total_discount_cost": result.total_discount_cost,
                "budget_remaining": sim.budget_remaining,
                "time_remaining": sim.time_remaining,
                "reward": reward,
            })

            # Log transactions
            for tx in result.transactions:
                all_transactions.append({
                    "day": day,
                    "product_id": tx["product_id"],
                    "product_name": self.config.products[tx["product_id"]].name,
                    "segment": self.config.segment_names[tx["segment"]],
                    "wtp": tx["wtp"],
                    "price_paid": tx["price_paid"],
                    "surplus": tx["surplus"],
                    "discount_applied": float(discounts[tx["product_id"]]),
                })

            prev_discounts = discounts.copy()
            prev_demand = result.units_sold.astype(float).copy()

        df_transitions = pd.DataFrame(transition_rows)
        df_transactions = pd.DataFrame(all_transactions) if all_transactions else pd.DataFrame()

        # ===== Save CSVs =====
        df_metadata.to_csv(os.path.join(self.output_dir, "product_metadata.csv"), index=False)
        df_transitions.to_csv(os.path.join(self.output_dir, "daily_state_transitions.csv"), index=False)
        if not df_transactions.empty:
            df_transactions.to_csv(os.path.join(self.output_dir, "transaction_log.csv"), index=False)

        print(f"Generated data in '{self.output_dir}/':")
        print(f"  product_metadata.csv      : {len(df_metadata)} products")
        print(f"  daily_state_transitions.csv: {len(df_transitions)} days")
        print(f"  transaction_log.csv        : {len(df_transactions)} transactions")
        print(f"\nEpisode summary:")
        print(f"  Total revenue   : ${df_transitions['total_revenue'].sum():,.2f}")
        print(f"  Total margin    : ${df_transitions['total_margin'].sum():,.2f}")
        print(f"  Total disc. cost: ${df_transitions['total_discount_cost'].sum():,.2f}")
        print(f"  Budget remaining: ${sim.budget_remaining:,.2f}")
        print(f"  Units cleared   : {sim.total_units_sold.sum()}")
        inv_init = sum(p.initial_inventory for p in self.config.products)
        print(f"  Clearance rate  : {sim.total_units_sold.sum()/inv_init:.1%}")

        return {
            "metadata": df_metadata,
            "transitions": df_transitions,
            "transactions": df_transactions,
        }
