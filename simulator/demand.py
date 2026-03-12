"""
Demand and elasticity engine.

Computes effective demand by combining:
  1. Base demand (from customer arrivals + choice model)
  2. Self-elasticity: demand scales with (price / base_price)^ε_own
  3. Cross-elasticity: demand shifts due to discounts on other products
"""

import numpy as np
from typing import Tuple, List

from simulator.customer import CustomerChoiceModel, Customer


class DemandEngine:
    """
    Orchestrates the demand generation pipeline.

    The flow on each step:
      1. Compute shelf prices from base prices and discount vector.
      2. Apply self-elasticity to modulate arrival rate.
      3. Generate customer arrivals and purchase decisions.
      4. Apply cross-elasticity adjustments to realized demand.
    """

    def __init__(self, config, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.choice_model = CustomerChoiceModel(config, rng)
        self.base_prices = np.array([p.base_price for p in config.products])

    def compute_shelf_prices(self, discounts: np.ndarray) -> np.ndarray:
        """Compute shelf prices given discount fractions."""
        return self.base_prices * (1.0 - discounts)

    def self_elasticity_multiplier(self, discounts: np.ndarray) -> np.ndarray:
        """
        Compute demand multiplier from self-elasticity using log-log model.

        demand_multiplier = (price / base_price)^ε = (1 - discount)^ε

        Since ε < 0 (elastic), lower prices yield multipliers > 1.
        """
        eps = self.config.default_self_elasticity
        price_ratio = 1.0 - discounts
        # Clamp to avoid zero-price issues
        price_ratio = np.clip(price_ratio, 0.01, 1.0)
        return np.power(price_ratio, eps)

    def cross_elasticity_adjustment(
        self, base_demand: np.ndarray, discounts: np.ndarray
    ) -> np.ndarray:
        """
        Adjust demand vector using cross-elasticity matrix.

        Δq_i = Σ_j E[i,j] * d_j * base_demand_i

        Where E[i,j] > 0 means j is a substitute for i (discounting j steals from i),
        and E[i,j] < 0 means j is a complement (discounting j boosts i).

        Note: In our convention, E[i,j] > 0 (substitutes) means discounting j
        *cannibalizes* i, so the effect is: demand_i decreases when j is discounted.
        E[i,j] < 0 (complements) means discounting j *boosts* i.
        """
        E = self.config.cross_elasticity_matrix
        # Cross-effect: negative sign because substitute discounting hurts product i
        cross_delta = -E @ discounts * base_demand
        return cross_delta

    def simulate_day(
        self,
        discounts: np.ndarray,
        inventory: np.ndarray,
        day_of_week: int,
        day: int = 0
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Simulate one day of demand.

        Args:
            discounts: discount fractions per product [0, 1).
            inventory: current inventory levels.
            day_of_week: 0=Monday, ..., 6=Sunday.
            day: absolute day index in the markdown period (for event calendar).

        Returns:
            units_sold: array (n_products,) of units sold.
            transactions: list of individual purchase records.
        """
        shelf_prices = self.compute_shelf_prices(discounts)

        # Self-elasticity modulates arrival volume
        demand_mult = self.self_elasticity_multiplier(discounts)
        # Average multiplier scales total arrivals
        avg_mult = np.mean(demand_mult)
        n_arrivals = self.choice_model.generate_arrivals(day_of_week)

        # Apply seasonal event multiplier (Black Friday, Christmas, etc.)
        event_mult = self.config.get_event_multiplier(day)
        n_arrivals = int(n_arrivals * avg_mult * event_mult)
        n_arrivals = max(n_arrivals, 0)

        # Generate customers and simulate purchases
        customers = self.choice_model.generate_customers(n_arrivals)
        units_sold, transactions = self.choice_model.purchase_decisions(
            customers, shelf_prices, inventory, discounts=discounts
        )

        # Apply cross-elasticity adjustments (additive delta)
        cross_delta = self.cross_elasticity_adjustment(
            units_sold.astype(float), discounts
        )
        adjusted = units_sold.astype(float) + cross_delta
        # Floor at 0 and cap at inventory
        adjusted = np.clip(adjusted, 0, inventory).astype(int)

        # If cross-elasticity increased demand, we may need more transactions
        # (simplified: just adjust totals, keep existing transaction log)
        final_sold = adjusted

        return final_sold, transactions
