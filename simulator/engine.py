"""
Core simulator engine: orchestrates demand, inventory, and financials for each time step.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

from simulator.config import SimulatorConfig
from simulator.demand import DemandEngine


@dataclass
class StepResult:
    """Result of a single simulation step."""
    day: int
    day_of_week: int
    discounts: np.ndarray
    shelf_prices: np.ndarray
    units_sold: np.ndarray
    revenue: np.ndarray          # per-product revenue
    cost_of_goods: np.ndarray    # per-product COGS
    discount_cost: np.ndarray    # per-product discount expenditure
    margin: np.ndarray           # per-product profit margin
    inventory_after: np.ndarray  # inventory after sales
    transactions: List[dict]
    total_revenue: float
    total_discount_cost: float
    total_margin: float


class RetailSimulator:
    """
    Core retail simulator engine.

    Manages the internal dynamics:
      - Demand generation (customer choice + elasticity)
      - Inventory depletion
      - Financial computations

    This class is used internally by the Gym environment and the data generator.
    """

    def __init__(self, config: Optional[SimulatorConfig] = None):
        self.config = config or SimulatorConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.demand_engine = DemandEngine(self.config, self.rng)

        # Extract price/cost arrays
        self.base_prices = np.array([p.base_price for p in self.config.products])
        self.unit_costs = np.array([p.unit_cost for p in self.config.products])
        self.initial_inventory = np.array([p.initial_inventory for p in self.config.products])

        self.reset()

    def reset(self):
        """Reset simulator to initial state."""
        self.inventory = self.initial_inventory.copy()
        self.day = 0
        self.cumulative_discount_cost = np.zeros(self.config.n_products)
        self.budget_remaining = self.config.total_markdown_budget
        self.total_units_sold = np.zeros(self.config.n_products, dtype=int)

    def step(self, discounts: np.ndarray) -> StepResult:
        """
        Execute one day of the simulation.

        Args:
            discounts: array of shape (n_products,) with discount fractions in [0, 1).

        Returns:
            StepResult with all metrics for this day.
        """
        n = self.config.n_products
        discounts = np.clip(discounts, 0.0, 0.99)

        # Mask discounts for out-of-stock products
        discounts = np.where(self.inventory > 0, discounts, 0.0)

        # Compute day of week
        day_of_week = (self.config.start_day_of_week + self.day) % 7

        # Compute shelf prices
        shelf_prices = self.base_prices * (1.0 - discounts)

        # Simulate demand
        units_sold, transactions = self.demand_engine.simulate_day(
            discounts, self.inventory, day_of_week, day=self.day
        )

        # Cap sales at available inventory
        units_sold = np.minimum(units_sold, self.inventory)

        # --- Financial calculations ---
        revenue = shelf_prices * units_sold
        cost_of_goods = self.unit_costs * units_sold
        discount_cost = (self.base_prices - shelf_prices) * units_sold  # = base_price * discount * qty
        margin = revenue - cost_of_goods

        # --- Update state ---
        self.inventory -= units_sold
        self.cumulative_discount_cost += discount_cost
        self.budget_remaining -= discount_cost.sum()
        self.total_units_sold += units_sold
        self.day += 1

        return StepResult(
            day=self.day - 1,
            day_of_week=day_of_week,
            discounts=discounts.copy(),
            shelf_prices=shelf_prices,
            units_sold=units_sold,
            revenue=revenue,
            cost_of_goods=cost_of_goods,
            discount_cost=discount_cost,
            margin=margin,
            inventory_after=self.inventory.copy(),
            transactions=transactions,
            total_revenue=float(revenue.sum()),
            total_discount_cost=float(discount_cost.sum()),
            total_margin=float(margin.sum()),
        )

    @property
    def done(self) -> bool:
        """Check if the markdown period has ended."""
        return self.day >= self.config.markdown_horizon

    @property
    def time_remaining(self) -> int:
        return self.config.markdown_horizon - self.day
