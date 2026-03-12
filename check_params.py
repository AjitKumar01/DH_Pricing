#!/usr/bin/env python3
"""Quick parameter check."""
import numpy as np
from simulator.config import SimulatorConfig
from simulator.engine import RetailSimulator

config = SimulatorConfig(seed=42)
sim = RetailSimulator(config)
total_inv = sum(p.initial_inventory for p in config.products)
print(f"Total inventory: {total_inv}")
print(f"Arrivals/day (base): {config.base_daily_arrivals}")
print(f"Horizon: {config.markdown_horizon} days")
print(f"Budget: ${config.total_markdown_budget}")

# Zero discount run
while not sim.done:
    result = sim.step(np.zeros(config.n_products))
total_sold = sim.total_units_sold.sum()
print(f"\nZero discount: {total_sold} sold of {total_inv} ({total_sold/total_inv:.1%} clearance)")
print(f"Remaining: {sim.inventory.sum()} units")

# 20% uniform discount run
sim2 = RetailSimulator(config)
while not sim2.done:
    result = sim2.step(np.full(config.n_products, 0.2))
total_sold2 = sim2.total_units_sold.sum()
print(f"\n20% discount: {total_sold2} sold of {total_inv} ({total_sold2/total_inv:.1%} clearance)")
print(f"Remaining: {sim2.inventory.sum()} units")

# 50% discount run
sim3 = RetailSimulator(config)
while not sim3.done:
    result = sim3.step(np.full(config.n_products, 0.5))
total_sold3 = sim3.total_units_sold.sum()
print(f"\n50% discount: {total_sold3} sold of {total_inv} ({total_sold3/total_inv:.1%} clearance)")
print(f"Remaining: {sim3.inventory.sum()} units")
