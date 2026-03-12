#!/usr/bin/env python3
"""Verify the promotional attention effect in the customer choice model."""
import numpy as np
from simulator.config import SimulatorConfig
from simulator.customer import CustomerChoiceModel

config = SimulatorConfig(seed=42)
rng = np.random.default_rng(42)
cm = CustomerChoiceModel(config, rng)

base_prices = np.array([p.base_price for p in config.products])
inventory = np.array([p.initial_inventory for p in config.products])
product_names = [p.name for p in config.products]

print("=" * 70)
print("Promotional Attention Effect Verification")
print("=" * 70)

n_trials = 50
results_no_disc = np.zeros(config.n_products)
results_disc_p0 = np.zeros(config.n_products)

for trial in range(n_trials):
    rng_t = np.random.default_rng(trial)
    cm_t = CustomerChoiceModel(config, rng_t)
    customers = cm_t.generate_customers(100)

    # No discounts
    sold_no, _ = cm_t.purchase_decisions(
        customers, base_prices, inventory.copy(), discounts=np.zeros(config.n_products)
    )
    results_no_disc += sold_no

    # Discount product 0 (Organic Milk) at 30%
    rng_t2 = np.random.default_rng(trial)
    cm_t2 = CustomerChoiceModel(config, rng_t2)
    customers2 = cm_t2.generate_customers(100)
    disc = np.zeros(config.n_products)
    disc[0] = 0.30  # 30% off Organic Milk
    prices_disc = base_prices * (1 - disc)
    sold_disc, _ = cm_t2.purchase_decisions(
        customers2, prices_disc, inventory.copy(), discounts=disc
    )
    results_disc_p0 += sold_disc

results_no_disc /= n_trials
results_disc_p0 /= n_trials

print(f"\nAverage units sold per 100 customers (over {n_trials} trials):\n")
print(f"{'Product':<25} {'No Disc':>10} {'30% P0':>10} {'Change':>10}")
print("-" * 55)
for i in range(config.n_products):
    change = results_disc_p0[i] - results_no_disc[i]
    marker = ""
    if i == 0:
        marker = " <-- DISCOUNTED"
    elif config.products[i].category == config.products[0].category:
        marker = " <-- same category (substitute)"
    print(f"{product_names[i]:<25} {results_no_disc[i]:>10.1f} {results_disc_p0[i]:>10.1f} {change:>+10.1f}{marker}")

print(f"\n{'Total':<25} {results_no_disc.sum():>10.1f} {results_disc_p0.sum():>10.1f} {(results_disc_p0-results_no_disc).sum():>+10.1f}")
print(f"\nPromo lift factor: {results_disc_p0[0] / max(results_no_disc[0], 0.1):.2f}x for discounted product")
print(f"Substitutes (dairy) change: {(results_disc_p0[1]+results_disc_p0[2]) - (results_no_disc[1]+results_no_disc[2]):+.1f}")
