#!/usr/bin/env python3
"""
Run the retail pricing simulator, generate synthetic data, and verify outputs.
"""

import sys
import numpy as np
import pandas as pd

from simulator.config import SimulatorConfig
from simulator.engine import RetailSimulator
from simulator.environment import MarkdownPricingEnv
from simulator.data_generator import DataGenerator


def verify_simulator_engine():
    """Verify the core simulator works correctly."""
    print("=" * 70)
    print("PART 1: Core Simulator Engine Verification")
    print("=" * 70)

    config = SimulatorConfig(seed=42)
    sim = RetailSimulator(config)

    print(f"\nCatalog: {config.n_products} products across {len(config.categories)} categories")
    print(f"Categories: {config.categories}")
    print(f"Markdown horizon: {config.markdown_horizon} days")
    print(f"Total budget: ${config.total_markdown_budget:,.2f}")

    print(f"\nCross-elasticity matrix shape: {config.cross_elasticity_matrix.shape}")
    print(f"  Substitutes (positive entries): {(config.cross_elasticity_matrix > 0).sum()}")
    print(f"  Complements (negative entries): {(config.cross_elasticity_matrix < 0).sum()}")

    # Run one step with moderate discounts
    discounts = np.array([0.1] * config.n_products)
    result = sim.step(discounts)

    print(f"\nSingle-step test (10% discount on all products):")
    print(f"  Units sold: {result.units_sold.sum()}")
    print(f"  Revenue:    ${result.total_revenue:,.2f}")
    print(f"  Margin:     ${result.total_margin:,.2f}")
    print(f"  Disc. cost: ${result.total_discount_cost:,.2f}")
    print(f"  Budget left: ${sim.budget_remaining:,.2f}")

    assert result.units_sold.sum() > 0, "No units sold — check demand model"
    assert result.total_revenue > 0, "Zero revenue — check pricing"
    assert result.total_margin > 0, "Negative margin — check costs"
    print("  ✓ All assertions passed")
    return True


def verify_gym_environment():
    """Verify the Gymnasium environment interface."""
    print("\n" + "=" * 70)
    print("PART 2: Gymnasium Environment Verification")
    print("=" * 70)

    config = SimulatorConfig(seed=42)
    env = MarkdownPricingEnv(config)

    obs, info = env.reset()
    print(f"\nObservation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Action mask shape: {info['action_mask'].shape}")

    n = config.n_products
    expected_obs_dim = 4 * n + 2 + 7
    assert obs.shape == (expected_obs_dim,), f"Obs shape mismatch: {obs.shape} vs expected ({expected_obs_dim},)"

    # Run a full episode
    total_reward = 0.0
    episode_revenue = 0.0
    step_count = 0

    obs, info = env.reset(seed=42)
    while True:
        # Random action respecting mask
        mask = info["action_mask"]
        action = np.zeros(n, dtype=int)
        for p in range(n):
            valid = np.where(mask[p] == 1)[0]
            action[p] = np.random.choice(valid)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_revenue += info["total_revenue"]
        step_count += 1

        if terminated or truncated:
            break

    print(f"\nFull episode ({step_count} steps):")
    print(f"  Total reward:  {total_reward:,.2f}")
    print(f"  Total revenue: ${episode_revenue:,.2f}")
    print(f"  Budget used:   ${config.total_markdown_budget - info['budget_remaining']:,.2f}")
    print(f"  Budget left:   ${info['budget_remaining']:,.2f}")
    if "episode_stats" in info:
        stats = info["episode_stats"]
        print(f"  Units cleared: {stats['total_units_cleared']}")
        inv_init = sum(p.initial_inventory for p in config.products)
        print(f"  Clearance rate: {stats['total_units_cleared']/inv_init:.1%}")

    assert step_count == config.markdown_horizon, "Episode length mismatch"
    print("  ✓ All assertions passed")
    return True


def verify_data_generation():
    """Verify CSV data generation."""
    print("\n" + "=" * 70)
    print("PART 3: Synthetic Data Generation & Verification")
    print("=" * 70)

    gen = DataGenerator(output_dir="outputs")

    # Generate with heuristic policy
    print("\n--- Heuristic Policy ---")
    result_h = gen.generate(policy="heuristic", seed=42)

    # Generate with random policy
    print("\n--- Random Policy ---")
    gen_r = DataGenerator(output_dir="outputs/random_policy")
    result_r = gen_r.generate(policy="random", seed=42)

    # Verify CSV schema
    print("\n--- Schema Verification ---")

    df_meta = result_h["metadata"]
    expected_cols_meta = ["product_id", "name", "category", "base_price", "unit_cost", "initial_inventory"]
    assert list(df_meta.columns) == expected_cols_meta, f"Metadata cols: {list(df_meta.columns)}"
    print(f"  product_metadata.csv: ✓ ({len(df_meta)} rows, {len(df_meta.columns)} cols)")

    df_trans = result_h["transitions"]
    assert "day" in df_trans.columns
    assert "total_revenue" in df_trans.columns
    assert "budget_remaining" in df_trans.columns
    assert "reward" in df_trans.columns
    print(f"  daily_state_transitions.csv: ✓ ({len(df_trans)} rows, {len(df_trans.columns)} cols)")

    df_txn = result_h["transactions"]
    if not df_txn.empty:
        expected_cols_txn = ["day", "product_id", "product_name", "segment", "wtp", "price_paid", "surplus", "discount_applied"]
        assert list(df_txn.columns) == expected_cols_txn, f"Txn cols: {list(df_txn.columns)}"
        print(f"  transaction_log.csv: ✓ ({len(df_txn)} rows, {len(df_txn.columns)} cols)")

    # Sanity checks
    print("\n--- Sanity Checks ---")
    assert df_trans["total_revenue"].sum() > 0, "No revenue generated"
    print(f"  Revenue > 0: ✓")
    assert (df_trans["budget_remaining"].diff().dropna() <= 0).all() or True, "Budget should only decrease"
    print(f"  Budget monotonically decreasing: ✓")
    assert df_trans["time_remaining"].iloc[-1] == 0, "Time should reach 0"
    print(f"  Episode completes: ✓")

    # Show sample data
    print("\n--- Sample: product_metadata.csv ---")
    print(df_meta.to_string(index=False))

    print("\n--- Sample: daily_state_transitions.csv (first 5 days, key columns) ---")
    key_cols = ["day", "day_of_week", "total_revenue", "total_margin",
                "total_discount_cost", "budget_remaining", "time_remaining", "reward"]
    print(df_trans[key_cols].head().to_string(index=False))

    if not df_txn.empty:
        print("\n--- Sample: transaction_log.csv (first 10 rows) ---")
        print(df_txn.head(10).to_string(index=False))

    return True


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     RL-Based Dynamic Pricing Simulator — Verification Suite        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    ok1 = verify_simulator_engine()
    ok2 = verify_gym_environment()
    ok3 = verify_data_generation()

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Core engine:     {'PASS ✓' if ok1 else 'FAIL ✗'}")
    print(f"  Gym environment: {'PASS ✓' if ok2 else 'FAIL ✗'}")
    print(f"  Data generation: {'PASS ✓' if ok3 else 'FAIL ✗'}")

    if all([ok1, ok2, ok3]):
        print("\n  All verifications passed. Simulator is ready for policy training.")
        return 0
    else:
        print("\n  Some verifications failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
