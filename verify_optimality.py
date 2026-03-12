#!/usr/bin/env python3
"""
Policy optimality verification.

Determines how close the trained SAC policy is to theoretical bounds by:
1. Computing theoretical upper bounds (perfect-info oracle)
2. Grid-searching constant-discount policies to understand the landscape
3. Analysing SAC's actual per-step discount strategy
4. Multi-seed stability test
"""

import numpy as np
import pandas as pd
from simulator.config import SimulatorConfig
from simulator.environment import MarkdownPricingEnv
from simulator.wrappers import ContinuousActionWrapper
from stable_baselines3 import SAC


# ============================================================================
# Policies
# ============================================================================

def zero_fn(obs, info, det=True):
    return np.zeros(15, dtype=int)

def constant_tier_fn(tier):
    """Factory for constant-discount policies."""
    def fn(obs, info, det=True):
        mask = info.get("action_mask", np.ones((15, 5), dtype=np.int8))
        action = np.full(15, tier, dtype=int)
        for p in range(15):
            if mask[p, tier] != 1:
                action[p] = 0
        return action
    return fn

def heuristic_fn(obs, info, det=True):
    n = 15
    i_norm = obs[2*n:3*n]
    B_frac = obs[4*n]
    T_frac = obs[4*n + 1]
    mask = info.get("action_mask", np.ones((n, 5), dtype=np.int8))
    action = np.zeros(n, dtype=int)
    for p in range(n):
        if mask[p].sum() <= 1:
            continue
        pressure = 0.5 * (1.0 - T_frac) + 0.5 * i_norm[p]
        tier_idx = min(int(pressure * 5), 4)
        if B_frac < 0.2:
            tier_idx = min(tier_idx, 1)
        action[p] = tier_idx
    return action


def run_episode(env, predict_fn, seed=42):
    """Run one episode, return total reward and per-step data."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    total_margin = 0.0
    total_disc_cost = 0.0
    total_units = 0
    steps_data = []

    while True:
        action = predict_fn(obs, info, True)
        obs_next, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_margin += info.get("total_margin", 0)
        total_disc_cost += info.get("total_discount_cost", 0)
        total_units += sum(info.get("units_sold", []))

        # Record per-step discount choices
        if hasattr(action, '__len__'):
            steps_data.append({
                "step": len(steps_data) + 1,
                "day": info.get("day", 0),
                "mean_discount_tier": np.mean(action),
                "max_discount_tier": np.max(action),
                "reward": reward,
                "margin": info.get("total_margin", 0),
                "disc_cost": info.get("total_discount_cost", 0),
                "units": sum(info.get("units_sold", [])),
                "budget_left": info.get("budget_remaining", 0),
            })

        obs = obs_next
        if terminated or truncated:
            break

    stats = info.get("episode_stats", {})
    total_inv = sum(p.initial_inventory for p in env.config.products) if hasattr(env, 'config') else 7370

    return {
        "reward": total_reward,
        "margin": total_margin,
        "disc_cost": total_disc_cost,
        "units": total_units,
        "clearance": stats.get("total_units_cleared", total_units),
        "steps_data": steps_data,
    }


def main():
    config = SimulatorConfig(seed=42)
    total_inv = sum(p.initial_inventory for p in config.products)

    # ========================================================================
    # 1. Theoretical Upper Bound
    # ========================================================================
    print("=" * 90)
    print(" PART 1: THEORETICAL UPPER BOUND")
    print("=" * 90)

    # Calculate max possible margin if every single unit were sold at full price
    max_possible_margin = sum(
        (p.base_price - p.unit_cost) * p.initial_inventory
        for p in config.products
    )
    print(f"  Total inventory:          {total_inv:,} units")
    print(f"  Max possible margin:      ${max_possible_margin:,.2f}")
    print(f"    (every unit sold at full price, 0% discount, all inventory cleared)")
    print(f"  This is unachievable because demand at zero discount only clears ~65% of stock.\n")

    # ========================================================================
    # 2. Grid Search: Constant-Discount Policies
    # ========================================================================
    print("=" * 90)
    print(" PART 2: CONSTANT-DISCOUNT POLICY LANDSCAPE")
    print("=" * 90)

    tier_labels = ["0%", "10%", "20%", "30%", "50%"]
    env_base = MarkdownPricingEnv(config)

    grid_results = {}
    for tier_idx in range(5):
        fn = constant_tier_fn(tier_idx)
        result = run_episode(env_base, fn, seed=42)
        grid_results[tier_labels[tier_idx]] = result
        pct = result["clearance"] / total_inv * 100
        print(f"  Constant {tier_labels[tier_idx]:>3s} discount: "
              f"Margin=${result['margin']:>10,.2f}  "
              f"Disc=${result['disc_cost']:>9,.2f}  "
              f"Units={result['clearance']:>5,}  "
              f"Clear={pct:>5.1f}%  "
              f"Reward={result['reward']:>10,.1f}")

    # ========================================================================
    # 3. SAC Policy Analysis
    # ========================================================================
    print("\n" + "=" * 90)
    print(" PART 3: SAC POLICY TRAJECTORY ANALYSIS")
    print("=" * 90)

    sac_model = SAC.load("outputs/training/sac_model")
    env_cont = ContinuousActionWrapper(MarkdownPricingEnv(config))

    def sac_fn(obs, info, det):
        action, _ = sac_model.predict(obs, deterministic=det)
        return action

    sac_result = run_episode(env_cont, sac_fn, seed=42)
    heur_result = run_episode(env_base, heuristic_fn, seed=42)
    zero_result = run_episode(env_base, zero_fn, seed=42)

    print("\n  Per-step SAC decisions:")
    print(f"  {'Step':>4}  {'Day':>4}  {'AvgTier':>8}  {'MaxTier':>8}  "
          f"{'Margin':>10}  {'DiscCost':>9}  {'Units':>6}  {'BudgetLeft':>11}")
    for s in sac_result["steps_data"]:
        print(f"  {s['step']:>4}  {s['day']:>4}  {s['mean_discount_tier']:>8.2f}  "
              f"{s['max_discount_tier']:>8}  ${s['margin']:>9,.2f}  "
              f"${s['disc_cost']:>8,.2f}  {s['units']:>5}  ${s['budget_left']:>10,.2f}")

    # ========================================================================
    # 4. Optimality Gap Calculation
    # ========================================================================
    print("\n" + "=" * 90)
    print(" PART 4: OPTIMALITY GAP ANALYSIS")
    print("=" * 90)

    # Best constant-discount policy
    best_const_name = max(grid_results, key=lambda k: grid_results[k]["reward"])
    best_const = grid_results[best_const_name]

    print(f"\n  Best constant-discount policy: {best_const_name} → Reward={best_const['reward']:,.1f}")
    print(f"  Heuristic policy:             Reward={heur_result['reward']:,.1f}")
    print(f"  SAC policy:                   Reward={sac_result['reward']:,.1f}")
    print(f"  Theoretical max margin:       ${max_possible_margin:,.2f}")

    sac_vs_const = (sac_result['reward'] - best_const['reward']) / abs(best_const['reward']) * 100
    sac_vs_heur = (sac_result['reward'] - heur_result['reward']) / abs(heur_result['reward']) * 100
    margin_efficiency = sac_result['margin'] / max_possible_margin * 100

    print(f"\n  SAC vs best constant:   {sac_vs_const:>+.1f}%")
    print(f"  SAC vs heuristic:       {sac_vs_heur:>+.1f}%")
    print(f"  SAC margin efficiency:  {margin_efficiency:.1f}% of theoretical max")
    print(f"  SAC disc cost:          ${sac_result['disc_cost']:,.2f} = "
          f"{sac_result['disc_cost']/config.total_markdown_budget*100:.1f}% of budget")
    print(f"  SAC clearance:          {sac_result['clearance']:,} / {total_inv:,} = "
          f"{sac_result['clearance']/total_inv*100:.1f}%")

    # ========================================================================
    # 5. Multi-Seed Stability
    # ========================================================================
    print("\n" + "=" * 90)
    print(" PART 5: MULTI-SEED STABILITY (SAC)")
    print("=" * 90)

    seed_rewards = []
    seed_margins = []
    seed_dcs = []
    seed_clearances = []
    for seed in range(1, 21):
        r = run_episode(env_cont, sac_fn, seed=seed)
        seed_rewards.append(r["reward"])
        seed_margins.append(r["margin"])
        seed_dcs.append(r["disc_cost"])
        seed_clearances.append(r["clearance"])

    print(f"  Over 20 episodes:")
    print(f"    Reward:    {np.mean(seed_rewards):>10,.1f} ± {np.std(seed_rewards):>8,.1f}")
    print(f"    Margin:    ${np.mean(seed_margins):>9,.2f} ± ${np.std(seed_margins):>7,.2f}")
    print(f"    Disc Cost: ${np.mean(seed_dcs):>9,.2f} ± ${np.std(seed_dcs):>7,.2f}")
    print(f"    Clearance: {np.mean(seed_clearances):>9,.1f} ± {np.std(seed_clearances):>7,.1f}")
    print(f"    Clearance%: {np.mean(seed_clearances)/total_inv*100:.1f}%")


if __name__ == "__main__":
    main()
