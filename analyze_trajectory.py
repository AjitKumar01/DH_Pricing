#!/usr/bin/env python3
"""
Detailed trajectory analysis for the extended 91-day simulator.
Shows weekly decision points, holiday events, and policy comparison.
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

def zero_discount_fn(obs, info, det=True):
    return np.zeros(15, dtype=int)

def random_fn(obs, info, det=False):
    mask = info.get("action_mask", np.ones((15, 5), dtype=np.int8))
    action = np.zeros(15, dtype=int)
    for p in range(15):
        valid = np.where(mask[p] == 1)[0]
        action[p] = np.random.choice(valid)
    return action

def heuristic_fn(obs, info, det=True):
    n = 15
    i_norm = obs[2*n:3*n]
    B_frac = obs[4*n]
    T_frac = obs[4*n + 1]
    mask = info.get("action_mask", np.ones((n, 5), dtype=np.int8))
    action = np.zeros(n, dtype=int)
    for p in range(n):
        if mask[p].sum() <= 1:
            action[p] = 0
            continue
        pressure = 0.5 * (1.0 - T_frac) + 0.5 * i_norm[p]
        tier_idx = min(int(pressure * 5), 4)
        if B_frac < 0.2:
            tier_idx = min(tier_idx, 1)
        action[p] = tier_idx
    return action


# ============================================================================
# Trajectory runner
# ============================================================================

def run_trajectory(env, predict_fn, config, label, seed=42):
    """Run one episode and collect per-MDP-step details."""
    obs, info = env.reset(seed=seed)
    rows = []
    total_reward = 0.0

    for step in range(100):  # max steps safeguard
        action = predict_fn(obs, info, True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        day = info["day"]
        events = config.get_active_events(day)
        rows.append({
            "step": step + 1,
            "day": day,
            "days_in_step": info.get("days_in_step", "?"),
            "revenue": info["total_revenue"],
            "margin": info["total_margin"],
            "disc_cost": info["total_discount_cost"],
            "units_sold": sum(info["units_sold"]),
            "inv_left": sum(info["inventory"]),
            "budget_left": info["budget_remaining"],
            "reward": reward,
            "events": " | ".join(events) if events else "",
        })
        if terminated or truncated:
            break

    df = pd.DataFrame(rows)
    stats = info.get("episode_stats", {})

    print(f"\n{'='*110}")
    print(f" {label} — Weekly Trajectory (seed={seed})")
    print(f"{'='*110}")
    print(df.to_string(index=False))
    print(f"\n  Total reward:     {total_reward:,.1f}")
    print(f"  Total revenue:    ${df['revenue'].sum():,.2f}")
    print(f"  Total margin:     ${df['margin'].sum():,.2f}")
    print(f"  Total disc. cost: ${df['disc_cost'].sum():,.2f}")
    print(f"  Units cleared:    {stats.get('total_units_cleared', 'N/A')} / 7,370")
    print(f"  Budget remaining: ${info['budget_remaining']:,.2f}")

    return df, total_reward


# ============================================================================
# Main
# ============================================================================

def main():
    config = SimulatorConfig(seed=42)

    # Load trained SAC model
    sac_model = SAC.load("outputs/training/sac_model")
    def sac_fn(obs, info, det):
        action, _ = sac_model.predict(obs, deterministic=det)
        return action

    # Environments
    env_base = MarkdownPricingEnv(config)
    env_cont = ContinuousActionWrapper(MarkdownPricingEnv(config))

    policies = [
        ("Zero Discount", env_base, zero_discount_fn),
        ("Heuristic", env_base, heuristic_fn),
        ("SAC (RL Agent)", env_cont, sac_fn),
    ]

    all_results = {}
    for label, env, fn in policies:
        df, reward = run_trajectory(env, fn, config, label, seed=42)
        all_results[label] = {
            "reward": reward,
            "revenue": df["revenue"].sum(),
            "margin": df["margin"].sum(),
            "disc_cost": df["disc_cost"].sum(),
            "units": df["units_sold"].sum(),
        }

    # Summary
    print(f"\n\n{'='*100}")
    print(f" POLICY COMPARISON SUMMARY (91-day / 13-week horizon)")
    print(f"{'='*100}")
    total_inv = sum(p.initial_inventory for p in config.products)
    print(f" {'Policy':<22} {'Reward':>10} {'Revenue':>12} {'Margin':>12} {'Disc Cost':>12} {'Units':>8} {'Clear%':>7}")
    print(f" {'-'*22} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*7}")
    for label, r in all_results.items():
        pct = r["units"] / total_inv * 100
        print(f" {label:<22} {r['reward']:>10,.1f} ${r['revenue']:>10,.2f} ${r['margin']:>10,.2f} ${r['disc_cost']:>10,.2f} {r['units']:>7} {pct:>6.1f}%")


if __name__ == "__main__":
    main()
