#!/usr/bin/env python3
"""
Detailed trajectory analysis:
1. Zero discount baseline comparison
2. Seasonality verification (day-of-week effects)
3. Sample trajectory walk-through
"""

import numpy as np
import pandas as pd
from simulator.config import SimulatorConfig
from simulator.environment import MarkdownPricingEnv
from simulator.wrappers import MaskableMarkdownEnv, ContinuousActionWrapper

from stable_baselines3 import SAC
from sb3_contrib import MaskablePPO

# ============================================================================
# 1. Policies
# ============================================================================

def zero_discount_fn(obs, info, deterministic=True):
    """Always apply zero discount."""
    return np.zeros(15, dtype=int)

def random_fn(obs, info, deterministic=False):
    mask = info.get("action_mask", np.ones((15, 5), dtype=np.int8))
    action = np.zeros(15, dtype=int)
    for p in range(15):
        valid = np.where(mask[p] == 1)[0]
        action[p] = np.random.choice(valid)
    return action

def heuristic_fn(obs, info, deterministic=True):
    n = 15
    tiers = [0.0, 0.10, 0.20, 0.30, 0.50]
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
        tier_idx = min(int(pressure * len(tiers)), len(tiers) - 1)
        if B_frac < 0.2:
            tier_idx = min(tier_idx, 1)
        action[p] = tier_idx
    return action


# ============================================================================
# 2. Full trajectory with daily details
# ============================================================================

def run_detailed_trajectory(env, predict_fn, label, seed=42):
    """Run one episode and collect daily details."""
    obs, info = env.reset(seed=seed)
    daily = []
    total_reward = 0.0

    config = env.config if hasattr(env, 'config') else env.unwrapped.config
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for step in range(config.markdown_horizon):
        action = predict_fn(obs, info, True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        dow = info.get('day', step) % 7 if 'day' not in info else (config.start_day_of_week + info['day']) % 7
        daily.append({
            'day': info.get('day', step),
            'dow': day_names[(config.start_day_of_week + step) % 7],
            'revenue': info['total_revenue'],
            'margin': info['total_margin'],
            'disc_cost': info['total_discount_cost'],
            'units_sold': sum(info['units_sold']),
            'inventory_left': sum(info['inventory']),
            'budget_left': info['budget_remaining'],
            'reward': reward,
        })
        if terminated:
            break

    df = pd.DataFrame(daily)
    stats = info.get('episode_stats', {})

    print(f"\n{'='*80}")
    print(f" {label} — Detailed Trajectory (seed={seed})")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"\n  Total reward:     {total_reward:.2f}")
    print(f"  Total revenue:    ${df['revenue'].sum():,.2f}")
    print(f"  Total margin:     ${df['margin'].sum():,.2f}")
    print(f"  Total disc. cost: ${df['disc_cost'].sum():,.2f}")
    print(f"  Units cleared:    {stats.get('total_units_cleared', 'N/A')}")
    print(f"  Budget remaining: ${info['budget_remaining']:,.2f}")

    return df, total_reward


# ============================================================================
# 3. Seasonality analysis
# ============================================================================

def check_seasonality(env, predict_fn, label, n_episodes=20):
    """Run many episodes and check day-of-week demand patterns."""
    config = env.config if hasattr(env, 'config') else env.unwrapped.config
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_sales = {d: [] for d in day_names}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 100)
        for step in range(config.markdown_horizon):
            action = predict_fn(obs, info, True)
            obs, reward, terminated, truncated, info = env.step(action)
            dow_name = day_names[(config.start_day_of_week + step) % 7]
            dow_sales[dow_name].append(sum(info['units_sold']))
            if terminated:
                break

    print(f"\n{'='*60}")
    print(f" Seasonality Check: {label} ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f" {'Day':<6} {'Avg Sales':>10} {'Std':>8} {'Expected λ mult':>15}")
    mults = config.day_of_week_multipliers
    for i, d in enumerate(day_names):
        vals = dow_sales[d]
        if vals:
            print(f" {d:<6} {np.mean(vals):>10.1f} {np.std(vals):>8.1f} {mults[i]:>15.2f}")


# ============================================================================
# Main
# ============================================================================

def main():
    config = SimulatorConfig(seed=42)

    # Load trained models
    ppo_model = MaskablePPO.load("outputs/training/ppo_model")
    sac_model = SAC.load("outputs/training/sac_model")

    def ppo_fn(obs, info, det):
        mask = info.get("action_mask", None)
        if mask is not None:
            action, _ = ppo_model.predict(obs, deterministic=det,
                                           action_masks=mask.flatten().astype(np.int8))
        else:
            action, _ = ppo_model.predict(obs, deterministic=det)
        return action

    def sac_fn(obs, info, det):
        action, _ = sac_model.predict(obs, deterministic=det)
        return action

    # --- Detailed trajectories ---
    env_disc = MarkdownPricingEnv(config)
    env_mask = MaskableMarkdownEnv(config)
    env_cont = ContinuousActionWrapper(MarkdownPricingEnv(config))

    policies = [
        ("Zero Discount", env_disc, zero_discount_fn),
        ("Random", env_disc, random_fn),
        ("Heuristic", env_disc, heuristic_fn),
        ("PPO (Maskable)", env_mask, ppo_fn),
        ("SAC (Continuous)", env_cont, sac_fn),
    ]

    all_results = {}
    for label, env, fn in policies:
        df, reward = run_detailed_trajectory(env, fn, label, seed=42)
        all_results[label] = {
            "reward": reward,
            "revenue": df["revenue"].sum(),
            "margin": df["margin"].sum(),
            "disc_cost": df["disc_cost"].sum(),
            "units": df["units_sold"].sum(),
        }

    # --- Summary comparison ---
    print(f"\n\n{'='*90}")
    print(f" POLICY COMPARISON SUMMARY")
    print(f"{'='*90}")
    print(f" {'Policy':<22} {'Reward':>10} {'Revenue':>12} {'Margin':>12} {'Disc Cost':>12} {'Units':>8}")
    print(f" {'-'*22} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for label, r in all_results.items():
        print(f" {label:<22} {r['reward']:>10.1f} ${r['revenue']:>10,.2f} ${r['margin']:>10,.2f} ${r['disc_cost']:>10,.2f} {r['units']:>7}")

    # --- Seasonality check using zero-discount (cleanest signal) ---
    check_seasonality(MarkdownPricingEnv(config), zero_discount_fn, "Zero Discount", n_episodes=30)


if __name__ == "__main__":
    main()
