#!/usr/bin/env python3
"""
Trajectory analysis for the Dunnhumby-calibrated SAC policy.

Answers: "Is the learned policy rational and near-optimal?"

Tests:
1. Per-week discount strategy: Does SAC increase discounts as time runs out 
   and inventory remains high? (rational temporal behavior)
2. Per-product discrimination: Does SAC discount high-elasticity products 
   more aggressively than low-elasticity ones? (exploiting elasticity)
3. Budget utilization: Does SAC pace its budget smoothly or waste it early?
4. Inventory-conditional discounting: Does SAC discount products with high 
   remaining inventory more than those nearly cleared?
5. Counterfactual analysis: Would locally perturbing SAC's actions improve 
   reward? (local optimality check)
6. Multi-seed stability: Is performance stable across random seeds?
"""

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dunnhumby.train_dunnhumby import (
    DunnhumbyConfig, DunnhumbyPricingEnv, DunnhumbyContinuousWrapper
)
from stable_baselines3 import SAC


def load_env_and_model():
    config = DunnhumbyConfig()
    env = DunnhumbyPricingEnv(config)
    cont_env = DunnhumbyContinuousWrapper(env)
    model = SAC.load("dunnhumby/outputs/training/sac_dunnhumby", env=cont_env)
    return config, env, cont_env, model


def heuristic_action(obs, info, n):
    i_norm = obs[2*n:3*n]
    B_frac = obs[4*n]
    T_frac = obs[4*n + 1]
    mask = info.get("action_mask", np.ones((n, 5), dtype=np.int8))
    action = np.zeros(n, dtype=int)
    for p in range(n):
        if mask[p].sum() <= 1:
            continue
        pressure = 0.5 * (1.0 - T_frac) + 0.5 * i_norm[p]
        tier = min(int(pressure * 5), 4)
        if B_frac < 0.2:
            tier = min(tier, 1)
        action[p] = tier
    return action


def run_trajectory(env, cont_env, model, config, policy="sac", seed=42):
    """Run one episode, record per-step details."""
    obs, info = env.reset(seed=seed)
    n = config.n_products
    tiers = np.array(config.allowed_discounts)
    steps = []

    total_reward = 0.0
    for step_i in range(config.markdown_horizon // config.decision_frequency + 1):
        # Get action
        if policy == "sac":
            cont_obs = obs.copy()
            action_cont, _ = model.predict(cont_obs, deterministic=True)
            action_cont = np.clip(action_cont, 0, 1)
            action = np.array([np.argmin(np.abs(tiers - a)) for a in action_cont])
        elif policy == "heuristic":
            action = heuristic_action(obs, info, n)
        elif policy == "zero":
            action = np.zeros(n, dtype=int)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Extract state before step
        i_norm = obs[2*n:3*n].copy()
        B_frac = float(obs[4*n])
        T_frac = float(obs[4*n + 1])
        discounts = tiers[action]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        steps.append({
            "week": step_i + 1,
            "day": info["day"],
            "T_frac_before": T_frac,
            "B_frac_before": B_frac,
            "mean_discount": float(discounts.mean()),
            "max_discount": float(discounts.max()),
            "min_discount": float(discounts.min()),
            "mean_inv_norm": float(i_norm.mean()),
            "action_tiers": action.copy(),
            "discounts": discounts.copy(),
            "inv_norm": i_norm.copy(),
            "reward": reward,
            "margin": info["total_margin"],
            "disc_cost": info["total_discount_cost"],
            "units_sold": int(info["units_sold"].sum()),
            "inv_remaining": float(info["inventory"].sum()),
        })

        if terminated or truncated:
            break

    return steps, total_reward


def test_temporal_rationality(sac_steps):
    """Test 1: SAC should increase discounts as time runs out."""
    print("\n" + "="*70)
    print("TEST 1: TEMPORAL RATIONALITY — Does SAC increase discounts over time?")
    print("="*70)

    weeks = [s["week"] for s in sac_steps]
    mean_discs = [s["mean_discount"] for s in sac_steps]

    # Split into early (first third) vs late (last third)
    n = len(weeks)
    early = mean_discs[:n//3]
    late = mean_discs[-(n//3):]
    
    early_mean = np.mean(early)
    late_mean = np.mean(late)
    
    print(f"  Early weeks (1-{n//3}) mean discount: {early_mean:.3f}")
    print(f"  Late weeks ({n - n//3 + 1}-{n}) mean discount:  {late_mean:.3f}")
    print(f"  Increase: {late_mean - early_mean:+.3f}")
    
    # Also show week-by-week
    print(f"\n  Week-by-week mean discount:")
    for s in sac_steps:
        bar = "█" * int(s["mean_discount"] * 100)
        print(f"    Week {s['week']:2d}: {s['mean_discount']:.3f} {bar}")
    
    passed = late_mean > early_mean
    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'} — SAC {'does' if passed else 'does NOT'} "
          f"increase discounts as markdown period progresses")
    return passed


def test_elasticity_exploitation(sac_steps, config):
    """Test 2: SAC should discount high-|ε| products more."""
    print("\n" + "="*70)
    print("TEST 2: ELASTICITY EXPLOITATION — Discounting elastic products more?")
    print("="*70)
    
    n = config.n_products
    elasticities = np.array([p.get('self_elasticity', -2.0) for p in config.products])
    abs_elast = np.abs(elasticities)
    
    # Average discount per product across all weeks
    all_discounts = np.zeros(n)
    for s in sac_steps:
        all_discounts += s["discounts"]
    all_discounts /= len(sac_steps)
    
    # Correlation between |elasticity| and avg discount
    corr = np.corrcoef(abs_elast, all_discounts)[0, 1]
    
    # Split into high-elasticity vs low-elasticity
    median_elast = np.median(abs_elast)
    high_mask = abs_elast >= median_elast
    low_mask = ~high_mask
    
    high_disc = all_discounts[high_mask].mean()
    low_disc = all_discounts[low_mask].mean()
    
    print(f"  Median |ε| threshold: {median_elast:.2f}")
    print(f"  High |ε| products (n={high_mask.sum()}): mean discount = {high_disc:.3f}")
    print(f"  Low |ε| products  (n={low_mask.sum()}): mean discount = {low_disc:.3f}")
    print(f"  Correlation(|ε|, avg discount): r = {corr:.3f}")
    
    # Show top-5 and bottom-5
    sorted_idx = np.argsort(abs_elast)
    print(f"\n  Bottom 5 (least elastic) → discount:")
    for i in sorted_idx[:5]:
        name = config.products[i].get('commodity', f'Product {i}')
        print(f"    {name[:30]:30s}  |ε|={abs_elast[i]:.2f}  avg_disc={all_discounts[i]:.3f}")
    print(f"\n  Top 5 (most elastic) → discount:")
    for i in sorted_idx[-5:]:
        name = config.products[i].get('commodity', f'Product {i}')
        print(f"    {name[:30]:30s}  |ε|={abs_elast[i]:.2f}  avg_disc={all_discounts[i]:.3f}")
    
    passed = corr > 0
    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'} — Correlation r={corr:.3f} "
          f"({'positive — SAC exploits elasticity' if passed else 'negative — SAC ignores elasticity'})")
    return passed, corr


def test_budget_pacing(sac_steps, config):
    """Test 3: SAC should use budget smoothly, not front-load or waste it."""
    print("\n" + "="*70)
    print("TEST 3: BUDGET PACING — Smooth budget utilization?")
    print("="*70)
    
    total_budget = config.total_markdown_budget
    cum_spend = np.cumsum([s["disc_cost"] for s in sac_steps])
    n = len(sac_steps)
    
    # Ideal pacing: linear spend
    ideal_cum = np.linspace(total_budget / n, total_budget, n)
    
    # Pace fraction at each step
    pace_frac = cum_spend / total_budget
    
    print(f"  Total budget: ${total_budget:,.0f}")
    print(f"  Total spent:  ${cum_spend[-1]:,.0f} ({cum_spend[-1]/total_budget*100:.1f}%)")
    print(f"\n  Weekly budget utilization:")
    for i, s in enumerate(sac_steps):
        spent_pct = cum_spend[i] / total_budget * 100
        ideal_pct = (i + 1) / n * 100
        bar = "█" * int(spent_pct / 2)
        print(f"    Week {s['week']:2d}: ${s['disc_cost']:8,.0f}  "
              f"cumul {spent_pct:5.1f}% (ideal: {ideal_pct:5.1f}%) {bar}")
    
    # Budget efficiency: did it use the budget or leave money on table?
    utilization = cum_spend[-1] / total_budget
    passed = utilization > 0.85
    print(f"\n  Budget utilization: {utilization*100:.1f}%")
    print(f"  RESULT: {'PASS ✓' if passed else 'FAIL ✗'} — "
          f"{'Efficient' if passed else 'Inefficient'} budget usage (>{85}% threshold)")
    return passed


def test_inventory_conditional(sac_steps, config):
    """Test 4: SAC should discount products with more remaining inventory."""
    print("\n" + "="*70)
    print("TEST 4: INVENTORY-CONDITIONAL — Discounting high-stock items more?")
    print("="*70)
    
    correlations = []
    for s in sac_steps:
        inv = s["inv_norm"]
        disc = s["discounts"]
        # Only look at products with some inventory
        mask = inv > 0.01
        if mask.sum() > 5:
            corr = np.corrcoef(inv[mask], disc[mask])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    mean_corr = np.mean(correlations)
    
    print(f"  Mean correlation(inventory_norm, discount) across weeks: r = {mean_corr:.3f}")
    print(f"  Per-week correlations:")
    for i, c in enumerate(correlations):
        print(f"    Week {i+1:2d}: r = {c:+.3f}")
    
    passed = mean_corr > 0
    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'} — SAC {'does' if passed else 'does NOT'} "
          f"discount high-inventory products more")
    return passed, mean_corr


def test_local_optimality(env, cont_env, model, config, sac_steps, seed=42):
    """Test 5: Perturb SAC actions and check if reward decreases."""
    print("\n" + "="*70)
    print("TEST 5: LOCAL OPTIMALITY — Do perturbations hurt performance?")
    print("="*70)
    
    n = config.n_products
    tiers = np.array(config.allowed_discounts)
    sac_reward = sum(s["reward"] for s in sac_steps)
    
    perturbation_results = []
    
    # Try several perturbation strategies
    perturbations = {
        "+1 tier (all products)": lambda a: np.clip(a + 1, 0, len(tiers) - 1),
        "-1 tier (all products)": lambda a: np.clip(a - 1, 0, len(tiers) - 1),
        "+1 tier (random 50%)": lambda a: _perturb_random(a, +1, 0.5, len(tiers)),
        "-1 tier (random 50%)": lambda a: _perturb_random(a, -1, 0.5, len(tiers)),
        "Zero discount (all)": lambda a: np.zeros_like(a),
        "Max discount (all)": lambda a: np.full_like(a, len(tiers) - 1),
    }
    
    for name, perturb_fn in perturbations.items():
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        
        for step_i in range(config.markdown_horizon // config.decision_frequency + 1):
            # Get SAC's action
            cont_obs = obs.copy()
            action_cont, _ = model.predict(cont_obs, deterministic=True)
            action_cont = np.clip(action_cont, 0, 1)
            sac_action = np.array([np.argmin(np.abs(tiers - a)) for a in action_cont])
            
            # Apply perturbation
            action = perturb_fn(sac_action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        delta = total_reward - sac_reward
        perturbation_results.append((name, total_reward, delta))
        print(f"  {name:35s}: reward = {total_reward:10,.0f}  (Δ = {delta:+8,.0f})")
    
    # Count how many perturbations improve on SAC
    improvements = sum(1 for _, _, d in perturbation_results if d > 0)
    
    print(f"\n  SAC reward:     {sac_reward:10,.0f}")
    print(f"  Perturbations that improve: {improvements}/{len(perturbation_results)}")
    
    passed = improvements <= 1  # At most 1 perturbation should improve
    print(f"\n  RESULT: {'PASS ✓' if passed else 'WARN ⚠'} — "
          f"{'No' if improvements == 0 else f'{improvements}'} perturbation(s) improve on SAC")
    return passed, improvements


def _perturb_random(action, delta, frac, n_tiers):
    a = action.copy()
    mask = np.random.random(len(a)) < frac
    a[mask] = np.clip(a[mask] + delta, 0, n_tiers - 1)
    return a


def test_multi_seed_stability(env, cont_env, model, config, n_seeds=10):
    """Test 6: Performance shouldn't vary wildly across seeds."""
    print("\n" + "="*70)
    print("TEST 6: MULTI-SEED STABILITY — Consistent across random seeds?")
    print("="*70)
    
    n = config.n_products
    tiers = np.array(config.allowed_discounts)
    
    sac_rewards = []
    heur_rewards = []
    clearances_sac = []
    clearances_heur = []
    
    for seed in range(n_seeds):
        # SAC
        obs, info = env.reset(seed=seed * 100 + 7)
        total_reward = 0.0
        for _ in range(config.markdown_horizon // config.decision_frequency + 1):
            action_cont, _ = model.predict(obs, deterministic=True)
            action_cont = np.clip(action_cont, 0, 1)
            action = np.array([np.argmin(np.abs(tiers - a)) for a in action_cont])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        total_inv = sum(p['initial_inventory'] for p in config.products)
        clearance = (total_inv - info["inventory"].sum()) / total_inv
        sac_rewards.append(total_reward)
        clearances_sac.append(clearance)
        
        # Heuristic
        obs, info = env.reset(seed=seed * 100 + 7)
        total_reward = 0.0
        for _ in range(config.markdown_horizon // config.decision_frequency + 1):
            action = heuristic_action(obs, info, n)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        clearance = (total_inv - info["inventory"].sum()) / total_inv
        heur_rewards.append(total_reward)
        clearances_heur.append(clearance)
    
    sac_mean, sac_std = np.mean(sac_rewards), np.std(sac_rewards)
    heur_mean, heur_std = np.mean(heur_rewards), np.std(heur_rewards)
    
    print(f"  SAC:       {sac_mean:10,.0f} ± {sac_std:6,.0f}  "
          f"(clearance: {np.mean(clearances_sac)*100:.1f}% ± {np.std(clearances_sac)*100:.1f}%)")
    print(f"  Heuristic: {heur_mean:10,.0f} ± {heur_std:6,.0f}  "
          f"(clearance: {np.mean(clearances_heur)*100:.1f}% ± {np.std(clearances_heur)*100:.1f}%)")
    print(f"  SAC / Heuristic: {sac_mean / heur_mean * 100:.1f}%")
    
    # Check: SAC beats heuristic in how many seeds?
    wins = sum(s > h for s, h in zip(sac_rewards, heur_rewards))
    print(f"\n  SAC wins in {wins}/{n_seeds} seeds")
    
    # CV (coefficient of variation) should be low
    cv = sac_std / sac_mean
    print(f"  SAC coefficient of variation: {cv:.4f} ({cv*100:.2f}%)")
    
    passed = wins >= n_seeds * 0.8 and cv < 0.05
    print(f"\n  RESULT: {'PASS ✓' if passed else 'WARN ⚠'} — "
          f"SAC wins {wins}/{n_seeds} seeds, CV={cv*100:.2f}%")
    return passed, wins, cv


def compare_trajectories(sac_steps, heur_steps):
    """Side-by-side trajectory comparison."""
    print("\n" + "="*70)
    print("TRAJECTORY COMPARISON: SAC vs HEURISTIC (week by week)")
    print("="*70)
    
    print(f"\n  {'Week':>4s}  {'SAC disc':>8s}  {'Heur disc':>9s}  "
          f"{'SAC sold':>8s}  {'Heur sold':>9s}  "
          f"{'SAC rew':>8s}  {'Heur rew':>8s}")
    print("  " + "-" * 62)
    
    for s, h in zip(sac_steps, heur_steps):
        print(f"  {s['week']:4d}  "
              f"{s['mean_discount']:8.3f}  {h['mean_discount']:9.3f}  "
              f"{s['units_sold']:8d}  {h['units_sold']:9d}  "
              f"{s['reward']:8,.0f}  {h['reward']:8,.0f}")
    
    sac_total = sum(s["reward"] for s in sac_steps)
    heur_total = sum(s["reward"] for s in heur_steps)
    print("  " + "-" * 62)
    print(f"  {'TOTAL':>4s}  {'':8s}  {'':9s}  "
          f"{sum(s['units_sold'] for s in sac_steps):8d}  "
          f"{sum(s['units_sold'] for s in heur_steps):9d}  "
          f"{sac_total:8,.0f}  {heur_total:8,.0f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRAJECTORY ANALYSIS: Dunnhumby SAC Policy Optimality Verification")
    print("=" * 70)
    
    config, env, cont_env, model = load_env_and_model()
    print(f"  Products: {config.n_products}, Horizon: {config.markdown_horizon} days, "
          f"Budget: ${config.total_markdown_budget:,.0f}")
    
    # Run SAC and heuristic trajectories
    sac_steps, sac_reward = run_trajectory(env, cont_env, model, config, "sac", seed=42)
    heur_steps, heur_reward = run_trajectory(env, cont_env, model, config, "heuristic", seed=42)
    
    print(f"\n  SAC total reward:       {sac_reward:10,.0f}")
    print(f"  Heuristic total reward: {heur_reward:10,.0f}")
    print(f"  SAC advantage:          {sac_reward - heur_reward:+10,.0f} "
          f"({sac_reward/heur_reward*100:.1f}%)")
    
    # Run all tests
    results = {}
    compare_trajectories(sac_steps, heur_steps)
    results["temporal"] = test_temporal_rationality(sac_steps)
    results["elasticity"] = test_elasticity_exploitation(sac_steps, config)
    results["budget"] = test_budget_pacing(sac_steps, config)
    results["inventory"] = test_inventory_conditional(sac_steps, config)
    results["local_opt"] = test_local_optimality(env, cont_env, model, config, sac_steps)
    results["stability"] = test_multi_seed_stability(env, cont_env, model, config)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, result in results.items():
        if isinstance(result, bool):
            passed = result
        elif isinstance(result, tuple):
            passed = result[0]
        else:
            passed = bool(result)
        status = "PASS ✓" if passed else "WARN ⚠"
        if not passed:
            all_pass = False
        print(f"  {name:20s}: {status}")
    
    print(f"\n  Overall: {'ALL TESTS PASS ✓' if all_pass else 'SOME TESTS NEED ATTENTION ⚠'}")
    print(f"\n  SAC reward: {sac_reward:,.0f} ({sac_reward/heur_reward*100:.1f}% of heuristic)")
