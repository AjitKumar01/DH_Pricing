#!/usr/bin/env python3
"""
Dunnhumby-calibrated Gymnasium environment and SAC training.

Uses parameters estimated from the Dunnhumby dataset to create a
data-driven retail pricing simulator for RL policy training.
"""

import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ============================================================================
# Data-Driven Config
# ============================================================================

@dataclass
class DunnhumbyConfig:
    """Configuration calibrated from Dunnhumby data."""
    markdown_horizon: int = 91
    decision_frequency: int = 7
    allowed_discounts: List[float] = field(default_factory=lambda: [0.0, 0.10, 0.20, 0.30, 0.50])
    seed: Optional[int] = 42

    # Loaded from pipeline output
    products: List[Dict] = field(default_factory=list)
    n_products: int = 0
    day_of_week_multipliers: np.ndarray = field(default_factory=lambda: np.ones(7))
    cross_elasticity_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    total_markdown_budget: float = 10000.0
    customer_segments: Dict = field(default_factory=dict)

    # Reward shaping — economically grounded
    # Terminal salvage loss: fraction of unit_cost lost on each unsold unit.
    # At 0.7× price cost and 0.30 salvage fraction, each unsold unit loses
    # 0.21× price. This makes discounting worthwhile only when extra demand
    # (from elasticity) justifies the per-unit margin loss.
    salvage_loss_fraction: float = 0.30
    budget_overrun_penalty: float = -500.0

    def __post_init__(self):
        if len(self.products) == 0:
            self._load_from_pipeline()

    def _load_from_pipeline(self):
        config_path = "dunnhumby/outputs/simulator_config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Run dunnhumby/build_simulator.py first: {config_path}")

        with open(config_path) as f:
            cfg = json.load(f)

        self.products = cfg['products']
        self.n_products = cfg['n_products']
        self.day_of_week_multipliers = np.array(cfg['day_of_week_multipliers'])
        self.total_markdown_budget = cfg['total_markdown_budget']
        self.customer_segments = cfg.get('customer_segments', {})
        self.cross_elasticity_matrix = np.array(cfg.get('cross_elasticity_matrix', []))

        # Load cross-elasticity matrix: prefer hybrid (IV-calibrated), else semantic
        ce_hybrid = "dunnhumby/outputs/cross_elasticity_hybrid.npy"
        ce_semantic = "dunnhumby/outputs/cross_elasticity_matrix.npy"
        if os.path.exists(ce_hybrid):
            self.cross_elasticity_matrix = np.load(ce_hybrid)
        elif os.path.exists(ce_semantic):
            self.cross_elasticity_matrix = np.load(ce_semantic)

        # Load IV-corrected elasticities from config if available
        if 'elasticities' in cfg and cfg.get('elasticity_source') == 'hybrid_iv_ols':
            elast_list = cfg['elasticities']
            for i, p in enumerate(self.products):
                if i < len(elast_list):
                    p['self_elasticity'] = elast_list[i]


# ============================================================================
# Demand Engine (data-calibrated)
# ============================================================================

class CalibratedDemandEngine:
    """
    Demand engine calibrated to Dunnhumby data.

    Each product has its own Poisson demand rate. Discounts affect demand
    through calibrated self-elasticity: Q = Poisson(base * dow * (1-d)^ε)
    Cross-elasticity adjustments model halo and cannibalization.
    """

    def __init__(self, config: DunnhumbyConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.n = config.n_products

    def simulate_day(self, day: int, discounts: np.ndarray,
                     inventory: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Simulate one day of demand.

        Returns: (units_sold, revenue_per_product, transaction_list)
        """
        dow = day % 7
        dow_mult = self.config.day_of_week_multipliers[dow]

        # Base demand per product (Poisson)
        base_demands = np.zeros(self.n)
        for p in range(self.n):
            base_rate = self.config.products[p]['base_daily_demand']
            elasticity = self.config.products[p].get('self_elasticity', -2.0)

            # Self-elasticity effect: (1 - d)^ε
            # The elasticity was estimated from log(Q) vs log(P) on weekly data,
            # which already includes any promotional visibility effects.
            # No additional promo boost needed — that would double-count.
            if discounts[p] > 0:
                self_mult = (1 - discounts[p]) ** elasticity
            else:
                self_mult = 1.0

            rate = base_rate * dow_mult * self_mult
            base_demands[p] = max(0, self.rng.poisson(max(0.1, rate)))

        # Cross-elasticity adjustments
        if self.config.cross_elasticity_matrix.size > 0:
            E = self.config.cross_elasticity_matrix
            for i in range(self.n):
                cross_adj = 0.0
                for j in range(self.n):
                    if i != j and discounts[j] > 0:
                        cross_adj -= E[i, j] * discounts[j] * base_demands[i]
                base_demands[i] = max(0, base_demands[i] + cross_adj)

        # Apply inventory constraint
        units_sold = np.minimum(base_demands.astype(int), inventory.astype(int))

        # Revenue
        revenue = np.zeros(self.n)
        for p in range(self.n):
            price = self.config.products[p]['base_price'] * (1 - discounts[p])
            revenue[p] = units_sold[p] * price

        return units_sold, revenue, []


# ============================================================================
# Simulator Engine
# ============================================================================

class DunnhumbySimulator:
    """Core retail simulator engine calibrated from Dunnhumby data."""

    def __init__(self, config: DunnhumbyConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.demand = CalibratedDemandEngine(config, self.rng)

        self.n = config.n_products
        self.inventory = np.array([p['initial_inventory'] for p in config.products], dtype=float)
        self.initial_inventory = self.inventory.copy()
        self.day = 0
        self.time_remaining = config.markdown_horizon
        self.budget_remaining = config.total_markdown_budget
        self.cumulative_discount_cost = np.zeros(self.n)
        self.done = False

    def step(self, discounts: np.ndarray) -> Dict:
        """Advance one day."""
        if self.done:
            return self._empty_result()

        units_sold, revenue, _ = self.demand.simulate_day(
            self.day, discounts, self.inventory
        )

        # Financials
        costs = np.zeros(self.n)
        disc_cost = np.zeros(self.n)
        for p in range(self.n):
            costs[p] = units_sold[p] * self.config.products[p]['unit_cost']
            disc_cost[p] = units_sold[p] * self.config.products[p]['base_price'] * discounts[p]

        margin = revenue.sum() - costs.sum()
        total_disc_cost = disc_cost.sum()

        # Update state
        self.inventory -= units_sold
        self.inventory = np.maximum(self.inventory, 0)
        self.day += 1
        self.time_remaining -= 1
        self.budget_remaining -= total_disc_cost
        self.cumulative_discount_cost += disc_cost

        if self.time_remaining <= 0:
            self.done = True

        return {
            'day': self.day,
            'units_sold': units_sold.astype(int),
            'revenue': revenue,
            'total_revenue': float(revenue.sum()),
            'total_margin': float(margin),
            'total_discount_cost': float(total_disc_cost),
            'inventory': self.inventory.copy(),
        }

    def _empty_result(self):
        return {
            'day': self.day,
            'units_sold': np.zeros(self.n, dtype=int),
            'revenue': np.zeros(self.n),
            'total_revenue': 0.0,
            'total_margin': 0.0,
            'total_discount_cost': 0.0,
            'inventory': self.inventory.copy(),
        }


# ============================================================================
# Gymnasium Environment
# ============================================================================

class DunnhumbyPricingEnv(gym.Env):
    """
    RL environment for dynamic pricing, calibrated from Dunnhumby data.

    Observation: [d_{t-1}, q_{t-1}, i_t, c_t_cumul, B_t, T_t, W_t]
    Action: MultiDiscrete([n_tiers] * n_products)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[DunnhumbyConfig] = None):
        super().__init__()
        self.config = config or DunnhumbyConfig()
        self.sim = DunnhumbySimulator(self.config)

        n = self.config.n_products
        n_tiers = len(self.config.allowed_discounts)
        self.n_tiers = n_tiers
        self.discount_tiers = np.array(self.config.allowed_discounts)
        self.decision_freq = self.config.decision_frequency
        self.n_decisions = int(np.ceil(self.config.markdown_horizon / self.decision_freq))

        self.action_space = spaces.MultiDiscrete([n_tiers] * n)

        # Obs: d_prev(n) + q_prev(n) + i_t(n) + c_cumul(n) + B(1) + T(1) + DOW(7)
        obs_dim = 4 * n + 2 + 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._prev_discounts = np.zeros(n)
        self._prev_demand = np.zeros(n)

    def _get_obs(self) -> np.ndarray:
        n = self.config.n_products
        max_inv = np.array([p['initial_inventory'] for p in self.config.products], dtype=np.float32)
        max_inv = np.where(max_inv > 0, max_inv, 1.0)

        d_prev = self._prev_discounts.astype(np.float32)
        q_prev = (self._prev_demand / max_inv).astype(np.float32)
        i_t = (self.sim.inventory / max_inv).astype(np.float32)
        c_cumul = (self.sim.cumulative_discount_cost /
                   (self.config.total_markdown_budget + 1e-8)).astype(np.float32)
        B_t = np.array([self.sim.budget_remaining /
                        (self.config.total_markdown_budget + 1e-8)], dtype=np.float32)
        T_t = np.array([self.sim.time_remaining / self.config.markdown_horizon], dtype=np.float32)

        dow = self.sim.day % 7
        W_t = np.zeros(7, dtype=np.float32)
        W_t[dow] = 1.0

        return np.concatenate([d_prev, q_prev, i_t, c_cumul, B_t, T_t, W_t])

    def _get_action_mask(self) -> np.ndarray:
        n = self.config.n_products
        mask = np.ones((n, self.n_tiers), dtype=np.int8)
        for p in range(n):
            if self.sim.inventory[p] <= 0:
                mask[p, :] = 0
                mask[p, 0] = 1
        # Hard budget constraint: no discounts when budget exhausted
        if self.sim.budget_remaining <= 0:
            mask[:, 1:] = 0
        return mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.config.seed = seed
        self.sim = DunnhumbySimulator(self.config)
        self._prev_discounts = np.zeros(self.config.n_products)
        self._prev_demand = np.zeros(self.config.n_products)
        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask(),
                "budget_remaining": self.sim.budget_remaining,
                "time_remaining": self.sim.time_remaining}
        return obs, info

    def step(self, action):
        n = self.config.n_products
        action = np.asarray(action, dtype=int)
        action = np.clip(action, 0, self.n_tiers - 1)

        for p in range(n):
            if self.sim.inventory[p] <= 0:
                action[p] = 0

        discounts = self.discount_tiers[action]

        # HARD budget constraint: no discounts once budget is exhausted
        if self.sim.budget_remaining <= 0:
            action[:] = 0
            discounts = self.discount_tiers[action]

        total_reward = 0.0
        total_revenue = 0.0
        total_margin = 0.0
        total_disc_cost = 0.0
        total_units = np.zeros(n, dtype=int)
        last_result = None

        days_to_run = min(self.decision_freq, self.sim.time_remaining)
        for _ in range(days_to_run):
            budget_before = self.sim.budget_remaining
            result = self.sim.step(discounts)
            reward = self._compute_reward(result, budget_before)
            total_reward += reward
            total_revenue += result['total_revenue']
            total_margin += result['total_margin']
            total_disc_cost += result['total_discount_cost']
            total_units += result['units_sold']
            last_result = result
            if self.sim.done:
                break

        self._prev_discounts = discounts.copy()
        self._prev_demand = total_units.astype(float).copy()

        terminated = self.sim.done
        truncated = False
        obs = self._get_obs()

        # Episode stats
        total_inv = sum(p['initial_inventory'] for p in self.config.products)
        units_cleared = int(total_inv - self.sim.inventory.sum())

        info = {
            "action_mask": self._get_action_mask(),
            "budget_remaining": self.sim.budget_remaining,
            "time_remaining": self.sim.time_remaining,
            "day": last_result['day'] if last_result else 0,
            "days_in_step": days_to_run,
            "total_revenue": total_revenue,
            "total_margin": total_margin,
            "total_discount_cost": total_disc_cost,
            "units_sold": total_units,
            "inventory": self.sim.inventory.copy(),
            "episode_stats": {
                "total_units_cleared": units_cleared,
                "budget_used": self.config.total_markdown_budget - self.sim.budget_remaining,
            }
        }

        return obs, total_reward, terminated, truncated, info

    def _compute_reward(self, result: Dict, budget_before: float) -> float:
        cfg = self.config

        # Primary: daily margin (revenue - COGS)
        margin = result['total_margin']

        # Terminal: salvage loss on unsold inventory at horizon end
        salvage_loss = 0.0
        if self.sim.done:
            for p_idx in range(self.config.n_products):
                unit_cost = self.config.products[p_idx]['unit_cost']
                salvage_loss += (
                    self.sim.inventory[p_idx]
                    * unit_cost
                    * cfg.salvage_loss_fraction
                )

        return margin - salvage_loss


# ============================================================================
# Continuous Action Wrapper (for SAC)
# ============================================================================

class DunnhumbyContinuousWrapper(gym.Wrapper):
    """Maps continuous [0,1]^n actions to nearest discrete discount tier."""

    def __init__(self, env: DunnhumbyPricingEnv):
        super().__init__(env)
        n = env.config.n_products
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)
        self.tiers = np.array(env.config.allowed_discounts)
        self.config = env.config

    def step(self, action):
        # Map continuous action to nearest discrete tier index
        action = np.clip(action, 0, 1)
        tier_indices = np.zeros(len(action), dtype=int)
        for i, a in enumerate(action):
            tier_indices[i] = np.argmin(np.abs(self.tiers - a))
        return self.env.step(tier_indices)


# ============================================================================
# Training & Evaluation
# ============================================================================

def evaluate_policy(env, predict_fn, n_episodes=10):
    """Evaluate a policy over n episodes."""
    results = {'reward': [], 'margin': [], 'disc_cost': [], 'clearance': [],
               'clearance_pct': []}
    cfg = env.config if hasattr(env, 'config') else env.unwrapped.config
    total_inv = sum(p['initial_inventory'] for p in cfg.products)
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 1000)
        ep_reward = ep_margin = ep_disc = 0.0
        while True:
            action = predict_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_margin += info.get('total_margin', 0)
            ep_disc += info.get('total_discount_cost', 0)
            if terminated or truncated:
                break
        stats = info.get('episode_stats', {})
        units_cleared = stats.get('total_units_cleared', 0)
        results['reward'].append(ep_reward)
        results['margin'].append(ep_margin)
        results['disc_cost'].append(ep_disc)
        results['clearance'].append(units_cleared)
        results['clearance_pct'].append(units_cleared / total_inv * 100 if total_inv > 0 else 0)

    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}


def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from collections import defaultdict

    os.makedirs("dunnhumby/outputs/training", exist_ok=True)

    config = DunnhumbyConfig(seed=42)
    n = config.n_products
    total_inv = sum(p['initial_inventory'] for p in config.products)

    print("=" * 90)
    print(f" DUNNHUMBY-CALIBRATED RL TRAINING")
    print("=" * 90)
    print(f"  Products:    {n}")
    print(f"  Inventory:   {total_inv:,}")
    print(f"  Budget:      ${config.total_markdown_budget:,.0f}")
    print(f"  Horizon:     {config.markdown_horizon} days ({config.markdown_horizon // 7} weeks)")
    print(f"  Decisions:   {config.markdown_horizon // config.decision_frequency} per episode")
    print(f"  Obs dim:     {4*n + 2 + 7}")
    print(f"  Action dim:  {n} × {len(config.allowed_discounts)} tiers")

    # Training callback
    class ProgressCallback(BaseCallback):
        def __init__(self, total_steps, verbose=0):
            super().__init__(verbose)
            self.total_steps = total_steps
            self.episode_rewards = []
            self._current = defaultdict(float)
            self._ep_count = 0
            self._best = -np.inf

        def _on_step(self):
            for i, done in enumerate(self.locals.get("dones", [])):
                self._current[i] += self.locals["rewards"][i]
                if done:
                    r = self._current[i]
                    self.episode_rewards.append(r)
                    self._current[i] = 0
                    self._ep_count += 1
                    self._best = max(self._best, r)
                    if self._ep_count % 100 == 0:
                        avg = np.mean(self.episode_rewards[-50:])
                        pct = self.num_timesteps / self.total_steps * 100
                        print(f"  Ep {self._ep_count:>5} | Steps {self.num_timesteps:>7,}/{self.total_steps:,} "
                              f"({pct:.0f}%) | Avg(50): {avg:>8.1f} | Best: {self._best:>8.1f}")
            return True

    # ---- Train SAC (longer for gap closure) ----
    TRAIN_STEPS = 500_000
    base_env = DunnhumbyPricingEnv(config)
    train_env = DunnhumbyContinuousWrapper(base_env)

    print(f"\n  Training SAC for {TRAIN_STEPS:,} timesteps (IV-corrected elasticities)...")

    # TensorBoard logging
    tb_log = configure("dunnhumby/outputs/training/tb_sac", ["stdout", "tensorboard"])

    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.999,  # high gamma: value terminal salvage penalty from early steps
        ent_coef="auto",
        target_entropy="auto",  # let SAC auto-tune exploration
        policy_kwargs=dict(net_arch=[512, 256]),
        verbose=0,
        seed=42,
        device="cpu",
    )
    model.set_logger(tb_log)

    callback = ProgressCallback(TRAIN_STEPS)
    model.learn(total_timesteps=TRAIN_STEPS, callback=callback)
    model.save("dunnhumby/outputs/training/sac_dunnhumby")
    print(f"  Model saved to dunnhumby/outputs/training/sac_dunnhumby")

    # ---- Evaluate ----
    print(f"\n{'='*90}")
    print(f"  EVALUATION (10 episodes)")
    print(f"{'='*90}")

    eval_env_disc = DunnhumbyPricingEnv(config)
    eval_env_cont = DunnhumbyContinuousWrapper(DunnhumbyPricingEnv(config))

    # Zero discount
    zero_res = evaluate_policy(eval_env_disc,
                                lambda obs, info: np.zeros(n, dtype=int))

    # Heuristic
    def heuristic(obs, info):
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

    heur_res = evaluate_policy(eval_env_disc, heuristic)

    # SAC
    def sac_predict(obs, info):
        action, _ = model.predict(obs, deterministic=True)
        return action

    sac_res = evaluate_policy(eval_env_cont, sac_predict)

    # Print results
    print(f"\n  {'Policy':<15} {'Reward':>12} {'Margin':>12} {'Disc Cost':>12} {'Clearance %':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for name, res in [("Zero Disc", zero_res), ("Heuristic", heur_res), ("SAC", sac_res)]:
        print(f"  {name:<15} {res['reward'][0]:>10,.1f}±{res['reward'][1]:>4.0f} "
              f"${res['margin'][0]:>9,.0f}±{res['margin'][1]:>4.0f} "
              f"${res['disc_cost'][0]:>9,.0f}±{res['disc_cost'][1]:>4.0f} "
              f"{res['clearance_pct'][0]:>9.1f}%±{res['clearance_pct'][1]:>3.1f}%")

    # Gap analysis
    heur_rew = heur_res['reward'][0]
    sac_rew = sac_res['reward'][0]
    gap_pct = (sac_rew / heur_rew) * 100 if heur_rew > 0 else 0
    print(f"\n  SAC: {gap_pct:.1f}% of heuristic")
    if sac_rew > heur_rew:
        print(f"  → SAC SURPASSES heuristic by {sac_rew - heur_rew:+,.1f}")
    else:
        print(f"  → Gap to close: {heur_rew - sac_rew:,.1f}")

    # Save results
    results = {"zero_discount": {k: (float(v[0]), float(v[1])) for k, v in zero_res.items()},
               "heuristic": {k: (float(v[0]), float(v[1])) for k, v in heur_res.items()},
               "sac": {k: (float(v[0]), float(v[1])) for k, v in sac_res.items()}}

    training_curves = {"sac_rewards": [float(r) for r in callback.episode_rewards]}

    with open("dunnhumby/outputs/training/eval_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    with open("dunnhumby/outputs/training/training_curves.json", 'w') as f:
        json.dump(training_curves, f, indent=2)

    print(f"\n  Results saved to dunnhumby/outputs/training/")
    print(f"  Total inventory: {total_inv:,}")
    print(f"  Budget: ${config.total_markdown_budget:,.0f}")


if __name__ == "__main__":
    main()
