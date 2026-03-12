"""
Gymnasium-compatible MDP environment for markdown pricing.

State Space S_t = [d_{t-1}, q_{t-1}, i_t, c_{t,cumul}, B_t, T_t, W_t]
Action Space: MultiDiscrete — one discrete discount tier per product.
Action Masking: zero-discount forced for out-of-stock products.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from simulator.config import SimulatorConfig
from simulator.engine import RetailSimulator


class MarkdownPricingEnv(gym.Env):
    """
    RL environment for dynamic markdown pricing.

    Observation (flattened):
        d_{t-1}          : (n_products,)  previous discount vector
        q_{t-1}          : (n_products,)  previous demand (units sold)
        i_t              : (n_products,)  current inventory
        c_{t,cumulative} : (n_products,)  cumulative discount cost per product
        B_t              : (1,)           remaining markdown budget (scalar)
        T_t              : (1,)           time remaining (scalar)
        W_t              : (7,)           day-of-week one-hot encoding

    Action:
        MultiDiscrete([n_tiers] * n_products)
        Each element indexes into allowed_discounts.

    Reward:
        R_t = total_margin - pacing_penalty - budget_overrun_penalty + clearance_bonus
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[SimulatorConfig] = None, render_mode=None):
        super().__init__()
        self.config = config or SimulatorConfig()
        self.sim = RetailSimulator(self.config)
        self.render_mode = render_mode

        n = self.config.n_products
        n_tiers = len(self.config.allowed_discounts)
        self.n_tiers = n_tiers
        self.discount_tiers = np.array(self.config.allowed_discounts)

        # Decision frequency: agent acts every k days, discounts are sticky
        self.decision_freq = self.config.decision_frequency
        self.n_decisions = int(np.ceil(self.config.markdown_horizon / self.decision_freq))

        # --- Action space: one discrete choice per product ---
        self.action_space = spaces.MultiDiscrete([n_tiers] * n)

        # --- Observation space ---
        # Components and their sizes:
        #   d_{t-1}:          n
        #   q_{t-1}:          n  (normalized by period length for weekly steps)
        #   i_t:              n
        #   c_{t,cumulative}: n
        #   B_t:              1
        #   T_t:              1
        #   W_t:              7
        #   E_t:              1  (upcoming event demand multiplier)
        obs_dim = 4 * n + 2 + 7 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Previous step memory
        self._prev_discounts = np.zeros(n)
        self._prev_demand = np.zeros(n)

        # Budget pacing target (per decision step, not per day)
        self._daily_budget_target = (
            self.config.total_markdown_budget / self.config.markdown_horizon
        )

    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector S_t."""
        n = self.config.n_products

        # Normalize features for neural network friendliness
        max_inv = np.array([p.initial_inventory for p in self.config.products], dtype=np.float32)
        max_inv = np.where(max_inv > 0, max_inv, 1.0)

        d_prev = self._prev_discounts.astype(np.float32)
        q_prev = (self._prev_demand / max_inv).astype(np.float32)  # normalize by initial inv
        i_t = (self.sim.inventory / max_inv).astype(np.float32)
        c_cumul = (self.sim.cumulative_discount_cost /
                   (self.config.total_markdown_budget + 1e-8)).astype(np.float32)
        B_t = np.array([self.sim.budget_remaining /
                        (self.config.total_markdown_budget + 1e-8)], dtype=np.float32)
        T_t = np.array([self.sim.time_remaining / self.config.markdown_horizon],
                       dtype=np.float32)

        # Day-of-week one-hot
        dow = (self.config.start_day_of_week + self.sim.day) % 7
        W_t = np.zeros(7, dtype=np.float32)
        W_t[dow] = 1.0

        # Upcoming event demand multiplier (normalized: 1.0 = no event)
        event_mult = self.config.get_event_multiplier(self.sim.day)
        E_t = np.array([event_mult / 2.5], dtype=np.float32)  # normalize by max event mult

        obs = np.concatenate([d_prev, q_prev, i_t, c_cumul, B_t, T_t, W_t, E_t])
        return obs

    def _get_action_mask(self) -> np.ndarray:
        """
        Build action mask: shape (n_products, n_tiers).
        For out-of-stock products, only tier 0 (0% discount) is allowed.
        """
        n = self.config.n_products
        mask = np.ones((n, self.n_tiers), dtype=np.int8)
        for p in range(n):
            if self.sim.inventory[p] <= 0:
                mask[p, :] = 0
                mask[p, 0] = 1  # only allow 0% discount
        return mask

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.config.seed = seed
        self.sim = RetailSimulator(self.config)
        self._prev_discounts = np.zeros(self.config.n_products)
        self._prev_demand = np.zeros(self.config.n_products)

        obs = self._get_obs()
        info = {
            "action_mask": self._get_action_mask(),
            "budget_remaining": self.sim.budget_remaining,
            "time_remaining": self.sim.time_remaining,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one MDP step (covers decision_freq days).

        The agent selects discount tiers once; these are held constant
        ("sticky pricing") for decision_freq consecutive simulation days.
        Rewards are summed over the sub-steps.

        Args:
            action: array of shape (n_products,) with tier indices.

        Returns:
            obs, reward, terminated, truncated, info
        """
        n = self.config.n_products

        # Decode action indices to discount fractions
        action = np.asarray(action, dtype=int)
        action = np.clip(action, 0, self.n_tiers - 1)

        # Apply action mask: force 0 for out-of-stock
        for p in range(n):
            if self.sim.inventory[p] <= 0:
                action[p] = 0

        discounts = self.discount_tiers[action]

        # --- Run decision_freq sub-steps with sticky discounts ---
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
            total_revenue += result.total_revenue
            total_margin += result.total_margin
            total_disc_cost += result.total_discount_cost
            total_units += result.units_sold
            last_result = result
            if self.sim.done:
                break

        # Update previous step memory (from last sub-step)
        self._prev_discounts = discounts.copy()
        self._prev_demand = total_units.astype(float).copy()

        # Check termination
        terminated = self.sim.done
        truncated = False

        # Build info dict
        obs = self._get_obs()
        info = {
            "action_mask": self._get_action_mask(),
            "budget_remaining": self.sim.budget_remaining,
            "time_remaining": self.sim.time_remaining,
            "day": last_result.day if last_result else 0,
            "days_in_step": days_to_run,
            "total_revenue": total_revenue,
            "total_margin": total_margin,
            "total_discount_cost": total_disc_cost,
            "units_sold": total_units.tolist(),
            "inventory": self.sim.inventory.tolist(),
        }

        # Terminal info
        if terminated:
            info["episode_stats"] = {
                "total_units_cleared": int(self.sim.total_units_sold.sum()),
                "final_inventory": self.sim.inventory.tolist(),
                "budget_used": float(
                    self.config.total_markdown_budget - self.sim.budget_remaining
                ),
                "budget_remaining": float(self.sim.budget_remaining),
            }

        return obs, total_reward, terminated, truncated, info

    def _compute_reward(self, result, budget_before: float) -> float:
        """
        Reward function:

        R_t = margin_t
              + clearance_bonus * units_sold
              - pacing_penalty * |actual_spend - target_spend|
              - budget_overrun_penalty * I(B_t < 0 and T_t > 0)

        The pacing penalty encourages spending at a uniform rate over time.
        """
        cfg = self.config

        # 1. Margin component (primary objective)
        margin = result.total_margin

        # 2. Clearance bonus (incentivize selling through inventory)
        clearance = cfg.clearance_bonus_per_unit * result.units_sold.sum()

        # 3. Pacing penalty — penalize deviating from target daily spend
        actual_spend = result.total_discount_cost
        pacing_dev = abs(actual_spend - self._daily_budget_target)
        pacing_penalty = cfg.pacing_penalty_coeff * pacing_dev

        # 4. Budget overrun penalty
        budget_penalty = 0.0
        if self.sim.budget_remaining < 0 and self.sim.time_remaining > 0:
            budget_penalty = cfg.budget_overrun_penalty

        reward = margin + clearance - pacing_penalty + budget_penalty
        return float(reward)

    def render(self):
        if self.render_mode == "human":
            print(f"Day {self.sim.day}/{self.config.markdown_horizon} | "
                  f"Budget: ${self.sim.budget_remaining:.2f} | "
                  f"Inventory: {self.sim.inventory.sum()} units")
