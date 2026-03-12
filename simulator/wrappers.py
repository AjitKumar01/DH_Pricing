"""
Wrappers for the MarkdownPricingEnv to interface with SB3/sb3-contrib.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from simulator.environment import MarkdownPricingEnv
from simulator.config import SimulatorConfig


class MaskableMarkdownEnv(MarkdownPricingEnv):
    """
    Wrapper that makes MarkdownPricingEnv compatible with sb3-contrib's MaskablePPO.
    Implements the action_masks() method required by MaskablePPO.

    MaskablePPO expects action_masks() to return a flat boolean/int array.
    For MultiDiscrete([5,5,...,5]) with 15 products, the mask is flat: (15*5,) = (75,).
    """

    def __init__(self, config: Optional[SimulatorConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self._current_mask = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._current_mask = info["action_mask"]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._current_mask = info["action_mask"]
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return flat action mask for MaskablePPO. Shape: (n_products * n_tiers,)."""
        if self._current_mask is None:
            return np.ones(self.config.n_products * self.n_tiers, dtype=np.int8)
        return self._current_mask.flatten().astype(np.int8)


class ContinuousActionWrapper(gym.Wrapper):
    """
    Wraps the discrete MarkdownPricingEnv with a continuous action space
    for use with SAC and other continuous-action algorithms.

    Actions are continuous in [0, 1]^n_products, then mapped to the nearest
    allowed discrete discount tier.
    """

    def __init__(self, env: MarkdownPricingEnv):
        super().__init__(env)
        n = env.config.n_products
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n,), dtype=np.float32
        )
        self._discount_tiers = np.array(env.config.allowed_discounts)

    def _continuous_to_discrete(self, continuous_action: np.ndarray) -> np.ndarray:
        """Map continuous [0,1] actions to nearest discrete tier indices."""
        continuous_action = np.clip(continuous_action, 0.0, 1.0)
        # Scale to discount range and find nearest tier
        max_discount = self._discount_tiers[-1]
        target_discounts = continuous_action * max_discount
        # Find nearest tier for each product
        indices = np.array([
            np.argmin(np.abs(self._discount_tiers - d))
            for d in target_discounts
        ])
        return indices

    def step(self, action):
        discrete_action = self._continuous_to_discrete(action)
        return self.env.step(discrete_action)
