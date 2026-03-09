"""Product-level markdown pricing simulator.

MDP environment with per-product state and action vectors.
State: [selection, discount, demand, budget_consumed, budget_remaining, time_remaining]
Action: [new_selection, new_discount]

The demand transition is no longer memoryless: lagged demand (AR(1)
persistence) feeds forward, and cross-product substitution within
segments means the discount decision for one product affects demand
for others.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class ProductLevelSimulator:
    """Product-level markdown pricing MDP simulator.

    The simulator maintains an individual state for each of N products
    and computes demand via calibrated log-log models with lagged demand
    persistence, cross-product substitution, and stochastic promotional
    events.

    State space dimension: 4*N + 2
        - selection (N): binary, which products are discounted
        - discount (N): discount depth per product
        - demand (N): most recent demand realization per product
        - budget_consumed (N): cumulative discount cost per product
        - budget_remaining (1): remaining total discount budget
        - time_remaining (1): remaining periods

    Action space dimension: 2*N
        - new_selection (N): binary, which products to discount
        - new_discount (N): new discount depth per product

    The demand transition uses the state:
        - demand(t): previous demand enters via AR(1) persistence (rho_i)
        - selection/discount: cross-product substitution means other
          products' discounts affect own demand (lambda_i)
    """

    def __init__(
        self,
        product_params: pd.DataFrame,
        budget: float,
        horizon: int = 12,
        max_discount: float = 0.45,
        phi: float = 1.0,
        noise: bool = True,
        start_week: int = 83,
    ):
        """Initialize the simulator.

        Args:
            product_params: DataFrame with columns:
                PRODUCT_ID, intercept_calibrated, elasticity_shrunk,
                disc_effect_shrunk, base_price, base_demand, residual_std,
                segment, demand_persistence_shrunk, substitution_effect_shrunk
            budget: Total discount budget for the episode.
            horizon: Number of time periods.
            max_discount: Maximum allowed discount depth.
            phi: Log-retransformation bias correction factor.
            noise: Whether to add stochastic demand noise.
            start_week: Calendar week at which the episode starts (for seasonality).
        """
        self.pp = product_params.reset_index(drop=True)
        self.N = len(self.pp)
        self.budget = budget
        self.horizon = horizon
        self.max_discount = max_discount
        self.phi = phi
        self.noise = noise
        self.start_week = start_week

        # Pre-extract arrays for fast vectorized computation
        self._intercepts = self.pp["intercept_calibrated"].values
        self._elasticities = self.pp["elasticity_shrunk"].values
        self._disc_effects = self.pp["disc_effect_shrunk"].values
        self._base_prices = self.pp["base_price"].values
        self._base_demands = self.pp["base_demand"].values
        self._residual_stds = self.pp["residual_std"].values

        # AR(1) demand persistence coefficients
        self._persistence = self.pp.get(
            "demand_persistence_shrunk",
            self.pp.get("demand_persistence", pd.Series(np.zeros(self.N)))
        ).fillna(0).values

        # Cross-product substitution coefficients
        self._substitution = self.pp.get(
            "substitution_effect_shrunk",
            self.pp.get("substitution_effect", pd.Series(np.zeros(self.N)))
        ).fillna(0).values

        # Build segment membership for substitution computation
        self._segments = self.pp.get(
            "segment", pd.Series(np.zeros(self.N, dtype=int))
        ).values
        self._unique_segments = np.unique(self._segments)
        # For each product, precompute segment mask (other products in same segment)
        self._seg_masks = {}
        for seg in self._unique_segments:
            mask = self._segments == seg
            self._seg_masks[seg] = mask

        # Mean log-demand per product (for AR(1) deviation computation)
        self._mean_log_demand = np.log(np.clip(self._base_demands, 0.5, None))

        # Pre-extract seasonal coefficients
        self._cos_coefs = self.pp.get(
            "cos_season_coef", pd.Series(np.zeros(self.N))
        ).fillna(0).values
        self._sin_coefs = self.pp.get(
            "sin_season_coef", pd.Series(np.zeros(self.N))
        ).fillna(0).values

        # Stochastic promotional events: display and mailer
        self._display_effects = self.pp.get(
            "display_effect_shrunk",
            self.pp.get("display_effect", pd.Series(np.zeros(self.N)))
        ).fillna(0).values
        self._mailer_effects = self.pp.get(
            "mailer_effect_shrunk",
            self.pp.get("mailer_effect", pd.Series(np.zeros(self.N)))
        ).fillna(0).values
        self._display_mean = self.pp.get(
            "display_mean", pd.Series(np.zeros(self.N))
        ).fillna(0).values
        self._display_std = self.pp.get(
            "display_std", pd.Series(np.zeros(self.N))
        ).fillna(0).values
        self._mailer_mean = self.pp.get(
            "mailer_mean", pd.Series(np.zeros(self.N))
        ).fillna(0).values
        self._mailer_std = self.pp.get(
            "mailer_std", pd.Series(np.zeros(self.N))
        ).fillna(0).values

        # Training-period mean substitution level for deviation computation
        self._mean_seg_disc_other = self.pp.get(
            "mean_seg_disc_other", pd.Series(np.zeros(self.N))
        ).fillna(0).values

        self.state_dim = 4 * self.N + 2
        self.action_dim = 2 * self.N

        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self._selection = np.zeros(self.N)
        self._discount = np.zeros(self.N)
        self._demand = self._base_demands.copy()
        self._budget_consumed = np.zeros(self.N)
        self._budget_remaining = self.budget
        self._time_remaining = self.horizon
        self._current_week = self.start_week
        self._done = False

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Construct the state vector."""
        return np.concatenate([
            self._selection,
            self._discount,
            self._demand,
            self._budget_consumed,
            [self._budget_remaining],
            [self._time_remaining],
        ])

    def _compute_demand(
        self, selection: np.ndarray, discount: np.ndarray
    ) -> np.ndarray:
        """Compute demand for all products given selection and discount.

        The demand model uses the full state:
            log(q_t) = alpha + epsilon * log(shelf_price) + gamma * discount_depth
                       + rho * (log(q_{t-1}) - mean_log_q)    [AR(1) persistence]
                       + lambda * seg_disc_other               [substitution]
                       + delta_D * (D - mean_D) + delta_M * (M - mean_M)
                       + b1*(cos - mean_cos) + b2*(sin - mean_sin) + noise

        The AR(1) term makes demand genuinely dynamic: previous demand
        realizations feed forward. The substitution term means other
        products' discount decisions affect own demand.
        """
        effective_discount = selection * discount

        # Shelf price is fixed — only discount_depth varies
        log_base_price = np.log(np.clip(self._base_prices, 0.01, None))

        log_demand = (
            self._intercepts
            + self._elasticities * log_base_price
            + self._disc_effects * effective_discount
        )

        # ── AR(1) demand persistence ──
        # Use deviation of lagged log-demand from the training-period
        # mean (which is absorbed in the calibrated intercept).
        log_prev_demand = np.log(np.clip(self._demand, 0.5, None))
        log_demand += self._persistence * (log_prev_demand - self._mean_log_demand)

        # ── Cross-product substitution ──
        # For each product, compute the mean discount depth of OTHER
        # products in the same segment. The substitution effect (<=0)
        # captures cannibalization from competing promotions.
        # Enter as deviation from training-period mean (calibration absorbed mean).
        seg_disc_other = np.zeros(self.N)
        for seg in self._unique_segments:
            mask = self._seg_masks[seg]
            n_seg = mask.sum()
            if n_seg <= 1:
                continue
            total_disc = np.sum(effective_discount[mask])
            seg_idx = np.where(mask)[0]
            for idx in seg_idx:
                seg_disc_other[idx] = (total_disc - effective_discount[idx]) / (n_seg - 1)

        log_demand += self._substitution * (seg_disc_other - self._mean_seg_disc_other)

        # Seasonal deviation from mean (calibration absorbed mean season)
        week = self._current_week
        cos_val = np.cos(2 * np.pi * week / 52)
        sin_val = np.sin(2 * np.pi * week / 52)
        train_weeks = np.arange(18, 83)
        mean_cos = np.mean(np.cos(2 * np.pi * train_weeks / 52))
        mean_sin = np.mean(np.sin(2 * np.pi * train_weeks / 52))
        log_demand += self._cos_coefs * (cos_val - mean_cos)
        log_demand += self._sin_coefs * (sin_val - mean_sin)

        # Stochastic promotional events (deviation from mean)
        if self.noise:
            display_draw = np.clip(
                self._display_mean + self._display_std * np.random.randn(self.N),
                0, 1
            )
            mailer_draw = np.clip(
                self._mailer_mean + self._mailer_std * np.random.randn(self.N),
                0, 1
            )
            log_demand += self._display_effects * (
                display_draw - self._display_mean
            )
            log_demand += self._mailer_effects * (
                mailer_draw - self._mailer_mean
            )

        demand = self.phi * np.exp(log_demand)

        if self.noise:
            noise = np.random.normal(0, self._residual_stds)
            demand = demand * np.exp(noise)

        return np.clip(demand, 0, None)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Array of shape (2*N,) where first N values are selection
                (0/1) and last N values are discount depths.

        Returns:
            state: New state vector.
            reward: Total revenue in this period.
            done: Whether the episode is finished.
            info: Dictionary with diagnostic information.
        """
        if self._done:
            return self._get_state(), 0.0, True, {}

        # Parse action
        selection = (action[:self.N] > 0.5).astype(float)
        discount = np.clip(action[self.N:], 0, self.max_discount)

        # Compute demand
        demand = self._compute_demand(selection, discount)

        # Compute revenue and cost
        effective_price = self._base_prices * (1 - selection * discount)
        revenue = effective_price * demand
        discount_cost = self._base_prices * selection * discount * demand
        total_cost = discount_cost.sum()

        # Budget enforcement: scale down discount if over budget.
        # Do NOT recompute demand (which would redraw stochastic events),
        # just rescale the discount and recompute revenue/cost analytically.
        if total_cost > self._budget_remaining and total_cost > 0:
            scale = self._budget_remaining / total_cost
            discount = discount * scale
            effective_price = self._base_prices * (1 - selection * discount)
            revenue = effective_price * demand
            discount_cost = self._base_prices * selection * discount * demand
            total_cost = discount_cost.sum()

        # Update state
        self._selection = selection
        self._discount = discount
        self._demand = demand
        self._budget_consumed += discount_cost
        self._budget_remaining -= min(total_cost, self._budget_remaining)
        self._time_remaining -= 1
        self._current_week += 1
        self._done = self._time_remaining <= 0

        total_revenue = revenue.sum()

        info = {
            "total_revenue": total_revenue,
            "total_demand": demand.sum(),
            "total_discount_cost": total_cost,
            "budget_remaining": self._budget_remaining,
            "time_remaining": self._time_remaining,
            "per_product_revenue": revenue,
            "per_product_demand": demand,
        }

        return self._get_state(), total_revenue, self._done, info

    def rollout(
        self, policy_fn, seed: Optional[int] = None
    ) -> Tuple[float, list]:
        """Execute a full episode rollout with a given policy function.

        Args:
            policy_fn: Callable(state) -> action
            seed: Optional random seed.

        Returns:
            total_revenue: Cumulative revenue over the episode.
            history: List of per-step info dicts.
        """
        state = self.reset(seed=seed)
        total_revenue = 0
        history = []

        while not self._done:
            action = policy_fn(state)
            state, reward, done, info = self.step(action)
            total_revenue += reward
            history.append(info)

        return total_revenue, history
