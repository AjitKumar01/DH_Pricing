#!/usr/bin/env python3
"""
Train PPO (with action masking) and SAC on the markdown pricing environment.
Compare performance against baseline heuristic and random policies.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from sb3_contrib import MaskablePPO

from simulator.config import SimulatorConfig
from simulator.environment import MarkdownPricingEnv
from simulator.wrappers import MaskableMarkdownEnv, ContinuousActionWrapper

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.panel import Panel

console = Console()

# For small RL networks, CPU is faster than MPS due to transfer overhead.
DEVICE = "cpu"
if torch.backends.mps.is_available():
    console.print("[yellow]Apple Silicon GPU (MPS) available but using CPU (faster for small RL networks)[/]")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    console.print("[green]Using CUDA GPU[/]")
else:
    console.print("[blue]Using CPU[/]")


# ============================================================================
# Evaluation helper
# ============================================================================

def evaluate_policy(env, predict_fn, n_episodes=10, deterministic=True):
    """
    Evaluate a policy over n_episodes. Returns dict of mean metrics.
    predict_fn: callable(obs, info) -> action
    """
    all_rewards = []
    all_revenue = []
    all_margin = []
    all_discount_cost = []
    all_clearance = []
    all_budget_used = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 1000)
        ep_reward = 0.0
        ep_revenue = 0.0
        ep_margin = 0.0
        ep_disc = 0.0

        while True:
            action = predict_fn(obs, info, deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_revenue += info.get("total_revenue", 0)
            ep_margin += info.get("total_margin", 0)
            ep_disc += info.get("total_discount_cost", 0)

            if terminated or truncated:
                break

        stats = info.get("episode_stats", {})
        all_rewards.append(ep_reward)
        all_revenue.append(ep_revenue)
        all_margin.append(ep_margin)
        all_discount_cost.append(ep_disc)
        all_clearance.append(stats.get("total_units_cleared", 0))
        all_budget_used.append(stats.get("budget_used", 0))

    return {
        "reward_mean": np.mean(all_rewards),
        "reward_std": np.std(all_rewards),
        "revenue_mean": np.mean(all_revenue),
        "margin_mean": np.mean(all_margin),
        "discount_cost_mean": np.mean(all_discount_cost),
        "clearance_mean": np.mean(all_clearance),
        "budget_used_mean": np.mean(all_budget_used),
    }


# ============================================================================
# Logging callback with rich live display
# ============================================================================

class RichTrainingCallback(BaseCallback):
    """
    Real-time training monitor using Rich.
    Shows a live-updating table with episode stats and a progress bar.
    """

    def __init__(self, total_timesteps: int, algo_name: str = "Agent", verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.algo_name = algo_name
        self.episode_rewards = []
        self._current_rewards = defaultdict(float)
        self._ep_count = 0
        self._best_reward = -np.inf
        self._last_10_rewards = []
        self.console = Console()

    def _on_training_start(self):
        self.console.print(Panel(
            f"[bold cyan]{self.algo_name}[/] Training Started\n"
            f"Total timesteps: {self.total_timesteps:,}",
            title="Training",
        ))

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            self._current_rewards[i] += self.locals["rewards"][i]
            if done:
                ep_reward = self._current_rewards[i]
                self.episode_rewards.append(ep_reward)
                self._current_rewards[i] = 0.0
                self._ep_count += 1
                self._best_reward = max(self._best_reward, ep_reward)
                self._last_10_rewards.append(ep_reward)
                if len(self._last_10_rewards) > 10:
                    self._last_10_rewards.pop(0)

                # Print progress every 50 episodes
                if self._ep_count % 50 == 0:
                    avg_10 = np.mean(self._last_10_rewards)
                    pct = self.num_timesteps / self.total_timesteps * 100
                    self.console.print(
                        f"  [cyan]{self.algo_name}[/] | "
                        f"Ep {self._ep_count:>5d} | "
                        f"Steps {self.num_timesteps:>7,}/{self.total_timesteps:,} ({pct:.0f}%) | "
                        f"Avg(10): {avg_10:>8.1f} | "
                        f"Best: {self._best_reward:>8.1f}"
                    )
        return True

    def _on_training_end(self):
        avg = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
        self.console.print(Panel(
            f"[bold green]{self.algo_name}[/] Training Complete\n"
            f"Episodes: {self._ep_count} | Final Avg(50): {avg:.1f} | Best: {self._best_reward:.1f}",
            title="Done",
        ))


# ============================================================================
# Baseline policies
# ============================================================================

def zero_discount_policy_fn(obs, info, deterministic=True):
    """Zero-discount baseline: never discount anything."""
    return np.zeros(15, dtype=int)


def random_policy_fn(obs, info, deterministic=False):
    """Random baseline: uniform random tier per product."""
    mask = info.get("action_mask", None)
    n_products = 15
    n_tiers = 5
    action = np.zeros(n_products, dtype=int)
    if mask is not None:
        for p in range(n_products):
            valid = np.where(mask[p] == 1)[0]
            action[p] = np.random.choice(valid)
    else:
        action = np.random.randint(0, n_tiers, size=n_products)
    return action


def heuristic_policy_fn(obs, info, deterministic=True):
    """Heuristic baseline: moderate discounting based on time and inventory."""
    n_products = 15
    n_tiers = 5
    tiers = [0.0, 0.10, 0.20, 0.30, 0.50]

    # Parse state components from normalized observation
    d_prev = obs[:n_products]
    q_prev = obs[n_products:2*n_products]
    i_norm = obs[2*n_products:3*n_products]
    B_frac = obs[4*n_products]
    T_frac = obs[4*n_products + 1]

    action = np.zeros(n_products, dtype=int)
    mask = info.get("action_mask", np.ones((n_products, n_tiers), dtype=np.int8))

    for p in range(n_products):
        if mask[p].sum() <= 1:  # only no-discount allowed
            action[p] = 0
            continue
        # Pressure: high inventory + late in period -> higher discounts
        time_pressure = 1.0 - T_frac
        inv_pressure = i_norm[p]
        pressure = 0.5 * time_pressure + 0.5 * inv_pressure

        tier_idx = min(int(pressure * n_tiers), n_tiers - 1)
        # Budget guard
        if B_frac < 0.2:
            tier_idx = min(tier_idx, 1)
        action[p] = tier_idx

    return action


# ============================================================================
# Training functions
# ============================================================================

def train_ppo(config, total_timesteps=100_000, seed=42):
    """Train MaskablePPO with action masking."""
    console.print("\n[bold cyan]═══ Training PPO (MaskablePPO with Action Masking) ═══[/]")

    env = MaskableMarkdownEnv(config)

    # TensorBoard logging
    tb_log = configure("outputs/training/tb_ppo", ["stdout", "tensorboard"])

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=15,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        ),
        verbose=0,
        seed=seed,
        device=DEVICE,
    )
    model.set_logger(tb_log)

    tracker = RichTrainingCallback(total_timesteps, algo_name="PPO")
    model.learn(total_timesteps=total_timesteps, callback=tracker)

    return model, tracker.episode_rewards


def train_sac(config, total_timesteps=100_000, seed=42):
    """Train SAC with continuous-to-discrete action wrapper."""
    console.print("\n[bold magenta]═══ Training SAC (Continuous Action Wrapper) ═══[/]")

    base_env = MarkdownPricingEnv(config)
    env = ContinuousActionWrapper(base_env)

    # TensorBoard logging
    tb_log = configure("outputs/training/tb_sac", ["stdout", "tensorboard"])

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.995,
        ent_coef="auto",
        policy_kwargs=dict(
            net_arch=[256, 128],
        ),
        verbose=0,
        seed=seed,
        device=DEVICE,
    )
    model.set_logger(tb_log)

    tracker = RichTrainingCallback(total_timesteps, algo_name="SAC")
    model.learn(total_timesteps=total_timesteps, callback=tracker)

    return model, tracker.episode_rewards


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs("outputs/training", exist_ok=True)

    config = SimulatorConfig(seed=42)
    TRAIN_STEPS = 120_000  # Total training timesteps

    # --- Train SAC ---
    sac_model, sac_rewards = train_sac(config, total_timesteps=TRAIN_STEPS, seed=42)
    sac_model.save("outputs/training/sac_model")

    # --- Evaluate all policies ---
    print("\n" + "=" * 60)
    print("EVALUATION (10 episodes each)")
    print("=" * 60)

    eval_env = MarkdownPricingEnv(config)
    eval_env_cont = ContinuousActionWrapper(MarkdownPricingEnv(config))

    # Zero-discount baseline
    zero_results = evaluate_policy(
        eval_env,
        zero_discount_policy_fn,
        n_episodes=10,
    )

    # Random baseline
    rand_results = evaluate_policy(
        eval_env,
        random_policy_fn,
        n_episodes=10,
    )

    # Heuristic baseline
    heur_results = evaluate_policy(
        eval_env,
        heuristic_policy_fn,
        n_episodes=10,
    )

    # SAC
    def sac_predict(obs, info, deterministic):
        action, _ = sac_model.predict(obs, deterministic=deterministic)
        return action

    sac_results = evaluate_policy(eval_env_cont, sac_predict, n_episodes=10)

    # --- Print results ---
    table = Table(title="Policy Comparison (10-episode evaluation)")
    table.add_column("Metric", style="cyan")
    table.add_column("Zero Disc", justify="right")
    table.add_column("Random", justify="right")
    table.add_column("Heuristic", justify="right")
    table.add_column("SAC", justify="right", style="magenta")

    for key in ["reward_mean", "revenue_mean", "margin_mean",
                "discount_cost_mean", "clearance_mean", "budget_used_mean"]:
        label = key.replace("_mean", "").replace("_", " ").title()
        vals = [zero_results[key], rand_results[key], heur_results[key], sac_results[key]]
        if "clearance" in key:
            table.add_row(label, f"{vals[0]:.0f}", f"{vals[1]:.0f}", f"{vals[2]:.0f}", f"{vals[3]:.0f}")
        else:
            table.add_row(label, f"{vals[0]:.2f}", f"{vals[1]:.2f}", f"{vals[2]:.2f}", f"{vals[3]:.2f}")

    console.print(table)

    # --- Save results ---
    results_dict = {
        "zero_discount": zero_results,
        "random": rand_results,
        "heuristic": heur_results,
        "sac": sac_results,
    }

    # Training curves
    training_curves = {
        "sac_rewards": [float(r) for r in sac_rewards],
    }

    with open("outputs/training/eval_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    with open("outputs/training/training_curves.json", "w") as f:
        json.dump(training_curves, f, indent=2)

    # Save as CSV too
    rows = []
    for name, res in results_dict.items():
        row = {"policy": name}
        row.update(res)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/training/policy_comparison.csv", index=False)
    console.print(f"\n[green]Results saved to outputs/training/[/]")
    console.print("[dim]TensorBoard logs: outputs/training/tb_sac/[/]")
    console.print("[dim]Launch TensorBoard: tensorboard --logdir outputs/training/[/]")
    console.print(df.to_string(index=False))

    return results_dict


if __name__ == "__main__":
    main()
