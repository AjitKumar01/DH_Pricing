#!/usr/bin/env python3
"""Verify extended simulator with 91-day horizon, holidays, and weekly decisions."""

from simulator.config import SimulatorConfig
from simulator.environment import MarkdownPricingEnv
import numpy as np

config = SimulatorConfig(seed=42)
print(f"Horizon: {config.markdown_horizon} days ({config.markdown_horizon // 7} weeks)")
print(f"Decision freq: {config.decision_frequency} days")
print(f"Total inventory: {sum(p.initial_inventory for p in config.products)}")
print(f"Budget: ${config.total_markdown_budget:,.0f}")
print(f"Obs dim: {4*config.n_products + 2 + 7 + 1}")

# Show event calendar
print("\nEvent Calendar:")
for e in config.seasonal_events:
    print(f"  Day {e.day_start:>2}-{e.day_end:>2}: {e.name:<25} ({e.demand_multiplier:.1f}x)")

env = MarkdownPricingEnv(config)
print(f"\nMDP steps per episode: {env.n_decisions}")

# Run zero-discount episode
obs, info = env.reset()
print(f"Obs shape: {obs.shape}")
steps = 0
total_reward = 0
for _ in range(env.n_decisions + 5):
    action = np.zeros(15, dtype=int)
    obs, reward, term, trunc, info = env.step(action)
    steps += 1
    days = info.get("days_in_step", "?")
    events = config.get_active_events(info["day"])
    ev_str = f'  [{" | ".join(events)}]' if events else ""
    print(f"  Step {steps:>2}: {days}d, day={info['day']:>2}, inv={sum(info['inventory']):>5}, reward={reward:>8.1f}{ev_str}")
    total_reward += reward
    if term or trunc:
        break

stats = info["episode_stats"]
total_inv = sum(p.initial_inventory for p in config.products)
print(f"\nTotal steps: {steps}")
print(f"Total reward: {total_reward:.1f}")
print(f"Units cleared: {stats['total_units_cleared']} / {total_inv}")
print(f"Clearance: {stats['total_units_cleared']/total_inv*100:.1f}%")
