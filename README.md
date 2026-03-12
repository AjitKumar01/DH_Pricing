# RL-Based Dynamic Pricing Simulator

A reinforcement learning framework for dynamic markdown pricing in retail, featuring a high-fidelity demand simulator with customer choice models, cross-elasticity, promotional attention effects, and a seasonal event calendar spanning Black Friday through January Clearance.

## Overview

This project builds an **RL-based dynamic pricing system** that trains agents to optimally discount a catalog of 15 products over a **91-day (13-week) markdown period** with weekly pricing decisions. The system consists of:

1. **Retail Simulator** — Models customer arrivals (Poisson), demographic segmentation (3 segments), log-normal WTP, self-/cross-elasticity, promotional attention, seasonal events (Black Friday 2.5×, Christmas 2.0×, January Clearance 0.7×), inventory depletion, and budget constraints
2. **Gymnasium MDP Interface** — 70-dimensional state space, multi-discrete action space ($5^{15}$ combinations) with action masking, weekly sticky pricing (13 MDP steps per episode)
3. **RL Training Pipeline** — SAC agent trained via Stable Baselines3
4. **Synthetic Data Generation** — Produces offline MDP tuples and transaction logs for analysis

## Results

| Policy | Margin | Discount Cost | Clearance | Reward |
|--------|--------|---------------|-----------|--------|
| **SAC** | **$21,873** | **$2,868** | **7,364/7,370 (99.9%)** | **26,171** |
| Heuristic | $16,972 | $7,749 | 7,358 (99.8%) | 22,080 |
| Zero Discount | $17,693 | $0 | 4,760 (64.6%) | 17,953 |
| Random | $7,281 | $17,469 | 7,367 (99.9%) | −27,975 |

SAC achieves **29% higher margin** than the heuristic while using **63% less** discount budget.

## Project Structure

```
├── simulator/                  # Core simulator package
│   ├── config.py               # Product catalog, elasticity matrix, MDP parameters, seasonal events
│   ├── customer.py             # Customer arrival, segmentation, WTP, purchase decisions
│   ├── demand.py               # Self-elasticity, cross-elasticity, demand orchestration
│   ├── engine.py               # Core simulation engine (step transitions)
│   ├── environment.py          # Gymnasium Env with weekly sticky pricing, action masking, reward
│   ├── wrappers.py             # SB3 wrappers (MaskableMarkdownEnv, ContinuousActionWrapper)
│   └── data_generator.py       # CSV dataset generation under baseline policies
├── train_agents.py             # SAC training with Rich logging and TensorBoard
├── analyze_trajectory.py       # Detailed trajectory analysis and seasonality verification
├── check_promo_effect.py       # Promotional attention mechanism validation
├── run_simulator.py            # Simulator verification suite (3-part test)
├── report.tex                  # LaTeX report with full mathematical formulation
├── outputs/                    # Trained models, evaluation results, TensorBoard logs
│   └── training/
│       ├── sac_model.zip       # Trained SAC model
│       ├── eval_results.json   # Evaluation metrics (all policies)
│       └── tb_sac/             # TensorBoard logs
├── src/                        # Legacy v1 code (elasticity models, data loading)
├── data/                       # Dunnhumby dataset CSVs
├── archive/                    # Archived previous work
│   └── v1_elasticity_models/   # Original elasticity-based approach
└── scripts/                    # Utility scripts
```

## Setup

```bash
python3 -m venv pricing_env
source pricing_env/bin/activate
pip install numpy pandas gymnasium torch stable-baselines3 sb3-contrib rich tensorboard
```

## Quick Start

### 1. Verify the Simulator

```bash
python run_simulator.py
```

Runs a 3-part verification:
- **Part 1**: Single-step engine test (financial mechanics)
- **Part 2**: Full 13-step Gymnasium episode (weekly decisions over 91 days)
- **Part 3**: CSV data generation under heuristic and random policies

### 2. Train RL Agents

```bash
python train_agents.py
```

Trains SAC for 200K timesteps with Rich progress display. Evaluates against zero-discount, random, and heuristic baselines. Saves models and results to `outputs/training/`.

### 3. Analyze Trajectories

```bash
python analyze_trajectory.py
```

Loads the trained SAC model and runs detailed weekly trajectory analysis showing per-MDP-step metrics, holiday event annotations, and policy comparison across zero-discount, heuristic, and SAC.

### 4. Monitor Training

```bash
tensorboard --logdir outputs/training/
```

## Simulator Design

### Product Catalog

15 products across 5 categories (dairy, bakery, protein, beverages, snacks) with realistic pricing, costs, and inventory levels. Total inventory: 7,370 units over a 91-day markdown horizon.

### Customer Model

- **Arrivals**: Poisson($\lambda=80 \cdot m_{dow} \cdot e_t$) with day-of-week seasonality (Mon 0.80× to Sat 1.30×) and seasonal events (Black Friday 2.5×, Christmas Eve 2.0×, January Clearance 0.7×)
- **Segments**: Budget (40%, WTP 0.70×), Mainstream (40%, 1.00×), Premium (20%, 1.35×)
- **WTP**: LogNormal($\ln(\alpha_k \cdot b_p), 0.15$)
- **Purchase**: Surplus-weighted choice with promotional attention: $w_p = 1 + 3.0 \cdot d_p$

### Demand Dynamics

- **Self-elasticity**: $(1-d_p)^\varepsilon$ with $\varepsilon = -2.0$
- **Cross-elasticity**: Substitutes (+0.15) within categories, complements (−0.075) across dairy-bakery and snacks-beverages
- **Promotional attention**: 10.6× demand lift at 30% discount, with substitute cannibalization

### MDP Formulation

- **State** (70-dim): Previous discounts, normalized demand, inventory, cumulative costs, budget fraction, time fraction, day-of-week one-hot, event multiplier
- **Action**: MultiDiscrete([5]×15) — discount tiers {0%, 10%, 20%, 30%, 50%} per product, applied for 7 consecutive days (weekly sticky pricing)
- **Reward**: $R_t = M_t + 1.0 \cdot \sum q_{t,p} - 0.5 \cdot |\Delta_t - \Delta^*| + \Phi_t$
  - Margin + clearance bonus − pacing penalty + budget overrun penalty (−500)

### Key Design Choices

| Decision | Rationale |
|----------|-----------|
| $\lambda=80$ arrivals/day | Ensures inventory exceeds organic demand (64.6% clearance without discounts) |
| Promotional attention ($\phi=3.0$) | Discounts attract demand beyond affordability — models shelf signage and display effects |
| Budget $9,000 / 91 days | Forces strategic allocation — enough for moderate discounting but not unlimited |
| Weekly sticky pricing ($f=7$) | Matches real retail cadence; reduces MDP horizon to 13 steps |
| Seasonal event calendar | Creates non-stationary demand; agent must anticipate holiday surges and post-holiday decline |
| Action masking | Zero-inventory products forced to 0% discount — prevents wasted budget |

## Report

Compile the LaTeX report:

```bash
pdflatex report.tex
pdflatex report.tex  # twice for table of contents
```
