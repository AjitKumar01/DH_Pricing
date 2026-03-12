"""
RL-Based Dynamic Pricing Simulator.

A Gymnasium-compatible environment for training markdown pricing policies
on a small-scale retail product catalog.
"""

from simulator.config import SimulatorConfig
from simulator.engine import RetailSimulator
from simulator.environment import MarkdownPricingEnv
from simulator.data_generator import DataGenerator
