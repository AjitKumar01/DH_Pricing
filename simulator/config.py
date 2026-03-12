"""
Simulator configuration: product catalog, elasticity parameters, and MDP settings.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class ProductSpec:
    """Specification for a single product."""
    product_id: int
    name: str
    category: str
    base_price: float
    unit_cost: float
    initial_inventory: int


@dataclass
class SeasonalEvent:
    """A named event that modifies demand over a range of days."""
    name: str
    day_start: int          # inclusive, relative to markdown period start
    day_end: int            # inclusive
    demand_multiplier: float  # applied on top of day-of-week effect


@dataclass
class SimulatorConfig:
    """Master configuration for the retail pricing simulator."""

    # --- Time horizon ---
    markdown_horizon: int = 91          # 13 weeks (a full retail quarter)
    start_day_of_week: int = 0          # 0=Monday
    decision_frequency: int = 7         # agent acts every k days (sticky pricing)

    # --- Customer arrival ---
    base_daily_arrivals: float = 80.0    # λ for Poisson arrival process
    day_of_week_multipliers: np.ndarray = field(default_factory=lambda: np.array(
        [0.8, 0.85, 0.9, 1.0, 1.15, 1.3, 1.2]  # Mon-Sun
    ))

    # --- Holiday / seasonal event calendar ---
    # Day offsets assume the 91-day period starts the first Monday of November
    # Week 1 = days 0-6 (Nov 3-9), Week 4 = days 21-27 (Nov 24-30, Thanksgiving week),
    # Week 8 = days 49-55 (Dec 22-28, Christmas), Week 9 = days 56-62 (Dec 29 - Jan 4, NYE),
    # Weeks 10-13 = January clearance.
    seasonal_events: List[SeasonalEvent] = field(default_factory=lambda: [
        SeasonalEvent("Black Friday",          day_start=25, day_end=25, demand_multiplier=2.5),
        SeasonalEvent("Thanksgiving Weekend",  day_start=26, day_end=27, demand_multiplier=1.8),
        SeasonalEvent("Cyber Week",            day_start=28, day_end=34, demand_multiplier=1.4),
        SeasonalEvent("Pre-Christmas Rush",    day_start=42, day_end=52, demand_multiplier=1.5),
        SeasonalEvent("Christmas Eve",         day_start=53, day_end=53, demand_multiplier=2.0),
        SeasonalEvent("Boxing Day Sales",      day_start=55, day_end=57, demand_multiplier=1.8),
        SeasonalEvent("New Year's Eve",        day_start=62, day_end=62, demand_multiplier=1.6),
        SeasonalEvent("January Clearance",     day_start=70, day_end=90, demand_multiplier=0.7),
    ])

    # --- Customer demographics ---
    n_customer_segments: int = 3
    segment_names: List[str] = field(default_factory=lambda: [
        "budget", "mainstream", "premium"
    ])
    segment_proportions: np.ndarray = field(default_factory=lambda: np.array(
        [0.40, 0.40, 0.20]
    ))
    # WTP multiplier per segment (applied to base_price to get mean WTP)
    segment_wtp_multipliers: np.ndarray = field(default_factory=lambda: np.array(
        [0.70, 1.00, 1.35]
    ))
    wtp_noise_std: float = 0.15  # std of log-normal noise on WTP

    # --- Demand & elasticity ---
    # Self-elasticity: log-log exponent (negative = elastic)
    default_self_elasticity: float = -2.0
    # Cross-elasticity is defined per category pair (set in build_catalog)
    cross_elasticity_strength: float = 0.15  # magnitude of cross effects

    # --- Discount action space ---
    allowed_discounts: List[float] = field(default_factory=lambda: [
        0.0, 0.10, 0.20, 0.30, 0.50
    ])

    # --- Budget ---
    total_markdown_budget: float = 9000.0

    # --- Reward shaping ---
    budget_overrun_penalty: float = -500.0   # terminal penalty if budget exhausted early
    pacing_penalty_coeff: float = 0.5        # coefficient for pacing deviation penalty
    clearance_bonus_per_unit: float = 1.0    # bonus per unit cleared

    # --- Random seed ---
    seed: Optional[int] = 42

    # --- Product catalog (built by helper) ---
    products: List[ProductSpec] = field(default_factory=list)
    n_products: int = 0
    categories: List[str] = field(default_factory=list)
    cross_elasticity_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    # Precomputed daily event multipliers (built in __post_init__)
    _daily_event_multipliers: np.ndarray = field(default_factory=lambda: np.array([]),
                                                  repr=False)

    def __post_init__(self):
        if len(self.products) == 0:
            self._build_default_catalog()
        self._build_event_calendar()

    def _build_event_calendar(self):
        """Precompute a daily multiplier array from the seasonal event list."""
        mults = np.ones(self.markdown_horizon)
        for event in self.seasonal_events:
            start = max(0, event.day_start)
            end = min(self.markdown_horizon - 1, event.day_end)
            for d in range(start, end + 1):
                if event.demand_multiplier >= 1.0:
                    mults[d] = max(mults[d], event.demand_multiplier)
                else:
                    mults[d] = min(mults[d], event.demand_multiplier)
        self._daily_event_multipliers = mults

    def get_event_multiplier(self, day: int) -> float:
        """Get the seasonal event demand multiplier for a given day."""
        if 0 <= day < len(self._daily_event_multipliers):
            return float(self._daily_event_multipliers[day])
        return 1.0

    def get_active_events(self, day: int) -> List[str]:
        """Get names of active events on a given day."""
        return [e.name for e in self.seasonal_events
                if e.day_start <= day <= e.day_end]

    def _build_default_catalog(self):
        """Build a 15-product catalog across 5 categories."""
        catalog_defs = [
            # (name, category, base_price, unit_cost, initial_inventory)
            ("Organic Milk 1L",       "dairy",       4.50,  2.20,  600),
            ("Greek Yogurt 500g",     "dairy",       5.00,  2.50,  550),
            ("Cheddar Cheese 200g",   "dairy",       6.00,  3.00,  450),
            ("Sourdough Bread",       "bakery",      5.50,  2.00,  650),
            ("Croissant 4-pack",      "bakery",      4.00,  1.50,  600),
            ("Whole Wheat Wrap 6pk",  "bakery",      3.50,  1.20,  550),
            ("Atlantic Salmon 300g",  "protein",    12.00,  7.00,  300),
            ("Chicken Breast 500g",   "protein",     8.00,  4.50,  450),
            ("Plant-Based Burger 2pk","protein",     9.50,  5.00,  350),
            ("Cabernet Sauvignon",    "beverages",  15.00,  8.00,  250),
            ("Craft IPA 6-pack",      "beverages",  12.00,  6.50,  300),
            ("Sparkling Water 12pk",  "beverages",   6.00,  2.50,  650),
            ("Dark Chocolate Bar",    "snacks",      4.00,  1.80,  750),
            ("Trail Mix 400g",        "snacks",      7.00,  3.50,  500),
            ("Protein Bar 6-pack",    "snacks",      8.50,  4.00,  420),
        ]
        self.products = []
        for pid, (name, cat, bp, uc, inv) in enumerate(catalog_defs):
            self.products.append(ProductSpec(
                product_id=pid, name=name, category=cat,
                base_price=bp, unit_cost=uc, initial_inventory=inv
            ))
        self.n_products = len(self.products)
        self.categories = sorted(set(p.category for p in self.products))
        self._build_cross_elasticity_matrix()

    def _build_cross_elasticity_matrix(self):
        """
        Build cross-elasticity matrix.
        Same-category products are substitutes (positive cross-elasticity).
        Select cross-category pairs are complements (negative cross-elasticity).
        """
        n = self.n_products
        E = np.zeros((n, n))
        s = self.cross_elasticity_strength

        # Substitutes: same category
        for i in range(n):
            for j in range(n):
                if i != j and self.products[i].category == self.products[j].category:
                    E[i, j] = s  # discounting j boosts demand for substitutes

        # Complements: define specific category pairs
        complement_pairs = [
            ("dairy", "bakery"),      # bread + cheese/milk
            ("snacks", "beverages"),   # snacks + drinks
        ]
        cat_map = {p.product_id: p.category for p in self.products}
        for i in range(n):
            for j in range(n):
                if i != j:
                    ci, cj = cat_map[i], cat_map[j]
                    if (ci, cj) in complement_pairs or (cj, ci) in complement_pairs:
                        E[i, j] = -s * 0.5  # negative = complement boost

        self.cross_elasticity_matrix = E
