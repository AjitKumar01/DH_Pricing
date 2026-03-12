"""
Customer choice model: Poisson arrivals, demographic segmentation, WTP-based purchase decisions.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Customer:
    """A single arriving customer."""
    segment: int        # index into segment_names
    wtp: np.ndarray     # willingness-to-pay vector (one per product)


class CustomerChoiceModel:
    """
    Generates daily customer arrivals and purchase decisions.

    Each day:
      1. Draw N ~ Poisson(λ * dow_multiplier) customers.
      2. Assign each customer a demographic segment.
      3. For each customer & product, draw WTP ~ LogNormal(μ_seg, σ).
      4. Customer buys product p if discounted_price_p ≤ wtp_p.
    """

    def __init__(self, config, rng: np.random.Generator):
        self.config = config
        self.rng = rng

        # Pre-compute per-segment mean WTP for each product
        n_prod = config.n_products
        n_seg = config.n_customer_segments
        self.mean_wtp = np.zeros((n_seg, n_prod))
        for s in range(n_seg):
            for p in range(n_prod):
                self.mean_wtp[s, p] = (
                    config.products[p].base_price
                    * config.segment_wtp_multipliers[s]
                )

    def generate_arrivals(self, day_of_week: int) -> int:
        """Draw number of customer arrivals for this day."""
        lam = (self.config.base_daily_arrivals
               * self.config.day_of_week_multipliers[day_of_week])
        return int(self.rng.poisson(lam))

    def generate_customers(self, n_customers: int) -> List[Customer]:
        """Generate a batch of customers with segments and WTP vectors."""
        segments = self.rng.choice(
            self.config.n_customer_segments,
            size=n_customers,
            p=self.config.segment_proportions
        )
        customers = []
        for seg in segments:
            # Log-normal WTP: log(WTP) ~ N(log(μ), σ)
            log_mu = np.log(self.mean_wtp[seg] + 1e-8)
            log_sigma = self.config.wtp_noise_std
            wtp = self.rng.lognormal(mean=log_mu, sigma=log_sigma)
            customers.append(Customer(segment=int(seg), wtp=wtp))
        return customers

    def purchase_decisions(
        self,
        customers: List[Customer],
        shelf_prices: np.ndarray,
        inventory: np.ndarray,
        discounts: np.ndarray = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Determine what each customer buys.

        Discounts create two effects:
          1. Affordability: lower price means WTP >= price for more customers.
          2. Promotional lift: discounted items attract attention, boosting
             selection probability via an attention weight.

        Returns:
            units_sold: array of shape (n_products,) with total units sold.
            transactions: list of dicts recording each purchase event.
        """
        n_prod = self.config.n_products
        units_sold = np.zeros(n_prod, dtype=int)
        available = inventory.copy()
        transactions = []

        # Promotional attention weights: discounted items are more prominent
        # attention_weight = 1 + promo_lift * discount_fraction
        promo_lift = 3.0  # a 30% discount gives 1.9x attention boost
        if discounts is not None:
            attention = 1.0 + promo_lift * discounts
        else:
            attention = np.ones(n_prod)

        for cust in customers:
            # Products the customer is willing to buy (WTP >= price) and in stock
            affordable = (cust.wtp >= shelf_prices) & (available > 0)
            candidate_ids = np.where(affordable)[0]

            if len(candidate_ids) == 0:
                continue

            # Selection probability: surplus * attention weight
            surplus = cust.wtp[candidate_ids] - shelf_prices[candidate_ids]
            surplus = np.maximum(surplus, 0.0)
            weighted_surplus = surplus * attention[candidate_ids]
            total = weighted_surplus.sum()
            if total < 1e-12:
                probs = np.ones(len(candidate_ids)) / len(candidate_ids)
            else:
                probs = weighted_surplus / total

            # Each customer buys at most 1 unit of 1 product (simplification)
            chosen = self.rng.choice(candidate_ids, p=probs)
            if available[chosen] > 0:
                units_sold[chosen] += 1
                available[chosen] -= 1
                transactions.append({
                    "segment": cust.segment,
                    "product_id": int(chosen),
                    "wtp": float(cust.wtp[chosen]),
                    "price_paid": float(shelf_prices[chosen]),
                    "surplus": float(cust.wtp[chosen] - shelf_prices[chosen]),
                })

        return units_sold, transactions
