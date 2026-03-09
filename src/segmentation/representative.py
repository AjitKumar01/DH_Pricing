"""Representative product selection from segments."""

import numpy as np
import pandas as pd


class RepresentativeSelector:
    """Select representative products using stratified sampling within segments.

    Instead of selecting only the top-revenue products (which creates
    head-SKU bias), this selector uses stratified sampling across
    revenue terciles within each segment: top, middle, and bottom
    thirds of the revenue distribution. This ensures the selected
    products span the full revenue range, covering not just high-velocity
    items but also mid-range and long-tail products where promotional
    response may differ.
    """

    def __init__(self, n_products: int = 150, min_weeks: int = 30, min_price_cv: float = 0.03):
        self.n_products = n_products
        self.min_weeks = min_weeks
        self.min_price_cv = min_price_cv

    def select(
        self,
        product_features: pd.DataFrame,
        segment_mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        """Select representative products via stratified revenue sampling.

        Within each segment, products are divided into revenue terciles
        (top, middle, bottom) and products are drawn from each tercile.
        This provides coverage of the full revenue distribution.

        Args:
            product_features: DataFrame with PRODUCT_ID, total_revenue,
                n_weeks, price_cv columns.
            segment_mapping: DataFrame with PRODUCT_ID, segment columns.

        Returns:
            DataFrame of selected products with segment assignments.
        """
        df = product_features.merge(segment_mapping, on="PRODUCT_ID", how="inner")

        # Apply data quality filters
        eligible = df[
            (df["n_weeks"] >= self.min_weeks) & (df["price_cv"] >= self.min_price_cv)
        ].copy()

        # Proportional allocation by segment revenue
        # Minimum 2 per segment for meaningful shrinkage estimation
        seg_rev = eligible.groupby("segment")["total_revenue"].sum()
        total_rev = seg_rev.sum()
        seg_alloc = (seg_rev / total_rev * self.n_products).round().astype(int).clip(lower=2)

        # Adjust to hit exact target
        while seg_alloc.sum() > self.n_products:
            seg_alloc[seg_alloc.idxmax()] -= 1
        while seg_alloc.sum() < self.n_products:
            seg_alloc[seg_alloc.idxmin()] += 1

        # Stratified selection within each segment: draw from revenue terciles
        selected = []
        for seg, n in seg_alloc.items():
            seg_products = eligible[eligible["segment"] == seg].copy()
            seg_products = seg_products.sort_values("total_revenue", ascending=False)

            n_seg = len(seg_products)
            if n_seg <= n:
                # Not enough products for stratification — take all
                selected.append(seg_products)
                continue

            # Split into 3 revenue tiers: top, middle, bottom
            third = n_seg // 3
            top_tier = seg_products.iloc[:third]
            mid_tier = seg_products.iloc[third:2*third]
            bot_tier = seg_products.iloc[2*third:]

            # Allocate across tiers: 50% top, 30% middle, 20% bottom
            # (still revenue-weighted but now with coverage guarantee)
            n_top = max(1, round(n * 0.5))
            n_mid = max(1, round(n * 0.3))
            n_bot = max(0, n - n_top - n_mid)

            # Clamp to available products in each tier
            n_top = min(n_top, len(top_tier))
            n_mid = min(n_mid, len(mid_tier))
            n_bot = min(n_bot, len(bot_tier))

            # If we're short due to clamping, redistribute
            remaining = n - n_top - n_mid - n_bot
            for tier, n_tier, n_taken in [
                (top_tier, len(top_tier), n_top),
                (mid_tier, len(mid_tier), n_mid),
                (bot_tier, len(bot_tier), n_bot),
            ]:
                if remaining <= 0:
                    break
                extra = min(remaining, n_tier - n_taken)
                if tier is top_tier:
                    n_top += extra
                elif tier is mid_tier:
                    n_mid += extra
                else:
                    n_bot += extra
                remaining -= extra

            # Select top-revenue within each tier
            tier_selected = []
            if n_top > 0:
                tier_selected.append(top_tier.nlargest(n_top, "total_revenue"))
            if n_mid > 0:
                tier_selected.append(mid_tier.nlargest(n_mid, "total_revenue"))
            if n_bot > 0:
                tier_selected.append(bot_tier.nlargest(n_bot, "total_revenue"))

            selected.append(pd.concat(tier_selected, ignore_index=True))

        result = pd.concat(selected, ignore_index=True)
        return result
