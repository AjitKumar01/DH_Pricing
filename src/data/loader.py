"""Data loader for the Dunnhumby Complete Journey dataset."""

import os
import pandas as pd
import numpy as np


class DunnhumbyLoader:
    """Loads and preprocesses the Dunnhumby Complete Journey dataset."""

    REQUIRED_FILES = [
        "transaction_data.csv",
        "product.csv",
        "hh_demographic.csv",
        "coupon.csv",
        "coupon_redempt.csv",
        "campaign_desc.csv",
        "campaign_table.csv",
        "causal_data.csv",
    ]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._validate_data_dir()

    def _validate_data_dir(self):
        missing = [
            f for f in self.REQUIRED_FILES
            if not os.path.exists(os.path.join(self.data_dir, f))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing files in {self.data_dir}: {missing}"
            )

    def load_transactions(self) -> pd.DataFrame:
        """Load and preprocess transaction data with price/discount features."""
        txn = pd.read_csv(os.path.join(self.data_dir, "transaction_data.csv"))

        # Parse dates and derive week number
        txn["DAY"] = txn["DAY"].astype(int)
        txn["WEEK_NO"] = txn["WEEK_NO"].astype(int)

        # Make discount columns positive (they come as negative)
        for col in ["RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"]:
            if col in txn.columns:
                txn[col] = txn[col].abs()

        # Reconstruct base price and discount depth
        # Include all discount components: retail, coupon, and coupon-match
        txn["total_discount"] = (
            txn["RETAIL_DISC"] + txn["COUPON_DISC"] + txn["COUPON_MATCH_DISC"]
        )
        txn["base_price"] = txn["SALES_VALUE"] + txn["total_discount"]
        txn["avg_unit_price"] = txn["base_price"] / txn["QUANTITY"].clip(lower=1)
        txn["discount_depth"] = (
            txn["total_discount"]
            / txn["base_price"].clip(lower=0.01)
        )
        txn["discount_flag"] = (txn["discount_depth"] > 0.001).astype(int)

        # Filter out returns/refunds (negative quantities or sales)
        # and transactions with zero base price (unreliable price data)
        n_before = len(txn)
        txn = txn[(txn["QUANTITY"] > 0) & (txn["SALES_VALUE"] > 0) & (txn["base_price"] > 0)].copy()
        n_removed = n_before - len(txn)
        if n_removed > 0:
            pass  # silently filter; ~1% of data

        return txn

    def load_products(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.data_dir, "product.csv"))

    def load_demographics(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.data_dir, "hh_demographic.csv"))

    def load_causal(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.data_dir, "causal_data.csv"))

    def load_coupons(self) -> pd.DataFrame:
        """Load coupon-product associations."""
        return pd.read_csv(os.path.join(self.data_dir, "coupon.csv"))

    def load_coupon_redemptions(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.data_dir, "coupon_redempt.csv"))

    def _build_causal_features(self, product_ids=None) -> pd.DataFrame:
        """Aggregate causal data (display/mailer) to product-week level.

        The causal_data is at store-product-week granularity. We aggregate
        to product-week level by computing the fraction of stores where
        a product had display/mailer exposure.
        """
        causal = self.load_causal()

        if product_ids is not None:
            causal = causal[causal["PRODUCT_ID"].isin(product_ids)]

        # display: 0 = not on display, >0 = on display
        causal["on_display"] = (causal["display"].fillna("0").astype(str) != "0").astype(int)
        # mailer: A = on mailer, other values = not
        causal["on_mailer"] = (causal["mailer"].fillna("").astype(str) == "A").astype(int)

        # Aggregate to product-week: fraction of stores with exposure
        causal_pw = (
            causal.groupby(["PRODUCT_ID", "WEEK_NO"])
            .agg(
                display_pct=("on_display", "mean"),
                mailer_pct=("on_mailer", "mean"),
                n_stores_display=("on_display", "sum"),
                n_stores_mailer=("on_mailer", "sum"),
            )
            .reset_index()
        )
        causal_pw["has_display"] = (causal_pw["display_pct"] > 0).astype(int)
        causal_pw["has_mailer"] = (causal_pw["mailer_pct"] > 0).astype(int)

        return causal_pw

    def _build_coupon_features(self) -> pd.DataFrame:
        """Build per-product coupon availability flags."""
        coupons = self.load_coupons()
        # Flag products that have any coupon association
        coupon_products = coupons["PRODUCT_ID"].unique()
        return pd.DataFrame({
            "PRODUCT_ID": coupon_products,
            "has_coupon": 1,
        })

    def _build_store_coverage(self, product_ids=None) -> pd.DataFrame:
        """Compute number of stores carrying each product per week.

        Store coverage is a supply-side variable that captures product
        availability/distribution — a key demand driver in the retail
        literature (Bronnenberg et al. 2009).
        """
        causal = self.load_causal()
        if product_ids is not None:
            causal = causal[causal["PRODUCT_ID"].isin(product_ids)]
        store_cov = (
            causal.groupby(["PRODUCT_ID", "WEEK_NO"])
            .agg(n_stores=("STORE_ID", "nunique"))
            .reset_index()
        )
        store_cov["log_store_coverage"] = np.log(store_cov["n_stores"].clip(lower=1))
        return store_cov

    def _build_campaign_features(self) -> pd.DataFrame:
        """Build product-week campaign activity indicators.

        Links campaigns → coupons → products and determines which
        product-weeks had an active campaign targeting them.
        """
        camp_desc = pd.read_csv(os.path.join(self.data_dir, "campaign_desc.csv"))
        coupons = self.load_coupons()

        # Map campaign → product list
        campaign_products = coupons[["CAMPAIGN", "PRODUCT_ID"]].drop_duplicates()

        # Convert campaign START_DAY/END_DAY to week numbers
        camp_desc["start_week"] = (camp_desc["START_DAY"] - 1) // 7 + 1
        camp_desc["end_week"] = (camp_desc["END_DAY"] - 1) // 7 + 1

        # Expand campaigns to product-week rows
        rows = []
        for _, camp in camp_desc.iterrows():
            prods = campaign_products[campaign_products["CAMPAIGN"] == camp["CAMPAIGN"]]["PRODUCT_ID"]
            for w in range(int(camp["start_week"]), int(camp["end_week"]) + 1):
                for pid in prods:
                    rows.append({"PRODUCT_ID": pid, "WEEK_NO": w, "campaign_active": 1})

        if not rows:
            return pd.DataFrame(columns=["PRODUCT_ID", "WEEK_NO", "campaign_active"])

        camp_pw = pd.DataFrame(rows).drop_duplicates()
        # Deduplicate: a product can be in multiple campaigns in same week
        camp_pw = camp_pw.groupby(["PRODUCT_ID", "WEEK_NO"]).agg(
            campaign_active=("campaign_active", "max")
        ).reset_index()
        return camp_pw

    def _build_coupon_redemption_features(self) -> pd.DataFrame:
        """Build product-week coupon redemption counts.

        Links coupon redemptions → coupon UPCs → product IDs.
        Redemption intensity captures the pull effect of coupons
        on demand (Neslin & Clarke 1987).
        """
        redempt = self.load_coupon_redemptions()
        coupons = self.load_coupons()

        # Map coupon UPC → product ID
        coupon_product = coupons[["COUPON_UPC", "PRODUCT_ID"]].drop_duplicates()
        redempt = redempt.merge(coupon_product, on="COUPON_UPC", how="inner")

        # Convert DAY to WEEK_NO
        redempt["WEEK_NO"] = (redempt["DAY"] - 1) // 7 + 1

        # Count redemptions per product-week
        redeem_pw = (
            redempt.groupby(["PRODUCT_ID", "WEEK_NO"])
            .size()
            .rename("coupon_redemptions")
            .reset_index()
        )
        return redeem_pw

    def _build_demographic_product_features(self, txn: pd.DataFrame) -> pd.DataFrame:
        """Compute per-product buyer demographic indices.

        Aggregates household demographics of buyers to product level:
        mean income index, mean household size. These are product-level
        (not time-varying) features useful for understanding cross-sectional
        demand variation and improving shrinkage (Wan et al. 2017).
        """
        demo = self.load_demographics()

        # Map income descriptions to numeric index
        income_map = {
            "Under 15K": 1, "15-24K": 2, "25-34K": 3, "35-49K": 4,
            "50-74K": 5, "75-99K": 6, "100-124K": 7, "125-149K": 8,
            "150-174K": 9, "175-199K": 10, "200-249K": 11, "250K+": 12,
        }
        demo["income_index"] = demo["INCOME_DESC"].map(income_map).fillna(5)

        # Map household size to numeric
        demo["hh_size_num"] = pd.to_numeric(
            demo["HOUSEHOLD_SIZE_DESC"].str.replace("+", "", regex=False),
            errors="coerce"
        ).fillna(2)

        # Join transactions to demographics
        txn_demo = txn[["household_key", "PRODUCT_ID"]].drop_duplicates()
        txn_demo = txn_demo.merge(
            demo[["household_key", "income_index", "hh_size_num"]],
            on="household_key", how="inner"
        )

        # Aggregate to product level
        prod_demo = (
            txn_demo.groupby("PRODUCT_ID")
            .agg(
                buyer_income_index=("income_index", "mean"),
                buyer_hh_size=("hh_size_num", "mean"),
                n_unique_buyers=("household_key", "nunique"),
            )
            .reset_index()
        )
        return prod_demo

    def build_product_weekly_panel(
        self, txn: pd.DataFrame, min_weeks: int = 30, min_price_cv: float = 0.03,
        stable_start: int = 18, stable_end: int = 102,
        include_causal: bool = True,
    ) -> pd.DataFrame:
        """Build a product-week level panel from transactions.

        Integrates promotional causal data (display/mailer exposure)
        and coupon availability when include_causal=True.

        Filters to stable period and products with sufficient temporal
        coverage and price variation for demand model estimation.
        """
        stable = txn[
            (txn["WEEK_NO"] >= stable_start) & (txn["WEEK_NO"] <= stable_end)
        ].copy()

        # Aggregate to product-week level
        panel = (
            stable.groupby(["PRODUCT_ID", "WEEK_NO"])
            .agg(
                quantity=("QUANTITY", "sum"),
                revenue=("SALES_VALUE", "sum"),
                total_base_value=("base_price", "sum"),
                discount_depth=("discount_depth", "mean"),
                discount_flag=("discount_flag", "max"),
                n_transactions=("QUANTITY", "count"),
            )
            .reset_index()
        )

        # Compute quantity-weighted average unit price
        panel["avg_price"] = panel["total_base_value"] / panel["quantity"].clip(lower=1)
        panel = panel.drop(columns=["total_base_value"])

        # Remove product-weeks with zero or negative quantity
        panel = panel[panel["quantity"] > 0].copy()

        # Remove extreme quantity outliers at the product level:
        # drop product-weeks where log(quantity) is > 4 std from
        # the product's own mean (catches weight/volume products
        # with millions of "units" that distort the regression).
        log_qty = np.log(panel["quantity"].clip(lower=0.5))
        product_stats = log_qty.groupby(panel["PRODUCT_ID"]).agg(["mean", "std"])
        panel = panel.merge(
            product_stats.rename(columns={"mean": "lq_mean", "std": "lq_std"}),
            left_on="PRODUCT_ID", right_index=True, how="left",
        )
        log_qty_merged = np.log(panel["quantity"].clip(lower=0.5))
        z_score = (log_qty_merged - panel["lq_mean"]) / panel["lq_std"].clip(lower=1e-6)
        panel = panel[z_score.abs() <= 4].copy()

        # Also apply a global cap: remove products whose median
        # log-quantity is > P99.5 of all product medians
        # (catches weight/volume-based products like product 6534178)
        log_qty = np.log(panel["quantity"].clip(lower=0.5))
        product_med_lq = log_qty.groupby(panel["PRODUCT_ID"]).median()
        global_cap = product_med_lq.quantile(0.995)
        extreme_products = set(product_med_lq[product_med_lq > global_cap].index)
        panel = panel[~panel["PRODUCT_ID"].isin(extreme_products)].copy()
        panel = panel.drop(columns=["lq_mean", "lq_std"], errors="ignore")

        # Filter by temporal coverage
        weeks_per_product = panel.groupby("PRODUCT_ID")["WEEK_NO"].nunique()
        eligible_products = weeks_per_product[weeks_per_product >= min_weeks].index

        # Filter by price variation
        price_cv = panel.groupby("PRODUCT_ID")["avg_price"].agg(
            lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        )
        price_vary_products = price_cv[price_cv >= min_price_cv].index

        eligible = set(eligible_products) & set(price_vary_products)
        panel = panel[panel["PRODUCT_ID"].isin(eligible)].copy()

        if include_causal:
            # Merge causal data (display/mailer)
            causal_pw = self._build_causal_features(
                product_ids=set(panel["PRODUCT_ID"].unique())
            )
            panel = panel.merge(
                causal_pw[["PRODUCT_ID", "WEEK_NO", "display_pct",
                           "mailer_pct", "has_display", "has_mailer"]],
                on=["PRODUCT_ID", "WEEK_NO"],
                how="left",
            )
            panel["display_pct"] = panel["display_pct"].fillna(0)
            panel["mailer_pct"] = panel["mailer_pct"].fillna(0)
            panel["has_display"] = panel["has_display"].fillna(0).astype(int)
            panel["has_mailer"] = panel["has_mailer"].fillna(0).astype(int)

            # Merge coupon availability
            coupon_df = self._build_coupon_features()
            panel = panel.merge(coupon_df, on="PRODUCT_ID", how="left")
            panel["has_coupon"] = panel["has_coupon"].fillna(0).astype(int)

            # Merge store coverage (supply-side distribution variable)
            store_cov = self._build_store_coverage(
                product_ids=set(panel["PRODUCT_ID"].unique())
            )
            panel = panel.merge(
                store_cov[["PRODUCT_ID", "WEEK_NO", "n_stores", "log_store_coverage"]],
                on=["PRODUCT_ID", "WEEK_NO"],
                how="left",
            )
            # Fill missing store coverage with product-level median
            med_stores = panel.groupby("PRODUCT_ID")["n_stores"].transform("median")
            panel["n_stores"] = panel["n_stores"].fillna(med_stores).fillna(1)
            panel["log_store_coverage"] = np.log(panel["n_stores"].clip(lower=1))

            # Merge campaign activity (time-varying promotional treatment)
            camp_pw = self._build_campaign_features()
            panel = panel.merge(
                camp_pw[["PRODUCT_ID", "WEEK_NO", "campaign_active"]],
                on=["PRODUCT_ID", "WEEK_NO"],
                how="left",
            )
            panel["campaign_active"] = panel["campaign_active"].fillna(0).astype(int)

            # Merge coupon redemption counts
            redeem_pw = self._build_coupon_redemption_features()
            panel = panel.merge(
                redeem_pw[["PRODUCT_ID", "WEEK_NO", "coupon_redemptions"]],
                on=["PRODUCT_ID", "WEEK_NO"],
                how="left",
            )
            panel["coupon_redemptions"] = panel["coupon_redemptions"].fillna(0)

            # Merge product category (DEPARTMENT) from product.csv
            products = self.load_products()
            panel = panel.merge(
                products[["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"]],
                on="PRODUCT_ID",
                how="left",
            )

            # Merge buyer demographics (product-level aggregates)
            demo_feats = self._build_demographic_product_features(txn)
            panel = panel.merge(demo_feats, on="PRODUCT_ID", how="left")
            panel["buyer_income_index"] = panel["buyer_income_index"].fillna(
                panel["buyer_income_index"].median()
            )
            panel["buyer_hh_size"] = panel["buyer_hh_size"].fillna(
                panel["buyer_hh_size"].median()
            )
            panel["n_unique_buyers"] = panel["n_unique_buyers"].fillna(0)

        return panel

    def compute_product_features(self, txn: pd.DataFrame) -> pd.DataFrame:
        """Compute per-product features for segmentation.

        Uses the same stable period (week >= 18) as the panel to avoid
        feature-period mismatch in representative product selection.
        """
        stable = txn[(txn["WEEK_NO"] >= 18) & (txn["WEEK_NO"] <= 102)].copy()

        n_weeks_total = stable["WEEK_NO"].nunique()

        feats = (
            stable.groupby("PRODUCT_ID")
            .agg(
                total_revenue=("SALES_VALUE", "sum"),
                total_qty=("QUANTITY", "sum"),
                n_transactions=("QUANTITY", "count"),
                mean_price=("avg_unit_price", "mean"),
                std_price=("avg_unit_price", "std"),
                pct_on_disc=(
                    "discount_flag",
                    "mean",
                ),
                mean_discount_depth=("discount_depth", "mean"),
                n_weeks=("WEEK_NO", "nunique"),
            )
            .reset_index()
        )

        feats["weekly_velocity"] = feats["total_qty"] / n_weeks_total
        feats["price_cv"] = feats["std_price"] / feats["mean_price"].clip(lower=0.01)
        feats["log_price"] = np.log(feats["mean_price"].clip(lower=0.01))
        feats["log_velocity"] = np.log(feats["weekly_velocity"].clip(lower=0.01))

        return feats
