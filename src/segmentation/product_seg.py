"""Product segmentation using K-Means clustering."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class ProductSegmenter:
    """K-Means product segmentation on price-velocity-discount features.
    
    Elasticity is intentionally excluded from segmentation features
    because the bivariate (uncontrolled) elasticity is noisy and biased
    — 57% of values are near zero due to omitted variable bias from
    not controlling for display, mailer, and seasonality. Including it
    would add noise to the clustering without improving the grouping
    of products with similar demand characteristics.
    """

    FEATURE_COLS = [
        "log_price",
        "log_velocity",
        "pct_on_disc",
        "mean_discount_depth",
        "price_cv",
    ]

    def __init__(self, n_clusters: int = 12, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None

    def fit(self, product_features: pd.DataFrame) -> pd.DataFrame:
        """Fit segmentation and return product-to-segment mapping.

        Args:
            product_features: DataFrame with columns matching FEATURE_COLS
                plus PRODUCT_ID.

        Returns:
            DataFrame with PRODUCT_ID and segment columns.
        """
        df = product_features.copy()

        X = df[self.FEATURE_COLS].values
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        df["segment"] = self.kmeans.fit_predict(X_scaled)

        return df[["PRODUCT_ID", "segment"]].copy()

    def evaluate_k_range(
        self, product_features: pd.DataFrame, k_range: range = range(5, 26)
    ) -> pd.DataFrame:
        """Evaluate silhouette and inertia for a range of K values."""
        df = product_features.copy()

        X = df[self.FEATURE_COLS].values
        X_scaled = self.scaler.fit_transform(X)

        results = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels, sample_size=min(10000, len(X_scaled)))
            results.append({
                "k": k,
                "silhouette": sil,
                "inertia": km.inertia_,
            })

        return pd.DataFrame(results)
