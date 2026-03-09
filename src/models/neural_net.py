"""Neural network demand model using PyTorch."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _DemandNet(nn.Module):
    """Simple feedforward network for demand prediction."""

    def __init__(self, input_dim: int = 3, hidden_dims: tuple = (32, 16)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeuralNetDemandModel:
    """Per-product neural network demand model.

    Uses a small feedforward network to capture nonlinear
    price-demand relationships. Particularly useful for products
    with enough data to benefit from flexible function approximation.

    References:
        - Hartford et al. (2017), "Deep IV: A Flexible Approach for
          Counterfactual Prediction"
        - Gabel & Timoshenko (2022), "Product Choice with Large
          Assortments: A Scalable Deep-Learning Model"
    """

    def __init__(self, hidden_dims: tuple = (32, 16), epochs: int = 200,
                 lr: float = 0.005, batch_size: int = 32, random_state: int = 42):
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_dims = {}  # track per-product feature dimension

    def _build_features(self, grp: pd.DataFrame) -> np.ndarray:
        cols = [
            np.log(grp["avg_price"].clip(lower=0.01).values),
            grp["discount_depth"].values,
        ]
        # Prefer continuous exposure fractions over binary indicators
        if "display_pct" in grp.columns:
            cols.append(grp["display_pct"].values)
        elif "has_display" in grp.columns:
            cols.append(grp["has_display"].values)
        if "mailer_pct" in grp.columns:
            cols.append(grp["mailer_pct"].values)
        elif "has_mailer" in grp.columns:
            cols.append(grp["has_mailer"].values)
        if "log_store_coverage" in grp.columns:
            cols.append(grp["log_store_coverage"].values)
        if "campaign_active" in grp.columns:
            cols.append(grp["campaign_active"].values)
        return np.column_stack(cols)

    def fit(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Fit per-product neural network models."""
        torch.manual_seed(self.random_state)
        results = []

        for pid, grp in panel.groupby("PRODUCT_ID"):
            if len(grp) < 15:
                continue

            y = np.log(grp["quantity"].clip(lower=0.5).values).astype(np.float32)
            X = self._build_features(grp).astype(np.float32)
            self.feature_dims[pid] = X.shape[1]

            # Standardize features
            x_mean, x_std = X.mean(0), X.std(0).clip(min=1e-6)
            X_scaled = (X - x_mean) / x_std
            self.scalers[pid] = (x_mean, x_std)

            X_t = torch.from_numpy(X_scaled)
            y_t = torch.from_numpy(y)
            dataset = TensorDataset(X_t, y_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            model = _DemandNet(input_dim=X.shape[1], hidden_dims=self.hidden_dims)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()

            model.train()
            for _ in range(self.epochs):
                for xb, yb in loader:
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred = model(X_t).numpy()

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Finite-difference elasticity
            X_up = X_scaled.copy()
            X_up[:, 0] += 0.01 / x_std[0]
            with torch.no_grad():
                y_up = model(torch.from_numpy(X_up)).numpy()
            elasticity = (y_up - y_pred).mean() / 0.01

            # Discount effect: marginal effect of 1pp discount depth
            X_no = X_scaled.copy()
            X_no[:, 1] = (0 - x_mean[1]) / x_std[1]
            X_yes = X_scaled.copy()
            X_yes[:, 1] = (0.15 - x_mean[1]) / x_std[1]
            with torch.no_grad():
                disc_effect = (
                    model(torch.from_numpy(X_yes)).numpy()
                    - model(torch.from_numpy(X_no)).numpy()
                ).mean() / 0.15

            residuals = y - y_pred
            self.models[pid] = model

            results.append({
                "PRODUCT_ID": pid,
                "intercept": y.mean(),
                "elasticity": float(elasticity),
                "disc_effect": float(disc_effect),
                "r2": float(r2),
                "residual_std": float(residuals.std()),
                "n_obs": len(y),
            })

        return pd.DataFrame(results)

    def predict(self, product_id, log_price: float, discount_depth: float = 0.0,
                has_display: int = 0, has_mailer: int = 0,
                log_store_coverage: float = 0.0, campaign_active: int = 0) -> float:
        x_mean, x_std = self.scalers[product_id]
        features = [log_price, discount_depth, has_display, has_mailer,
                     log_store_coverage, campaign_active]
        n_feats = self.feature_dims.get(product_id, len(x_mean))
        X = np.array([features[:n_feats]], dtype=np.float32)
        X_scaled = (X - x_mean) / x_std
        self.models[product_id].eval()
        with torch.no_grad():
            return self.models[product_id](torch.from_numpy(X_scaled)).item()

    def predict_level(self, product_id, price: float, discount_depth: float = 0.0,
                      has_display: int = 0, has_mailer: int = 0,
                      log_store_coverage: float = 0.0, campaign_active: int = 0) -> float:
        log_pred = self.predict(product_id, np.log(max(price, 0.01)),
                                discount_depth, has_display, has_mailer,
                                log_store_coverage, campaign_active)
        return np.exp(log_pred)
