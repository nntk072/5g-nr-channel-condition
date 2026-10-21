"""CORAL representation alignment and portable checkpoint loading."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from models.network import build_domain_mlp
from sklearn.preprocessing import StandardScaler


def load_coral_checkpoint(path, device: str = "cpu"):
    """Reload model/scaling parameters without pickle or training data."""
    import torch

    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "loslab.coral.v1":
        raise ValueError("Invalid checkpoint format")
    network = build_domain_mlp(payload["n_features"]).to(device)
    network.load_state_dict(payload["state_dict"])
    network.eval()
    center = payload["mean"].to(device)
    scale = payload["scale"].to(device)

    class LoadedPredictor:
        threshold = float(payload["threshold"])
        include_phase = bool(payload["include_phase"])

        def predict_proba(self, x):
            value = torch.tensor(x, dtype=torch.float32, device=device)
            with torch.no_grad():
                return torch.softmax(network((value - center) / scale)[0], dim=-1).cpu().numpy()

    return LoadedPredictor()


def fit_coral_mlp(
    x: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    penalty: float = 0.1,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_per_domain: int = 48,
    device: str = "cpu",
):
    """Fit same-capacity MLP with a second-order CORAL alignment penalty."""
    import torch
    from torch import nn

    if penalty < 0:
        raise ValueError("penalty must be >=0")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(x)
    xx = torch.tensor(scaler.transform(x), dtype=torch.float32, device=device)
    yy = torch.tensor(y, dtype=torch.long, device=device)
    xv = torch.tensor(scaler.transform(x_val), dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)
    domains = np.asarray(domains, dtype=str)
    domain_indices = [np.flatnonzero(domains == domain) for domain in sorted(set(domains))]
    if penalty > 0 and len(domain_indices) < 2:
        raise ValueError("CORAL needs >=2 training domains")

    model = build_domain_mlp(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_loss = float("inf")
    patience = 0
    for epoch in range(epochs):
        model.train()
        subset = np.concatenate(
            [
                rng.choice(indices, size=batch_per_domain, replace=len(indices) < batch_per_domain)
                for indices in domain_indices
            ]
        )
        subset = subset[rng.permutation(len(subset))]
        scores, embedding = model(xx[subset])
        objective = loss_fn(scores, yy[subset])
        if penalty > 0:
            selected_domains = domains[subset]
            covariances = []
            for domain in np.unique(selected_domains):
                z = embedding[torch.tensor(selected_domains == domain, device=device)]
                if len(z) >= 2:
                    z = z - z.mean(dim=0, keepdim=True)
                    covariances.append(z.T @ z / (len(z) - 1))
            if len(covariances) >= 2:
                alignment = sum(
                    ((covariances[i] - covariances[j]) ** 2).mean()
                    for i in range(len(covariances))
                    for j in range(i + 1, len(covariances))
                )
                objective = objective + penalty * alignment
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(xv)[0], yv).item())
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= 25:
            break
    if best_state is None:
        raise RuntimeError("Could not train CORAL MLP")
    model.load_state_dict(best_state)
    model.eval()

    class Predictor:
        def save(self, path, *, threshold: float, include_phase: bool = False):
            torch.save(
                {
                    "format": "loslab.coral.v1",
                    "n_features": x.shape[1],
                    "state_dict": model.state_dict(),
                    "mean": torch.tensor(scaler.mean_, dtype=torch.float32),
                    "scale": torch.tensor(scaler.scale_, dtype=torch.float32),
                    "threshold": float(threshold),
                    "include_phase": bool(include_phase),
                },
                str(path),
            )

        def predict_proba(self, value):
            with torch.no_grad():
                tensor = torch.tensor(scaler.transform(value), dtype=torch.float32, device=device)
                return torch.softmax(model(tensor)[0], dim=-1).cpu().numpy()

    return Predictor(), {
        "epoch_count": epoch + 1,
        "best_validation_ce": best_loss,
        "penalty": penalty,
        "train_domains": list(sorted(set(domains))),
    }
