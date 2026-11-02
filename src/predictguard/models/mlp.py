"""MLP architecture and single-window RUL inference."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class MLPRegressor(nn.Module):
    """Regress RUL from a flattened, scaled sensor window."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
        window_size: int | None = None,
        feature_count: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim debe ser un entero positivo.")
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims debe contener enteros positivos.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout debe estar en el intervalo [0, 1).")
        if (window_size is None) != (feature_count is None):
            raise ValueError("window_size y feature_count deben configurarse juntos.")
        if window_size is not None and (
            window_size < 1
            or feature_count is None
            or feature_count < 1
            or window_size * feature_count != input_dim
        ):
            raise ValueError("La forma de la ventana debe coincidir con input_dim.")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = width
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.window_shape = (
            (window_size, feature_count) if window_size is not None else None
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"Se esperaba un tensor (batch, {self.input_dim}); "
                f"se recibió {tuple(features.shape)}."
            )
        return self.network(features).squeeze(-1)


def predict_rul(
    window: np.ndarray,
    model: MLPRegressor,
    device: torch.device | str | None = None,
) -> float:
    """Predict nonnegative RUL for one already-scaled time window."""
    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("window debe tener forma (ciclos, features).")
    if model.window_shape is not None and values.shape != model.window_shape:
        raise ValueError(
            f"window debe tener forma {model.window_shape}; "
            f"se recibió {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("window solo puede contener valores finitos.")
    if values.size != model.input_dim:
        raise ValueError(
            f"window contiene {values.size} valores; "
            f"el modelo espera {model.input_dim}."
        )

    model_device = next(model.parameters()).device
    inference_device = torch.device(device) if device is not None else model_device
    if inference_device.type != model_device.type or (
        inference_device.index is not None
        and inference_device.index != model_device.index
    ):
        raise ValueError("model debe estar en el dispositivo de inferencia indicado.")
    inference_device = model_device
    model.eval()
    tensor = torch.from_numpy(values.reshape(1, -1)).to(inference_device)
    with torch.inference_mode():
        prediction = model(tensor).clamp_min(0.0).item()
    return float(prediction)


def load_mlp_checkpoint(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> tuple[MLPRegressor, dict[str, Any]]:
    """Load a saved MLP checkpoint and its metadata for inference."""
    inference_device = torch.device(device)
    payload = torch.load(
        checkpoint_path,
        map_location=inference_device,
        weights_only=True,
    )
    model_config = payload["model"]
    feature_columns = payload["feature_columns"]
    model = MLPRegressor(
        input_dim=int(payload["input_dim"]),
        hidden_dims=model_config["hidden_dims"],
        dropout=float(model_config["dropout"]),
        window_size=int(payload["window_size"]),
        feature_count=len(feature_columns),
    ).to(inference_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
