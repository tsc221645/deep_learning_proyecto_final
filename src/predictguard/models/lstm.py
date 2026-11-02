"""LSTM regression on scaled C-MAPSS windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


class LSTMRegressor(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        if feature_count < 1 or hidden_size < 1 or num_layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Parámetros de LSTM inválidos.")
        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(feature_count, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, windows: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[2] != self.feature_count:
            raise ValueError("Se esperaba (batch, ciclos, features).")
        if lengths is None:
            _, (hidden, _) = self.lstm(windows)
        else:
            if lengths.shape != (windows.shape[0],) or (lengths < 1).any() or (lengths > windows.shape[1]).any():
                raise ValueError("Longitudes de secuencia inválidas.")
            if bool((lengths == windows.shape[1]).all()):
                _, (hidden, _) = self.lstm(windows)
            else:
                # The preprocessing pipeline pads on the left. Remove that prefix
                # before packing so the LSTM never treats repeated rows as history.
                sequences = [row[-int(length):] for row, length in zip(windows, lengths)]
                right_padded = pad_sequence(sequences, batch_first=True)
                packed = pack_padded_sequence(right_padded, lengths.cpu(),
                                              batch_first=True, enforce_sorted=False)
                _, (hidden, _) = self.lstm(packed)
        return self.head(hidden[-1]).squeeze(-1)


def predict_rul(window: np.ndarray, model: LSTMRegressor,
                length: int | None = None) -> float:
    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != model.feature_count or not np.isfinite(values).all():
        raise ValueError("Ventana LSTM inválida.")
    length = values.shape[0] if length is None else length
    if not 1 <= length <= values.shape[0]:
        raise ValueError("Longitud válida fuera de la ventana.")
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(values[None]).to(device),
                           torch.tensor([length], device=device)).clamp_min(0).item()
    return float(prediction)


def load_lstm_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[LSTMRegressor, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    config = payload["model"]
    model = LSTMRegressor(len(payload["feature_columns"]),
                          int(config["hidden_size"]), int(config["num_layers"]),
                          float(config["dropout"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
