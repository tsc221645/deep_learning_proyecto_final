"""Regression metrics used by the RUL experiment protocol."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validated_pair(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true y y_pred deben ser vectores unidimensionales.")
    if actual.shape != predicted.shape:
        raise ValueError(
            "y_true y y_pred deben tener la misma cantidad de elementos."
        )
    if actual.size == 0:
        raise ValueError("No se pueden calcular métricas con vectores vacíos.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true y y_pred solo pueden contener valores finitos.")
    if (actual < 0).any():
        raise ValueError("y_true debe contener valores RUL no negativos.")
    return actual, predicted


def nasa_phm_score_per_sample(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return PHM08 asymmetric penalties with d = prediction - actual."""
    actual, predicted = _validated_pair(y_true, y_pred)
    try:
        with np.errstate(over="raise", invalid="raise"):
            error = predicted - actual
            penalties = np.empty_like(error)
            underestimated = error < 0
            penalties[underestimated] = (
                np.exp(-error[underestimated] / 13.0) - 1.0
            )
            penalties[~underestimated] = (
                np.exp(error[~underestimated] / 10.0) - 1.0
            )
    except FloatingPointError as error:
        raise OverflowError(
            "El error de predicción excede el rango numérico del score PHM08."
        ) from error
    if not np.isfinite(penalties).all():
        raise OverflowError("El score PHM08 produjo valores no finitos.")
    return penalties


def regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    """Calculate regression errors in cycles and the PHM08 asymmetric score."""
    actual, predicted = _validated_pair(y_true, y_pred)
    try:
        with np.errstate(over="raise", invalid="raise"):
            errors = predicted - actual
            absolute_errors = np.abs(errors)
            squared_errors = np.square(errors)
    except FloatingPointError as error:
        raise OverflowError("El error de regresión excede el rango numérico.") from error

    penalties = nasa_phm_score_per_sample(actual, predicted)
    score_sum = float(np.sum(penalties))
    if not np.isfinite(score_sum):
        raise OverflowError("La suma del score PHM08 excede el rango numérico.")

    return {
        "sample_count": int(actual.size),
        "mae_cycles": float(np.mean(absolute_errors)),
        "rmse_cycles": float(np.sqrt(np.mean(squared_errors))),
        "nasa_phm_score_sum": score_sum,
        "nasa_phm_score_mean": float(np.mean(penalties)),
        "bias_cycles": float(np.mean(errors)),
        "absolute_error_p50_cycles": float(np.quantile(absolute_errors, 0.50)),
        "absolute_error_p90_cycles": float(np.quantile(absolute_errors, 0.90)),
    }
