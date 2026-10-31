import numpy as np
import pytest

from predictguard.metrics import nasa_phm_score_per_sample, regression_metrics


def test_perfect_predictions_have_zero_errors_and_score():
    actual = np.array([0, 5, 20], dtype=float)

    metrics = regression_metrics(actual, actual.copy())

    assert metrics == {
        "sample_count": 3,
        "mae_cycles": 0.0,
        "rmse_cycles": 0.0,
        "nasa_phm_score_sum": 0.0,
        "nasa_phm_score_mean": 0.0,
        "bias_cycles": 0.0,
        "absolute_error_p50_cycles": 0.0,
        "absolute_error_p90_cycles": 0.0,
    }


def test_nasa_phm_score_uses_documented_error_sign_and_asymmetry():
    actual = np.array([10, 10, 10], dtype=float)
    predicted = np.array([20, 0, 10], dtype=float)

    penalties = nasa_phm_score_per_sample(actual, predicted)

    np.testing.assert_allclose(
        penalties,
        [np.exp(1) - 1, np.exp(10 / 13) - 1, 0],
    )
    assert penalties[0] > penalties[1]


def test_regression_metrics_report_signed_bias_and_error_percentiles():
    metrics = regression_metrics([10, 10, 10, 10], [12, 8, 10, 14])

    assert metrics["mae_cycles"] == 2.0
    assert metrics["rmse_cycles"] == pytest.approx(np.sqrt(6))
    assert metrics["bias_cycles"] == 1.0
    assert metrics["absolute_error_p50_cycles"] == 2.0
    assert metrics["absolute_error_p90_cycles"] == pytest.approx(3.4)
    assert metrics["nasa_phm_score_sum"] == pytest.approx(
        (np.exp(0.2) - 1) + (np.exp(2 / 13) - 1) + (np.exp(0.4) - 1)
    )


@pytest.mark.parametrize(
    "actual, predicted, message",
    [
        ([], [], "vectores vacíos"),
        ([1, 2], [1], "misma cantidad"),
        ([[1]], [[1]], "unidimensionales"),
        ([1, np.nan], [1, 1], "valores finitos"),
        ([-1], [1], "no negativos"),
    ],
)
def test_metrics_reject_invalid_inputs(actual, predicted, message):
    with pytest.raises(ValueError, match=message):
        regression_metrics(actual, predicted)


def test_nasa_score_rejects_numeric_overflow():
    with pytest.raises(OverflowError, match="rango numérico"):
        nasa_phm_score_per_sample([0], [1e6])
