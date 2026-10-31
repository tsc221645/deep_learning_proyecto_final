import numpy as np
import pytest

from predictguard.data.windows import build_windows


def test_all_windows_preserve_unit_and_cycle_boundaries():
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    unit_ids = np.array([1, 1, 1, 2, 2, 2])
    cycles = np.array([1, 2, 3, 1, 2, 3])
    labels = np.array([2, 1, 0, 2, 1, 0], dtype=np.float32)

    windows = build_windows(
        features,
        labels,
        unit_ids,
        cycles,
        window_size=2,
        mode="all",
    )

    assert windows["X"].shape == (4, 2, 2)
    np.testing.assert_array_equal(windows["unit_ids"], [1, 1, 2, 2])
    np.testing.assert_array_equal(windows["end_cycles"], [2, 3, 2, 3])
    np.testing.assert_array_equal(windows["y"], [1, 0, 1, 0])
    np.testing.assert_array_equal(windows["X_flat"], windows["X"].reshape(4, -1))


def test_last_window_left_pads_short_trajectory_without_future_data():
    features = np.array([[10, 11], [20, 21]], dtype=np.float32)
    windows = build_windows(
        features,
        labels=None,
        unit_ids=np.array([7, 7]),
        cycles=np.array([1, 2]),
        window_size=4,
        mode="last",
        labels_at_end={7: 5},
    )

    np.testing.assert_array_equal(
        windows["X"][0],
        [[10, 11], [10, 11], [10, 11], [20, 21]],
    )
    np.testing.assert_array_equal(windows["lengths"], [2])
    np.testing.assert_array_equal(windows["end_cycles"], [2])
    np.testing.assert_array_equal(windows["y"], [5])


@pytest.mark.parametrize(
    "unit_ids, cycles, message",
    [
        ([1, 2, 1], [1, 1, 2], "ordenados por unit_id"),
        ([1, 1, 1], [1, 2, 4], "ciclos consecutivos"),
    ],
)
def test_windows_reject_unsorted_units_and_gapped_cycles(
    unit_ids, cycles, message
):
    with pytest.raises(ValueError, match=message):
        build_windows(
            np.ones((3, 1), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.asarray(unit_ids),
            np.asarray(cycles),
            window_size=2,
            mode="all",
        )


def test_last_window_requires_explicit_terminal_labels():
    with pytest.raises(ValueError, match="labels_at_end"):
        build_windows(
            np.ones((2, 1), dtype=np.float32),
            labels=None,
            unit_ids=np.array([1, 1]),
            cycles=np.array([1, 2]),
            window_size=2,
            mode="last",
        )


def test_training_windows_reject_negative_labels():
    with pytest.raises(ValueError, match="no pueden ser negativas"):
        build_windows(
            np.ones((2, 1), dtype=np.float32),
            np.array([0, -1], dtype=np.float32),
            np.array([1, 1]),
            np.array([1, 2]),
            window_size=1,
            mode="all",
        )
