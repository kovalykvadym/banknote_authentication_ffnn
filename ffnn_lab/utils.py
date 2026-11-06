import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ffnn_lab import config


class Timer:
    """Context manager that measures elapsed wall-clock time.

    After the `with` block ends, `elapsed` holds the duration in seconds,
    measured with `time.perf_counter`.
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, error_type, error_val, traceback) -> None:
        self.elapsed = time.perf_counter() - self._start


def measure_mean_time(
    func: Callable[[], object],
    repeats: int = config.TIMING_REPEATS,
) -> float:
    """Return the mean execution time of `func` in seconds.

    `func` is called once before measuring (warm-up), then `repeats` times
    with timing. It must be safe to call repeatedly, so use it for fast
    operations such as prediction, not for training. Training is timed
    with a single run of `Timer`.

    Raises:
        ValueError: If `repeats` is less than 1.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")

    func()

    total = 0.0
    for _ in range(repeats):
        with Timer() as timer:
            func()
        total += timer.elapsed

    return total / repeats


def outputs_to_labels(outputs: np.ndarray, bipolar: bool) -> np.ndarray:
    """Convert model outputs to class labels 0 or 1.

    The threshold is 0.0 for bipolar outputs (range -1..1) and 0.5 for
    unipolar outputs (range 0..1). The comparison is strict, so an output
    exactly equal to the threshold gives class 0. Outputs of shape (n, 1)
    are flattened.
    """
    threshold = 0.0 if bipolar else 0.5
    flat = np.ravel(outputs)
    return (flat > threshold).astype(int)


def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the share of misclassified samples (0.0 to 1.0).

    Both arrays must contain labels 0 and 1 only.

    Raises:
        ValueError: If the arrays are empty, differ in length, or contain
            values other than 0 and 1.
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    if y_true.size == 0:
        raise ValueError("Cannot compute error for empty arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.size}, y_pred has {y_pred.size}"
        )

    if not (np.isin(y_true, (0, 1)).all() and np.isin(y_pred, (0, 1)).all()):
        raise ValueError(
            "Labels must be 0 or 1. Convert bipolar labels (-1/+1) "
            "back to 0/1 before computing the error."
        )

    return float(np.mean(y_true != y_pred))


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of one training run.

    Attributes:
        epochs_run: Number of epochs actually performed.
        stopped_early: True if training stopped before `max_epochs`.
        training_time: Training time in seconds.
        loss_history: Training loss after each epoch.
    """

    epochs_run: int
    stopped_early: bool
    training_time: float
    loss_history: list[float]
