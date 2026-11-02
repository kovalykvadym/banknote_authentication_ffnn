import time
from collections.abc import Callable

import numpy as np

from ffnn_lab import config


class Timer:
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
    threshold = 0.0 if bipolar else 0.5
    flat = np.ravel(outputs)
    return (flat > threshold).astype(int)


def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    if y_true.size == 0:
        raise ValueError("Cannot compute error for empty arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.size}, y_pred has {y_pred.size}"
        )

    return float(np.mean(y_true != y_pred))
