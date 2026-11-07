"""Evaluation of a trained model on one data set.

All models (single-layer perceptron, multilayer perceptron, RBF network)
are evaluated by the same function, so their errors and classification
times are directly comparable.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ffnn_lab import config
from ffnn_lab.utils import classification_error, measure_mean_time


class Classifier(Protocol):
    """Any model that can produce raw outputs and class labels."""

    def predict_outputs(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Evaluation:
    """Result of evaluating a model on one data set.

    Attributes:
        outputs: Raw model outputs, shape (n,).
        labels: Predicted classes 0 or 1, shape (n,).
        error: Share of misclassified samples (0.0 to 1.0).
        classification_time: Mean time in seconds to classify the whole set.
    """

    outputs: np.ndarray
    labels: np.ndarray
    error: float
    classification_time: float


def evaluate(
    model: Classifier,
    X: np.ndarray,
    y: np.ndarray,
    repeats: int = config.TIMING_REPEATS,
) -> Evaluation:
    """Evaluate a trained model on features `X` and labels `y` (0/1).

    The classification time is the mean of `repeats` runs of `model.predict`
    on the whole set, after one warm-up run (see `measure_mean_time`).

    Raises:
        ValueError: If `repeats` is less than 1 or the labels are not 0/1.
    """
    outputs = model.predict_outputs(X)
    labels = model.predict(X)
    error = classification_error(y, labels)
    classification_time = measure_mean_time(lambda: model.predict(X), repeats)

    return Evaluation(
        outputs=outputs,
        labels=labels,
        error=error,
        classification_time=classification_time,
    )
