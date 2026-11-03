"""Activation functions and their derivatives.

An activation function turns the weighted sum `z` of a neuron into its output.
Training by gradient descent also needs the derivative of the function.
All derivatives here take `z` (the value before activation), not the output.

The threshold function has no useful derivative, so it can be trained
only with the Rosenblatt rule, which does not use gradients.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Limit for the sigmoid argument: exp() of larger values overflows,
# while the sigmoid there is already indistinguishable from 0 or 1.
SIGMOID_CLIP = 500.0

ArrayFunction = Callable[[np.ndarray], np.ndarray]


def threshold(z: np.ndarray) -> np.ndarray:
    """Hard limit: 1.0 where z >= 0, otherwise 0.0."""
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0.0, 1.0, 0.0)


def linear(z: np.ndarray) -> np.ndarray:
    """Identity: the output equals the input."""
    return np.asarray(z, dtype=float).copy()


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function 1 / (1 + exp(-z)), output in (0, 1)."""
    z = np.clip(np.asarray(z, dtype=float), -SIGMOID_CLIP, SIGMOID_CLIP)
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent, output in (-1, 1)."""
    return np.tanh(np.asarray(z, dtype=float))


def relu(z: np.ndarray) -> np.ndarray:
    """Rectified linear unit: max(0, z)."""
    return np.maximum(0.0, np.asarray(z, dtype=float))


def linear_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of the identity: 1 everywhere."""
    return np.ones_like(np.asarray(z, dtype=float))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid: s * (1 - s), where s = sigmoid(z)."""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 - tanh(z) ** 2."""
    t = tanh(z)
    return 1.0 - t**2


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where z > 0, otherwise 0 (including z = 0)."""
    return np.where(np.asarray(z, dtype=float) > 0.0, 1.0, 0.0)


@dataclass(frozen=True)
class Activation:
    """An activation function bundled with its derivative.

    Attributes:
        name: Name used in the configuration.
        forward: Function mapping z to the neuron output.
        derivative: Function mapping z to the derivative of `forward`,
            or None if the function cannot be trained by gradients.
        bipolar: True if the output range is -1..1, so the targets must be
            encoded as -1/+1 and the decision threshold is 0. Otherwise, the
            targets are 0/1 and the threshold is 0.5.
    """

    name: str
    forward: ArrayFunction
    derivative: ArrayFunction | None
    bipolar: bool


ACTIVATIONS: dict[str, Activation] = {
    "threshold": Activation("threshold", threshold, None, bipolar=False),
    "linear": Activation("linear", linear, linear_derivative, bipolar=False),
    "sigmoid": Activation("sigmoid", sigmoid, sigmoid_derivative, bipolar=False),
    "tanh": Activation("tanh", tanh, tanh_derivative, bipolar=True),
    "relu": Activation("relu", relu, relu_derivative, bipolar=False),
}


def get_activation(name: str) -> Activation:
    """Return the activation registered under `name`.

    Raises:
        ValueError: If the name is unknown.
    """
    try:
        return ACTIVATIONS[name]
    except KeyError:
        available = ", ".join(ACTIVATIONS)
        raise ValueError(
            f"Unknown activation '{name}'. Available: {available}"
        ) from None
