"""Optimizers: rules that turn gradients into weight corrections.

The model computes the gradient of the loss for every parameter. The optimizer
decides how to change the parameters using these gradients. All optimizers
share one interface: `update(params, grads)` changes the parameter arrays
in place.
"""

import numpy as np

from ffnn_lab import config

RMSPROP_RHO = 0.9
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
EPSILON = 1e-8


class Optimizer:
    """Base class: stores the learning rate and defines the interface.

    Args:
        learning_rate: step size, must be positive.

    Raises:
        ValueError: If the learning rate is not positive.
    """

    def __init__(self, learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        self.learning_rate = learning_rate

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Change every array in `params` in place using `grads`.

        `params[i]` and `grads[i]` must have the same shape.
        """
        raise NotImplementedError

    @staticmethod
    def _check_shapes(params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Check that parameters and gradients match one to one."""
        if len(params) != len(grads):
            raise ValueError(f"Got {len(params)} parameters but {len(grads)} gradients")
        for index, (param, grad) in enumerate(zip(params, grads)):
            if param.shape != grad.shape:
                raise ValueError(
                    f"Shape mismatch at index {index}: parameter {param.shape}, "
                    f"gradient {grad.shape}"
                )


class SGD(Optimizer):
    """Plain gradient descent: p -= lr * g."""

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self._check_shapes(params, grads)
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad


class SGDMomentum(Optimizer):
    """Gradient descent with momentum.

    v = momentum * v - lr * g
    p += v

    The velocity `v` remembers the previous movement, which smooths
    zigzags and speeds up the descent on flat parts.
    """

    def __init__(self, learning_rate: float, momentum: float = config.MOMENTUM) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self._velocity: list[np.ndarray] | None = None

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self._check_shapes(params, grads)
        if self._velocity is None:
            self._velocity = [np.zeros_like(param) for param in params]

        for param, grad, velocity in zip(params, grads, self._velocity):
            velocity *= self.momentum
            velocity -= self.learning_rate * grad
            param += velocity


class RMSprop(Optimizer):
    """RMSprop: a separate step size for every parameter.

    s = rho * s + (1 - rho) * g^2
    p -= lr * g / (sqrt(s) + eps)

    The step is divided by the typical size of recent gradients, so
    parameters with steep slopes take smaller steps.
    """

    def __init__(self, learning_rate: float, rho: float = RMSPROP_RHO) -> None:
        super().__init__(learning_rate)
        self.rho = rho
        self._mean_square: list[np.ndarray] | None = None

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self._check_shapes(params, grads)
        if self._mean_square is None:
            self._mean_square = [np.zeros_like(param) for param in params]

        for param, grad, mean_square in zip(params, grads, self._mean_square):
            mean_square *= self.rho
            mean_square += (1.0 - self.rho) * grad**2
            param -= self.learning_rate * grad / (np.sqrt(mean_square) + EPSILON)


class Adam(Optimizer):
    """Adam: momentum plus a separate step size for every parameter.

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g^2
    m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
    p -= lr * m_hat / (sqrt(v_hat) + eps)

    The division by (1 - beta^t) corrects the bias of `m` and `v`
    caused by starting from zero. `t` is the number of update calls.
    """

    def __init__(
        self,
        learning_rate: float,
        beta1: float = ADAM_BETA1,
        beta2: float = ADAM_BETA2,
    ) -> None:
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self._step = 0
        self._first_moment: list[np.ndarray] | None = None
        self._second_moment: list[np.ndarray] | None = None

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self._check_shapes(params, grads)
        if self._first_moment is None or self._second_moment is None:
            self._first_moment = [np.zeros_like(param) for param in params]
            self._second_moment = [np.zeros_like(param) for param in params]

        self._step += 1
        correction1 = 1.0 - self.beta1**self._step
        correction2 = 1.0 - self.beta2**self._step

        for param, grad, m, v in zip(
            params, grads, self._first_moment, self._second_moment
        ):
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * grad**2

            m_hat = m / correction1
            v_hat = v / correction2
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + EPSILON)


OPTIMIZERS: dict[str, type[Optimizer]] = {
    "sgd": SGD,
    "sgd_momentum": SGDMomentum,
    "rmsprop": RMSprop,
    "adam": Adam,
}


def get_optimizer(name: str, learning_rate: float) -> Optimizer:
    """Create the optimizer registered under `name`.

    Raises:
        ValueError: If the name is unknown or the learning rate is not positive.
    """
    try:
        optimizer_class = OPTIMIZERS[name]
    except KeyError:
        available = ", ".join(OPTIMIZERS)
        raise ValueError(
            f"Unknown optimizer '{name}'. Available: {available}"
        ) from None
    return optimizer_class(learning_rate)
