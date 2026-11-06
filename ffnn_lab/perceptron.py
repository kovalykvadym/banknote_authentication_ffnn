import numpy as np

from ffnn_lab import config
from ffnn_lab.activations import get_activation
from ffnn_lab.data import to_bipolar
from ffnn_lab.utils import outputs_to_labels, Timer, TrainingResult

LEARNING_RULES = ("rosenblatt", "delta")


class SingleLayerPerceptron:
    """One neuron: output = f(weights . x + bias).

    Two learning rules are supported:
        * "rosenblatt": correction = error * x. Only for the threshold
          activation, which has no derivative.
        * "delta": correction = error * f'(z) * x. For differentiable
          activations (linear, sigmoid, tanh).

    Args:
        activation: Name of the activation function (see `activations.py`).
        learning_rule: "rosenblatt" or "delta".
        learning_rate: Step size of the weight correction, must be positive.
        n_features: Number of input features.
        seed: Seed for the initial weights and for shuffling.

    Raises:
        ValueError: If the rule is unknown, the learning rate is not positive,
            or the rule does not fit the activation.
    """

    def __init__(
        self,
        activation: str,
        learning_rule: str,
        learning_rate: float = config.DEFAULT_LEARNING_RATE,
        n_features: int = len(config.FEATURE_COLUMNS),
        seed: int = config.RANDOM_SEED,
    ) -> None:
        if learning_rule not in LEARNING_RULES:
            raise ValueError(
                f"Unknown learning rule '{learning_rule}'. "
                f"Available: {', '.join(LEARNING_RULES)}"
            )
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        self.activation = get_activation(activation)

        if learning_rule == "rosenblatt" and self.activation.name != "threshold":
            raise ValueError(
                "The Rosenblatt rule works only with the threshold activation, "
                f"got '{self.activation.name}'"
            )
        if learning_rule == "delta" and self.activation.derivative is None:
            raise ValueError(
                f"The delta rule needs a derivative, but '{self.activation.name}' "
                "has none. Use the Rosenblatt rule instead."
            )

        self.learning_rule = learning_rule
        self.learning_rate = learning_rate
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self.weights = self._rng.normal(0.0, 0.1, size=n_features)
        self.bias = 0.0

        self.initial_weights = self.weights.copy()
        self.initial_bias = self.bias

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Return the weighted sum z = X @ weights + bias, shape (n,)."""
        return X @ self.weights + self.bias

    def predict_outputs(self, X: np.ndarray) -> np.ndarray:
        """Return the raw neuron outputs f(z), shape (n,)."""
        return self.activation.forward(self.net_input(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels 0 or 1, shape (n,)."""
        outputs = self.predict_outputs(X)
        return outputs_to_labels(outputs, bipolar=self.activation.bipolar)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int | None = config.BATCH_SIZE,
        max_epochs: int = config.MAX_EPOCHS,
        patience: int = config.PATIENCE,
        min_delta: float = config.MIN_DELTA,
    ) -> TrainingResult:
        """Train the neuron with early stopping.

        Training stops when the loss is zero, or has not improved by at least
        `min_delta` for `patience` epochs in a row, or after `max_epochs`.
        The best weights seen during training are restored at the end.

        Args:
            X: Scaled features, shape (n, n_features).
            y: Labels 0/1, shape (n,).
            batch_size: Samples per weight update. None means the whole set.
            max_epochs: Upper limit of epochs.
            patience: Epochs without improvement before stopping.
            min_delta: Smallest loss decrease counted as an improvement.

        Returns:
            A `TrainingResult`. `stopped_early` is False if the epoch limit
            was reached.

        Raises:
            ValueError: If the shapes of `X` and `y` do not fit the neuron
                or `batch_size` is not positive.
        """
        self._check_training_data(X, y)
        targets = self._encode_targets(y)

        n_samples = len(X)
        if batch_size is None:
            batch_size = n_samples
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        loss_history: list[float] = []
        best_loss = np.inf
        best_weights = self.weights.copy()
        best_bias = self.bias
        wait = 0
        stopped_early = False

        with Timer() as timer:
            for _ in range(max_epochs):
                order = self._rng.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    idx = order[start : start + batch_size]
                    self._update_batch(X[idx], targets[idx])

                loss = self._epoch_loss(X, targets)
                loss_history.append(loss)

                if loss < best_loss - min_delta:
                    best_loss = loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    wait = 0
                else:
                    wait += 1

                if loss == 0.0 or wait >= patience:
                    stopped_early = True
                    break

        self.weights = best_weights
        self.bias = best_bias

        return TrainingResult(
            epochs_run=len(loss_history),
            stopped_early=stopped_early,
            training_time=timer.elapsed,
            loss_history=loss_history,
        )

    def _check_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Check that X and y have the shapes the neuron expects."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        if X.shape[1] != len(self.weights):
            raise ValueError(
                f"X has {X.shape[1]} features, the neuron expects {len(self.weights)}"
            )
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows, but y has {len(y)} labels")

    def _encode_targets(self, y: np.ndarray) -> np.ndarray:
        """Return targets in the range of the activation: -1/+1 or 0/1."""
        if self.activation.bipolar:
            return to_bipolar(y).astype(float)
        return np.asarray(y, dtype=float)

    def _update_batch(self, X_batch: np.ndarray, t_batch: np.ndarray) -> None:
        """Correct the weights once using the mean correction of the batch."""
        z = self.net_input(X_batch)
        error = t_batch - self.activation.forward(z)

        if self.learning_rule == "rosenblatt":
            factor = error
        else:
            factor = error * self.activation.derivative(z)

        self.weights += self.learning_rate * (X_batch.T @ factor) / len(X_batch)
        self.bias += self.learning_rate * float(np.mean(factor))

    def _epoch_loss(self, X: np.ndarray, targets: np.ndarray) -> float:
        """Return the mean squared error over the whole training set."""
        errors = targets - self.predict_outputs(X)
        return float(np.mean(errors**2))
