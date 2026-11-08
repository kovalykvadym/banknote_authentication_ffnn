"""Multilayer perceptron with one hidden layer, trained by backpropagation.

The network has the structure n_features -> hidden layer -> one output neuron.
The model computes outputs and gradients, while the weight corrections are
applied by an optimizer from `optimizers.py`.
"""

import numpy as np

from ffnn_lab import config
from ffnn_lab.activations import get_activation
from ffnn_lab.data import to_bipolar
from ffnn_lab.optimizers import get_optimizer
from ffnn_lab.utils import Timer, TrainingResult, outputs_to_labels


class MultilayerPerceptron:
    """Two-layer perceptron: hidden layer and one output neuron.

    The parameters are stored in the list `params = [W1, b1, W2, b2]`:
        W1: (n_features, hidden_units), b1: (hidden_units,)
        W2: (hidden_units, 1),          b2: (1,)

    The activation of the output layer decides how the labels are encoded:
    -1/+1 with the threshold 0 for bipolar activations (tanh), otherwise
    0/1 with the threshold 0.5.

    Args:
        activations: Pair (hidden activation, output activation) by name.
        optimizer: Name of the optimizer (see `optimizers.py`).
        learning_rate: Step size of the optimizer.
        n_features: Number of input features.
        hidden_units: Number of neurons in the hidden layer.
        output_units: Number of output neurons, must be 1.
        seed: Seed for the initial weights and for shuffling.

    Raises:
        ValueError: If the structure is not supported, an activation has
            no derivative, the optimizer is unknown or the learning rate
            is not positive.
    """

    def __init__(
        self,
        activations: tuple[str, str],
        optimizer: str,
        learning_rate: float = config.DEFAULT_LEARNING_RATE,
        n_features: int = len(config.FEATURE_COLUMNS),
        hidden_units: int = config.MLP_HIDDEN_UNITS,
        output_units: int = config.MLP_OUTPUT_UNITS,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        if output_units != 1:
            raise ValueError(f"Only one output neuron is supported, got {output_units}")
        if hidden_units < 1:
            raise ValueError(f"hidden_units must be at least 1, got {hidden_units}")

        self.hidden_activation = get_activation(activations[0])
        self.output_activation = get_activation(activations[1])
        for activation in (self.hidden_activation, self.output_activation):
            if activation.derivative is None:
                raise ValueError(
                    f"Backpropagation needs a derivative, but '{activation.name}' "
                    "has none"
                )

        self.optimizer_name = optimizer
        self.optimizer = get_optimizer(optimizer, learning_rate)
        self.learning_rate = learning_rate
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        # The spread 1 / sqrt(number of layer inputs) keeps the weighted sums
        # moderate, so the neurons do not start in the saturated zone.
        # Random weights are required: equal weights would make all hidden
        # neurons identical forever. The biases can start from zero.
        self.params: list[np.ndarray] = [
            self._rng.normal(0.0, n_features**-0.5, (n_features, hidden_units)),
            np.zeros(hidden_units),
            self._rng.normal(0.0, hidden_units**-0.5, (hidden_units, 1)),
            np.zeros(1),
        ]

        # Kept for the report: the state before any training.
        self.initial_params = [param.copy() for param in self.params]

    def predict_outputs(self, X: np.ndarray) -> np.ndarray:
        """Return the raw network outputs, shape (n,)."""
        return self._forward(X)[3][:, 0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels 0 or 1, shape (n,)."""
        return outputs_to_labels(
            self.predict_outputs(X), bipolar=self.output_activation.bipolar
        )

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean squared error for labels `y` (0/1)."""
        return self._epoch_loss(X, self._encode_targets(y))

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        """Return the gradients of 0.5 * MSE for labels `y` (0/1).

        The list matches `params`: [dW1, db1, dW2, db2]. It is public to
        allow the numerical gradient check.
        """
        return self._gradients(X, self._encode_targets(y))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int | None = config.BATCH_SIZE,
        max_epochs: int = config.MAX_EPOCHS,
        patience: int = config.PATIENCE,
        min_delta: float = config.MIN_DELTA,
    ) -> TrainingResult:
        """Train the network with early stopping.

        Training stops when the loss is zero, or has not improved by at least
        `min_delta` for `patience` epochs in a row, or when the loss is not a
        finite number (divergence), or after `max_epochs`. The best
        parameters seen during training are restored at the end.

        Args:
            X: Scaled features, shape (n, n_features).
            y: Labels 0/1, shape (n,).
            batch_size: Samples per weight update. None means the whole set.
            max_epochs: Upper limit of epochs.
            patience: Epochs without improvement before stopping.
            min_delta: Smallest loss decrease counted as an improvement.

        Returns:
            A `TrainingResult`. `stopped_early` is False if the epoch limit was
            reached or the training diverged.

        Raises:
            ValueError: If the shapes of `X` and `y` do not fit the network
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
        best_params = [param.copy() for param in self.params]
        wait = 0
        stopped_early = False

        with Timer() as timer:
            for _ in range(max_epochs):
                order = self._rng.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    idx = order[start : start + batch_size]
                    grads = self._gradients(X[idx], targets[idx])
                    self.optimizer.update(self.params, grads)

                loss = self._epoch_loss(X, targets)
                loss_history.append(loss)

                if not np.isfinite(loss):
                    break

                if loss < best_loss - min_delta:
                    best_loss = loss
                    best_params = [param.copy() for param in self.params]
                    wait = 0
                else:
                    wait += 1

                if loss == 0.0 or wait >= patience:
                    stopped_early = True
                    break

        # Restore in place, so the arrays in `self.params` stay the same objects.
        for param, best in zip(self.params, best_params):
            param[...] = best

        return TrainingResult(
            epochs_run=len(loss_history),
            stopped_early=stopped_early,
            training_time=timer.elapsed,
            loss_history=loss_history,
        )

    def _forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass. Returns (Z1, A1, Z2, A2), all needed by backpropagation.

        Z are the weighted sums, A are the layer outputs. Shapes: Z1 and A1
        are (n, hidden_units), Z2 and A2 are (n, 1).
        """
        W1, b1, W2, b2 = self.params
        Z1 = X @ W1 + b1
        A1 = self.hidden_activation.forward(Z1)
        Z2 = A1 @ W2 + b2
        A2 = self.output_activation.forward(Z2)
        return Z1, A1, Z2, A2

    def _gradients(self, X: np.ndarray, targets: np.ndarray) -> list[np.ndarray]:
        """Backward pass: gradients of 0.5 * MSE for all parameters.

        `targets` must have the shape (n, 1). Dividing by n turns the sums
        over the batch into means. Every gradient has the shape of its
        parameter.
        """
        _, _, W2, _ = self.params
        Z1, A1, Z2, A2 = self._forward(X)

        dZ2 = (A2 - targets) / len(X) * self.output_activation.derivative(Z2)
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0)

        # The blame of the output error is passed to the hidden neurons
        # through the weights of the second layer.
        dZ1 = (dZ2 @ W2.T) * self.hidden_activation.derivative(Z1)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)

        return [dW1, db1, dW2, db2]

    def _epoch_loss(self, X: np.ndarray, targets: np.ndarray) -> float:
        """Return the mean squared error over the whole set."""
        errors = targets[:, 0] - self.predict_outputs(X)
        return float(np.mean(errors**2))

    def _encode_targets(self, y: np.ndarray) -> np.ndarray:
        """Return targets as a column (n, 1): -1/+1 or 0/1 by output activation.

        The column shape matters: with the shape (n,) the subtraction
        `A2 - targets` would silently produce an (n, n) array.
        """
        y = np.asarray(y)
        targets = to_bipolar(y) if self.output_activation.bipolar else y
        return targets.astype(float).reshape(-1, 1)

    def _check_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Check that X and y have the shapes the network expects."""
        n_features = self.params[0].shape[0]
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        if X.shape[1] != n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, the network expects {n_features}"
            )
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows, but y has {len(y)} labels")
