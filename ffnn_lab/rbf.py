"""Radial basis function (RBF) network: Gaussian hidden layer, linear output."""

import numpy as np
from sklearn.cluster import KMeans

from ffnn_lab import config
from ffnn_lab.utils import Timer, TrainingResult, outputs_to_labels

KMEANS_RESTARTS = 10


class RBFNetwork:
    """RBF network for binary classification.

    Structure: n_features -> n_centers Gaussian neurons -> 1 linear neuron.
    Training has two stages:
        1.  The centers are found by KMeans, the common width sigma is
            computed from the distances between the centers.
        2.  The weights of the output neuron are found in one step by
            the method of least squares, so there is no learning rate.

    The labels are 0/1 and the decision threshold is 0.5.

    Args:
        n_centers: Number of Gaussian neurons, at least 2.
        seed: Seed for KMeans.

    Raises:
        ValueError: If `n_centers` is less than 2.
    """

    def __init__(self, n_centers: int, seed: int = config.RANDOM_SEED) -> None:
        if n_centers < 2:
            raise ValueError(f"n_centers must be at least 2, got {n_centers}")
        self.n_centers = n_centers
        self.seed = seed

        self.centers: np.ndarray | None = None
        self.sigma: float | None = None
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """Train the network on features `X` and labels `y` (0/1).

        Returns:
            A `TrainingResult`. The training is not iterative, so
            `epochs_run` is 0, `stopped_early` is True and `loss_history`
            holds one value: the final mean squared error.

        Raises:
            ValueError: If the shapes do not fit or there are fewer
                samples than centers.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows, but y has {len(y)} labels")
        if len(X) < self.n_centers:
            raise ValueError(f"Need at least {self.n_centers} samples, got {len(X)}")

        targets = np.asarray(y, dtype=float)

        with Timer() as timer:
            kmeans = KMeans(
                n_clusters=self.n_centers,
                n_init=KMEANS_RESTARTS,
                random_state=self.seed,
            ).fit(X)
            self.centers = kmeans.cluster_centers_
            self.sigma = self._compute_sigma(self.centers)

            # Design matrix: Gaussian outputs plus a column of ones (the bias).
            design = np.column_stack([self._hidden_outputs(X), np.ones(len(X))])
            solution, *_ = np.linalg.lstsq(design, targets, rcond=None)
            self.weights = solution[:-1]
            self.bias = float(solution[-1])

        loss = float(np.mean((targets - self.predict_outputs(X)) ** 2))
        return TrainingResult(
            epochs_run=0,
            stopped_early=True,
            training_time=timer.elapsed,
            loss_history=[loss],
        )

    def predict_outputs(self, X: np.ndarray) -> np.ndarray:
        """Return the raw network outputs, shape (n,)."""
        self._check_fitted()
        return self._hidden_outputs(X) @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels 0 or 1, shape (n,)."""
        return outputs_to_labels(self.predict_outputs(X), bipolar=False)

    def _hidden_outputs(self, X: np.ndarray) -> np.ndarray:
        """Return the Gaussian outputs exp(-d^2 / (2 sigma^2)), shape (n, K)."""
        self._check_fitted()
        # Squared distance from every sample to every center, shape (n, K).
        squared = ((X[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-squared / (2.0 * self.sigma**2))

    def _compute_sigma(self, centers: np.ndarray) -> float:
        """Common width: sigma = d_max / sqrt(2 * K).

        d_max is the largest distance between two centers and K is their
        number. The width grows with the spread of the centers and shrinks
        when there are more of them.
        """
        diff = centers[:, None, :] - centers[None, :, :]
        d_max = float(np.sqrt((diff**2).sum(axis=2)).max())
        if d_max == 0.0:
            raise ValueError("All centers coincide, cannot compute sigma")
        return d_max / np.sqrt(2.0 * len(centers))

    def _check_fitted(self) -> None:
        """Raise an error if the network has not been trained yet."""
        if self.centers is None:
            raise ValueError("The network is not trained yet, call fit first")
