"""Check the backpropagation gradients against numerical gradients.

For every parameter value the script changes it by a tiny step in both
directions and measures how the loss changes. This measured slope must
match the analytical gradient of the model.

Run from the project root. The check passes if every difference is below
the threshold.
"""

import numpy as np

from ffnn_lab import config
from ffnn_lab.mlp import MultilayerPerceptron

STEP = 1e-6
THRESHOLD = 1e-6
N_SAMPLES = 7
SEED = 0


def max_gradient_difference(model: MultilayerPerceptron, X, y) -> float:
    """Return the largest |numerical - analytical| gradient difference."""
    gradients = model.compute_gradients(X, y)
    worst = 0.0

    for param, gradient in zip(model.params, gradients):
        if param.shape != gradient.shape:
            raise ValueError(
                f"Gradient shape {gradient.shape} differs from "
                f"parameter shape {param.shape}"
            )
        for index in np.ndindex(param.shape):
            original = param[index]

            param[index] = original + STEP
            loss_up = 0.5 * model.loss(X, y)
            param[index] = original - STEP
            loss_down = 0.5 * model.loss(X, y)
            param[index] = original

            numerical = (loss_up - loss_down) / (2 * STEP)
            worst = max(worst, abs(numerical - gradient[index]))

    return worst


def main() -> None:
    """Run the check for every activation set from the configuration."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(N_SAMPLES, len(config.FEATURE_COLUMNS)))
    y = rng.integers(0, 2, size=N_SAMPLES)

    all_passed = True
    for name, activations in config.MLP_ACTIVATION_SETS.items():
        model = MultilayerPerceptron(activations, "sgd", seed=1)
        difference = max_gradient_difference(model, X, y)
        passed = difference < THRESHOLD
        all_passed = all_passed and passed
        label = "OK" if passed else "CHECK"
        print(f"{name:16s} max difference: {difference:.2e} [{label}]")

    print("All checks passed" if all_passed else "Some checks FAILED")


if __name__ == "__main__":
    main()
