"""Group A: single-layer perceptron with different activations and rules.

Trains one neuron for every pair (activation, learning rule) from
`config.SINGLE_LAYER_SETUPS` and saves the results to `results/runs/`.

Run from the project root: python run_single_layer.py
"""

from ffnn_lab import config
from ffnn_lab.data import prepare_dataset
from ffnn_lab.perceptron import SingleLayerPerceptron
from ffnn_lab.runner import RunSummary, run_experiment


def run_group(results_dir=config.RESULTS_DIR) -> list[RunSummary]:
    """Run all single-layer setups and return their summaries."""
    split = prepare_dataset()
    fit_kwargs = {
        "batch_size": config.BATCH_SIZE,
        "max_epochs": config.MAX_EPOCHS,
        "patience": config.PATIENCE,
        "min_delta": config.MIN_DELTA,
    }

    summaries = []
    for activation, rule in config.SINGLE_LAYER_SETUPS:
        summary = run_experiment(
            lambda: SingleLayerPerceptron(
                activation, rule, config.DEFAULT_LEARNING_RATE
            ),
            split,
            fit_kwargs=fit_kwargs,
            results_dir=results_dir,
        )
        summaries.append(summary)
    return summaries


def print_summaries(summaries: list[RunSummary]) -> None:
    """Print one line per run: epochs, training time and errors."""
    for s in summaries:
        print(
            f"{s.run_id}: epochs={s.training.epochs_run}, "
            f"time={s.training.training_time * 1000:.1f} ms, "
            f"train error={s.train_eval.error:.2%}, "
            f"test error={s.test_eval.error:.2%}"
        )


if __name__ == "__main__":
    print_summaries(run_group())
