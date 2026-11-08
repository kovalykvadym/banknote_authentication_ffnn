"""Group B: multilayer perceptron 4-1 with different activations and optimizers.

Trains the network for every combination of an activation set from
`config.MLP_ACTIVATION_SETS` and an optimizer from `config.MLP_OPTIMIZERS`.
Every combination is run twice: with the main stopping tolerance
`config.MIN_DELTA` and with the stricter `STRICT_MIN_DELTA`. The second run
is saved with the tag "md1e-06" in its name and is used in the report to
show how the stopping criterion affects the training time.

Run from the project root: python run_mlp.py
"""

from ffnn_lab import config
from ffnn_lab.data import prepare_dataset
from ffnn_lab.mlp import MultilayerPerceptron
from ffnn_lab.runner import RunSummary, run_experiment

STRICT_MIN_DELTA = 1e-6


def run_group(
    results_dir=config.RESULTS_DIR,
    repeats: int = config.N_REPEATS,
) -> list[RunSummary]:
    """Run all MLP setups for both tolerances and return their summaries."""
    split = prepare_dataset()

    summaries = []
    for min_delta in (config.MIN_DELTA, STRICT_MIN_DELTA):
        # The main run has no tag, so its folder name stays short.
        tag = "" if min_delta == config.MIN_DELTA else f"md{min_delta:g}"
        fit_kwargs = {
            "batch_size": config.BATCH_SIZE,
            "max_epochs": config.MAX_EPOCHS,
            "patience": config.PATIENCE,
            "min_delta": min_delta,
        }
        for activations in config.MLP_ACTIVATION_SETS.values():
            for optimizer in config.MLP_OPTIMIZERS:
                summary = run_experiment(
                    lambda: MultilayerPerceptron(
                        activations, optimizer, config.DEFAULT_LEARNING_RATE
                    ),
                    split,
                    fit_kwargs=fit_kwargs,
                    repeats=repeats,
                    tag=tag,
                    results_dir=results_dir,
                )
                summaries.append(summary)
                print_summary(summary)
    return summaries


def print_summary(s: RunSummary) -> None:
    """Print one line for a run: epochs, training time and errors."""
    print(
        f"{s.run_id}: epochs={s.training.epochs_run}, "
        f"time={s.training.training_time * 1000:.1f} ms, "
        f"train error={s.train_eval.error:.2%}, "
        f"test error={s.test_eval.error:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    run_group()
