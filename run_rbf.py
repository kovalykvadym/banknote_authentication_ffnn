"""Group C: RBF network with a different number of hidden neurons (centers).

Trains the network for every number of centers from
`config.RBF_CENTER_COUNTS` and saves the results to `results/runs/`.
The RBF network has no learning rate and no epochs: the centers come from
k-means and the output weights from least squares in one step.

Run from the project root: python run_rbf.py
"""

from ffnn_lab import config
from ffnn_lab.data import prepare_dataset
from ffnn_lab.rbf import RBFNetwork
from ffnn_lab.runner import RunSummary, run_experiment


def run_group(
    results_dir=config.RESULTS_DIR,
    repeats: int = config.N_REPEATS,
) -> list[RunSummary]:
    """Run all RBF setups and return their summaries."""
    split = prepare_dataset()

    summaries = []
    for n_centers in config.RBF_CENTER_COUNTS:
        summary = run_experiment(
            lambda: RBFNetwork(n_centers, config.RANDOM_SEED),
            split,
            repeats=repeats,
            results_dir=results_dir,
        )
        summaries.append(summary)
        print_summary(summary)
    return summaries


def print_summary(s: RunSummary) -> None:
    """Print one line for a run: training time and errors."""
    print(
        f"{s.run_id}: time={s.training.training_time * 1000:.1f} ms, "
        f"train error={s.train_eval.error:.2%}, "
        f"test error={s.test_eval.error:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    run_group()
