"""Group E: influence of the batch size on the training.

Trains a multilayer perceptron 4-1 (tanh-tanh) with three batch modes from
`config.BATCH_SIZE_STUDY_VALUES`: online (one sample per weight update),
mini-batch and full batch (the whole training set per update). The study is
made for plain SGD and for Adam, each with two learning rates, and every
setup is repeated with several seeds.

Results in `results/`:
    batch_study.csv             every run
    batch_study_summary.csv     mean and spread over the seeds

The graph is drawn from `batch_study.csv` by `plot_results.py`.

Run from the project root: python run_batch_study.py
"""

import math
from pathlib import Path

import pandas as pd

from ffnn_lab import config
from ffnn_lab.data import prepare_dataset
from ffnn_lab.mlp import MultilayerPerceptron
from ffnn_lab.utils import classification_error

# Optimizer -> learning rates to try. Adam works with smaller steps than SGD.
SETUPS: dict[str, tuple[float, ...]] = {
    "sgd": (0.01, 0.1),
    "adam": (0.001, 0.01),
}


def batch_label(batch_size: int | None) -> str:
    """Return the text label of a batch size: "1", "32" or "full"."""
    return "full" if batch_size is None else str(batch_size)


def run_study(repeats: int = config.N_REPEATS) -> pd.DataFrame:
    """Train every setup `repeats` times and return one row per run."""
    split = prepare_dataset()
    n_samples = len(split.X_train)

    rows = []
    for optimizer, learning_rates in SETUPS.items():
        for learning_rate in learning_rates:
            for batch_size in config.BATCH_SIZE_STUDY_VALUES:
                size = n_samples if batch_size is None else batch_size
                updates_per_epoch = math.ceil(n_samples / size)
                for repeat in range(repeats):
                    seed = config.RANDOM_SEED + repeat
                    model = MultilayerPerceptron(
                        ("tanh", "tanh"), optimizer, learning_rate, seed=seed
                    )
                    training = model.fit(
                        split.X_train,
                        split.y_train,
                        batch_size=batch_size,
                        max_epochs=config.MAX_EPOCHS,
                        patience=config.PATIENCE,
                        min_delta=config.MIN_DELTA,
                    )
                    final_loss = training.loss_history[-1]
                    test_error = classification_error(
                        split.y_test, model.predict(split.X_test)
                    )
                    rows.append(
                        {
                            "optimizer": optimizer,
                            "learning_rate": learning_rate,
                            "batch_size": batch_label(batch_size),
                            "seed": seed,
                            "epochs": training.epochs_run,
                            "weight_updates": training.epochs_run * updates_per_epoch,
                            "training_time_s": training.training_time,
                            "final_loss": final_loss,
                            "test_error_pct": test_error * 100,
                            "stopped_early": training.stopped_early,
                            "diverged": not math.isfinite(final_loss),
                        }
                    )
                print(
                    f"{optimizer} | lr={learning_rate:g} | "
                    f"batch={batch_label(batch_size)} done",
                    flush=True,
                )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Average the runs over the seeds for every (optimizer, rate, batch)."""
    table = table.assign(
        hit_limit=~table["stopped_early"] & ~table["diverged"],
    )
    summary = table.groupby(
        ["optimizer", "learning_rate", "batch_size"], sort=False
    ).agg(
        epochs_mean=("epochs", "mean"),
        weight_updates_mean=("weight_updates", "mean"),
        time_mean_s=("training_time_s", "mean"),
        time_min_s=("training_time_s", "min"),
        time_max_s=("training_time_s", "max"),
        test_error_mean_pct=("test_error_pct", "mean"),
        runs_hit_epoch_limit=("hit_limit", "sum"),
        runs_diverged=("diverged", "sum"),
    )
    return summary.reset_index()


def main(
    results_dir: Path = config.RESULTS_DIR,
    repeats: int = config.N_REPEATS,
) -> None:
    """Run the study and save the tables."""
    table = run_study(repeats=repeats)
    summary = summarize(table)

    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "batch_study.csv", index=False)
    summary.to_csv(results_dir / "batch_study_summary.csv", index=False)

    print()
    rounding = {
        "epochs_mean": 1,
        "weight_updates_mean": 0,
        "time_mean_s": 3,
        "time_min_s": 3,
        "time_max_s": 3,
        "test_error_mean_pct": 2,
    }
    print(summary.round(rounding).to_string(index=False))
    print(f"\nSaved to {results_dir}")


if __name__ == "__main__":
    main()
