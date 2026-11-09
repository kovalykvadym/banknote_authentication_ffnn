"""Group D: influence of the learning rate on the training time.

Trains a single-layer perceptron (tanh, delta rule) and a multilayer
perceptron 4-1 (tanh-tanh) with plain SGD for every learning rate from
`config.LEARNING_RATE_GRID`. Every setup is repeated with several seeds
(different initial weights and batch order), because one run is not enough
to see the typical behavior.

The study is made for two stopping tolerances: the main `config.MIN_DELTA`
and the stricter `STRICT_MIN_DELTA`, to check whether the conclusions depend
on the stopping criterion.

Results in `results/`:
    lr_study.csv            every run
    lr_study_summary.csv    mean and spread over the seeds

The graphs are drawn from `lr_study.csv` by `plot_results.py`.

Run from the project root: python run_lr_study.py
"""

import math
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ffnn_lab import config
from ffnn_lab.data import prepare_dataset
from ffnn_lab.mlp import MultilayerPerceptron
from ffnn_lab.perceptron import SingleLayerPerceptron
from ffnn_lab.utils import classification_error

STRICT_MIN_DELTA = 1e-6

# Model name -> function (learning rate, seed) -> new untrained model.
# The names are used in the CSV file and by `plot_results.py`.
MODELS: dict[str, Callable[[float, int], object]] = {
    "Single-layer perceptron (tanh, delta rule)": lambda lr, seed: (
        SingleLayerPerceptron("tanh", "delta", lr, seed=seed)
    ),
    "MLP 4-1 (tanh-tanh, SGD)": lambda lr, seed: (
        MultilayerPerceptron(("tanh", "tanh"), "sgd", lr, seed=seed)
    ),
}


def run_study(
    learning_rates: tuple[float, ...] = config.LEARNING_RATE_GRID,
    min_deltas: tuple[float, ...] = (config.MIN_DELTA, STRICT_MIN_DELTA),
    repeats: int = config.N_REPEATS,
) -> pd.DataFrame:
    """Train every setup `repeats` times and return one row per run."""
    split = prepare_dataset()

    rows = []
    for model_name, make_model in MODELS.items():
        for min_delta in min_deltas:
            for learning_rate in learning_rates:
                for repeat in range(repeats):
                    seed = config.RANDOM_SEED + repeat
                    model = make_model(learning_rate, seed)
                    training = model.fit(
                        split.X_train,
                        split.y_train,
                        batch_size=config.BATCH_SIZE,
                        max_epochs=config.MAX_EPOCHS,
                        patience=config.PATIENCE,
                        min_delta=min_delta,
                    )
                    final_loss = training.loss_history[-1]
                    test_error = classification_error(
                        split.y_test, model.predict(split.X_test)
                    )
                    rows.append(
                        {
                            "model": model_name,
                            "min_delta": min_delta,
                            "learning_rate": learning_rate,
                            "seed": seed,
                            "epochs": training.epochs_run,
                            "training_time_s": training.training_time,
                            "final_loss": final_loss,
                            "test_error_pct": test_error * 100,
                            "stopped_early": training.stopped_early,
                            "diverged": not math.isfinite(final_loss),
                        }
                    )
                print(
                    f"{model_name} | min_delta={min_delta:g} | "
                    f"lr={learning_rate:g} done",
                    flush=True,
                )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Average the runs over the seeds for every (model, tolerance, rate)."""
    table = table.assign(
        hit_limit=~table["stopped_early"] & ~table["diverged"],
    )
    summary = table.groupby(["model", "min_delta", "learning_rate"], sort=False).agg(
        epochs_mean=("epochs", "mean"),
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
    learning_rates: tuple[float, ...] = config.LEARNING_RATE_GRID,
    repeats: int = config.N_REPEATS,
) -> None:
    """Run the study and save the tables."""
    table = run_study(learning_rates=learning_rates, repeats=repeats)
    summary = summarize(table)

    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "lr_study.csv", index=False)
    summary.to_csv(results_dir / "lr_study_summary.csv", index=False)

    print()
    rounding = {
        "epochs_mean": 1,
        "time_mean_s": 3,
        "time_min_s": 3,
        "time_max_s": 3,
        "test_error_mean_pct": 2,
    }
    print(summary.round(rounding).to_string(index=False))
    print(f"\nSaved to {results_dir}")


if __name__ == "__main__":
    main()
