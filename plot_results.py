"""Draw all graphs of the report from the saved result tables.

Reads `results/summary.csv` (made by `build_summary.py`) and
`results/lr_study.csv` (made by `run_lr_study.py`) and writes the graphs
to `results/figures/`:
    lr_time.png, lr_epochs.png, lr_error.png    influence of the learning rate
    mlp_optimizers.png                          optimizers of the MLP
    rbf_centers.png                             RBF network, number of centers
    architectures_comparison.png                all architectures together

The graphs need no training, so they can be redrawn in a second. All texts
are in the `TEXT` dictionary: change `LANGUAGE` to switch between Ukrainian
("uk") and English ("en").

Run from the project root: python plot_results.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from ffnn_lab import config

LANGUAGE = "uk"

TEXT = {
    "uk": {
        "learning_rate": "Швидкість навчання",
        "epochs": "Кількість епох",
        "training_time": "Час навчання, с",
        "test_error": "Помилка на тесті, %",
        "error": "Помилка, %",
        "train_error": "Помилка на навчальній вибірці, %",
        "classification_time": "Час класифікації, мс",
        "centers": "Кількість центрів RBF",
        "train": "навчальна вибірка",
        "test": "тестова вибірка",
        "optimizers_title": "MLP 4-1: оптимізатори",
        "rbf_error_title": "RBF-мережа: помилка",
        "rbf_time_title": "RBF-мережа: час класифікації",
        "vs_training_time": "Точність і час навчання",
        "vs_classification_time": "Точність і час класифікації",
        "batch_size": "Розмір батча",
        "batch_full": "весь набір",
        "models": {
            "Single-layer perceptron (tanh, delta rule)": (
                "Одношаровий перцептрон (tanh, дельта-правило)"
            ),
            "MLP 4-1 (tanh-tanh, SGD)": "MLP 4-1 (tanh-tanh, SGD)",
        },
        "architectures": {
            "Single-layer perceptron": "Одношаровий перцептрон",
            "Multilayer perceptron": "Багатошаровий перцептрон",
            "RBF Network": "RBF-мережа",
        },
        "optimizer_names": {
            "sgd": "SGD",
            "sgd_momentum": "SGD з моментом",
            "rmsprop": "RMSprop",
            "adam": "Adam",
        },
    },
    "en": {
        "learning_rate": "Learning rate",
        "epochs": "Epochs",
        "training_time": "Training time, s",
        "test_error": "Test error, %",
        "error": "Error, %",
        "train_error": "Training error, %",
        "classification_time": "Classification time, ms",
        "centers": "Number of RBF centers",
        "train": "training set",
        "test": "test set",
        "optimizers_title": "MLP 4-1: optimizers",
        "rbf_error_title": "RBF network: error",
        "rbf_time_title": "RBF network: classification time",
        "vs_training_time": "Accuracy and training time",
        "vs_classification_time": "Accuracy and classification time",
        "batch_size": "Batch size",
        "batch_full": "full set",
        "models": {
            "Single-layer perceptron (tanh, delta rule)": (
                "Single-layer perceptron (tanh, delta rule)"
            ),
            "MLP 4-1 (tanh-tanh, SGD)": "MLP 4-1 (tanh-tanh, SGD)",
        },
        "architectures": {
            "Single-layer perceptron": "Single-layer perceptron",
            "Multilayer perceptron": "Multilayer perceptron",
            "RBF Network": "RBF network",
        },
        "optimizer_names": {
            "sgd": "SGD",
            "sgd_momentum": "SGD with momentum",
            "rmsprop": "RMSprop",
            "adam": "Adam",
        },
    },
}
ARCHITECTURE_STYLES = {
    "Single-layer perceptron": ("tab:green", "o"),
    "Multilayer perceptron": ("tab:blue", "s"),
    "RBF Network": ("tab:red", "^"),
}


def _text(key: str):
    return TEXT[LANGUAGE][key]


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _main_runs(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep the runs with the main stopping tolerance (and the RBF runs)."""
    is_main = summary["min_delta"].isna() | (summary["min_delta"] == config.MIN_DELTA)
    return summary[is_main]


def plot_lr_metric(
    table: pd.DataFrame, column: str, ylabel_key: str, path: Path, log_y: bool
) -> None:
    """Draw `column` against the learning rate, one panel per model.

    The line is the mean over the seeds, the band is the range from the
    smallest to the largest value. One line per stopping tolerance.
    """
    model_names = list(dict.fromkeys(table["model"]))
    fig, axes = plt.subplots(1, len(model_names), figsize=(11, 4.2))
    for axis, model_name in zip(np.atleast_1d(axes), model_names):
        part = table[table["model"] == model_name]
        for min_delta, group in part.groupby("min_delta", sort=True):
            stats = group.groupby("learning_rate")[column].agg(["mean", "min", "max"])
            line = axis.plot(
                stats.index,
                stats["mean"],
                marker="o",
                label=f"min_delta={min_delta:g}",
            )[0]
            axis.fill_between(
                stats.index,
                stats["min"],
                stats["max"],
                alpha=0.2,
                color=line.get_color(),
            )
        axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_title(_text("models").get(model_name, model_name), fontsize=10)
        axis.set_xlabel(_text("learning_rate"))
        axis.set_ylabel(_text(ylabel_key))
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    _save(fig, path)


def plot_mlp_optimizers(summary: pd.DataFrame, path: Path) -> None:
    """Grouped bars: epochs and training time of the MLP by optimizer."""
    mlp = _main_runs(summary)
    mlp = mlp[mlp["architecture"] == "Multilayer perceptron"]
    optimizers = list(config.MLP_OPTIMIZERS)
    activation_sets = [a.replace("-", ", ") for a in config.MLP_ACTIVATION_SETS]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.8 / len(optimizers)
    for axis, column, ylabel_key in zip(
        axes,
        ("epochs", "training_time_s"),
        ("epochs", "training_time"),
    ):
        for i, optimizer in enumerate(optimizers):
            part = mlp[mlp["training_method"] == optimizer]
            values = [
                part[part["activations"] == act][column].iloc[0]
                for act in activation_sets
            ]
            positions = np.arange(len(activation_sets)) + (i - 1.5) * width
            axis.bar(
                positions, values, width, label=_text("optimizer_names")[optimizer]
            )
        axis.set_xticks(np.arange(len(activation_sets)))
        axis.set_xticklabels([a.replace(", ", "-") for a in activation_sets])
        axis.set_yscale("log")
        axis.set_ylabel(_text(ylabel_key))
        axis.set_title(_text("optimizers_title"), fontsize=10)
        axis.grid(True, axis="y", which="both", alpha=0.3)
        axis.legend(fontsize=8)
    _save(fig, path)


def plot_rbf_centers(summary: pd.DataFrame, path: Path) -> None:
    """RBF network: errors and classification time against the centers."""
    rbf = summary[summary["architecture"] == "RBF Network"].copy()
    if rbf.empty:
        raise ValueError(
            "No RBF rows in the table, architectures found: "
            f"{sorted(summary['architecture'].unique())}"
        )
    rbf["centers"] = rbf["neurons"].str.split("-").str[0].astype(int)
    rbf = rbf.sort_values("centers")
    centers = rbf["centers"].to_numpy()

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot(centers, rbf["train_error_pct"], marker="o", label=_text("train"))
    left.plot(centers, rbf["test_error_pct"], marker="s", label=_text("test"))
    left.set_ylabel(_text("error"))
    left.set_title(_text("rbf_error_title"), fontsize=10)

    right.plot(
        centers,
        rbf["train_classification_time_ms"],
        marker="o",
        label=_text("train"),
    )
    right.plot(
        centers,
        rbf["test_classification_time_ms"],
        marker="s",
        label=_text("test"),
    )
    right.set_yscale("log")
    right.set_ylabel(_text("classification_time"))
    right.set_title(_text("rbf_time_title"), fontsize=10)

    for axis in (left, right):
        axis.set_xscale("log")
        axis.set_xticks(centers)
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(plt.NullFormatter())
        axis.set_xlabel(_text("centers"))
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    _save(fig, path)


def plot_architectures(summary: pd.DataFrame, path: Path) -> None:
    """Scatter plot: test error against the training and classification time."""
    runs = _main_runs(summary)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for architecture, (color, marker) in ARCHITECTURE_STYLES.items():
        part = runs[runs["architecture"] == architecture]
        label = _text("architectures")[architecture]
        for axis, column in (
            (left, "training_time_s"),
            (right, "test_classification_time_ms"),
        ):
            axis.scatter(
                part[column],
                part["test_error_pct"],
                color=color,
                marker=marker,
                alpha=0.75,
                label=label,
            )
    left.set_xlabel(_text("training_time"))
    left.set_title(_text("vs_training_time"), fontsize=10)
    right.set_xlabel(_text("classification_time"))
    right.set_title(_text("vs_classification_time"), fontsize=10)
    for axis in (left, right):
        axis.set_xscale("log")
        axis.set_ylabel(_text("test_error"))
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    _save(fig, path)


def plot_batch_study(table: pd.DataFrame, path: Path) -> None:
    """Grouped bars: epochs, training time and test error by batch size.

    One group of bars per batch size, one bar per (optimizer, learning rate).
    The whiskers show the range over the seeds.
    """
    table = table.assign(batch_size=table["batch_size"].astype(str))
    batch_sizes = ["1", "32", "full"]
    setups = list(table.groupby(["optimizer", "learning_rate"], sort=False))
    panels = (
        ("epochs", "epochs", True),
        ("training_time_s", "training_time", True),
        ("test_error_pct", "test_error", False),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    width = 0.8 / len(setups)
    for axis, (column, ylabel_key, log_y) in zip(axes, panels):
        for i, ((optimizer, learning_rate), part) in enumerate(setups):
            stats = (
                part.groupby("batch_size")[column]
                .agg(["mean", "min", "max"])
                .reindex(batch_sizes)
            )
            shift = (i - (len(setups) - 1) / 2) * width
            positions = np.arange(len(batch_sizes)) + shift
            axis.bar(
                positions,
                stats["mean"],
                width,
                yerr=[stats["mean"] - stats["min"], stats["max"] - stats["mean"]],
                capsize=3,
                label=f"{_text('optimizer_names')[optimizer]}, lr={learning_rate:g}",
            )
        axis.set_xticks(np.arange(len(batch_sizes)))
        axis.set_xticklabels(
            [_text("batch_full") if b == "full" else b for b in batch_sizes],
        )
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel(_text("batch_size"))
        axis.set_ylabel(_text(ylabel_key))
        axis.grid(True, axis="y", which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    _save(fig, path)


def main(results_dir: Path = config.RESULTS_DIR) -> None:
    """Draw all graphs that can be drawn from the saved tables."""
    figures = results_dir / "figures"

    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        plot_mlp_optimizers(summary, figures / "mlp_optimizers.png")
        plot_rbf_centers(summary, figures / "rbf_centers.png")
        plot_architectures(summary, figures / "architectures_comparison.png")
    else:
        print(f"Skipped: {summary_path} not found (run build_summary.py)")

    lr_path = results_dir / "lr_study.csv"
    if lr_path.exists():
        table = pd.read_csv(lr_path)
        plot_lr_metric(
            table, "training_time_s", "training_time", figures / "lr_time.png", True
        )
        plot_lr_metric(table, "epochs", "epochs", figures / "lr_epochs.png", True)
        plot_lr_metric(
            table, "test_error_pct", "test_error", figures / "lr_error.png", False
        )
    else:
        print(f"Skipped: {lr_path} not found (run run_lr_study.py)")

    batch_path = results_dir / "batch_study.csv"
    if batch_path.exists():
        plot_batch_study(pd.read_csv(batch_path), figures / "batch_study.png")
    else:
        print(f"Skipped: {batch_path} not found (run run_batch_study.py)")

    print(f"Graphs saved to {figures}")


if __name__ == "__main__":
    main()
