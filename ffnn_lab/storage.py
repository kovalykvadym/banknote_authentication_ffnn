"""Saving a trained model and its results to disk.

Every run gets its own folder `results/runs/<run_id>/` with:
    config.json structure, activations, training method, settings
    metrics.json times, errors, number of epochs
    initial_params.npz parameters before training (if the model has them)
    trained_params.npz parameters after training
    predictions_train.csv inputs, true class, output and predicted class
    predictions_test.csv the same for the test set

The models know nothing about files. This module describes them in one
common way, so the same code can save any of the three model types.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ffnn_lab import config
from ffnn_lab.data import DataSplit
from ffnn_lab.evaluation import Evaluation
from ffnn_lab.mlp import MultilayerPerceptron
from ffnn_lab.perceptron import SingleLayerPerceptron
from ffnn_lab.rbf import KMEANS_RESTARTS, RBFNetwork
from ffnn_lab.utils import TrainingResult

RUNS_DIRNAME = "runs"


@dataclass(frozen=True)
class ModelDescription:
    """Common description of any model, used for saving and for the table.

    Attributes:
        kind: "single_layer", "mlp" or "rbf".
        n_layers: Number of layers (the input is not counted).
        neurons: Neurons per layer as text, e.g. "4-1".
        activations: Activation functions per layer as text, e.g. "tanh, tanh".
        training_method: Learning rule, optimizer or training scheme.
        settings: Model settings (learning rate, seed, ...).
        parameters: Trained parameters by name.
        initial_parameters: Parameters before training, or None if the
            model has no such notion (the RBF network).
    """

    kind: str
    n_layers: int
    neurons: str
    activations: str
    training_method: str
    settings: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    initial_parameters: dict | None = None


def describe_model(model) -> ModelDescription:
    """Describe a model of any supported type in the common form.

    Raises:
        TypeError: If the model type is not supported.
        ValueError: If the RBF network is not trained yet.
    """
    if isinstance(model, SingleLayerPerceptron):
        return ModelDescription(
            kind="single_layer",
            n_layers=1,
            neurons="1",
            activations=model.activation.name,
            training_method=model.learning_rule,
            settings={
                "learning_rate": model.learning_rate,
                "seed": model.seed,
                "n_features": len(model.weights),
            },
            parameters={"weights": model.weights, "bias": np.array(model.bias)},
            initial_parameters={
                "weights": model.initial_weights,
                "bias": np.array(model.initial_bias),
            },
        )

    if isinstance(model, MultilayerPerceptron):
        n_features, hidden_units = model.params[0].shape
        names = ("W1", "b1", "W2", "b2")
        return ModelDescription(
            kind="mlp",
            n_layers=2,
            neurons=f"{hidden_units}-1",
            activations=(
                f"{model.hidden_activation.name}, {model.output_activation.name}"
            ),
            training_method=model.optimizer_name,
            settings={
                "learning_rate": model.learning_rate,
                "seed": model.seed,
                "n_features": n_features,
            },
            parameters=dict(zip(names, model.params)),
            initial_parameters=dict(zip(names, model.initial_params)),
        )

    if isinstance(model, RBFNetwork):
        if model.centers is None:
            raise ValueError("The RBF network is not trained yet")
        return ModelDescription(
            kind="rbf",
            n_layers=2,
            neurons=f"{model.n_centers}-1",
            activations="gaussian, linear",
            training_method="kmeans-lstsq",
            settings={
                "seed": model.seed,
                "n_features": model.centers.shape[1],
                "kmeans_restarts": KMEANS_RESTARTS,
                "sigma": model.sigma,
            },
            parameters={
                "centers": model.centers,
                "sigma": np.array(model.sigma),
                "weights": model.weights,
                "bias": np.array(model.bias),
            },
            initial_parameters=None,
        )

    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def build_run_id(description: ModelDescription, tag: str = "") -> str:
    """Build a readable unique name of a run.

    Example: "mlp_4-1_tanh-tanh_adam_lr0.01_seed42". The optional `tag`
    is appended at the end, e.g. "md1e-06" for a stricter stopping tolerance.
    """
    parts = [
        description.kind,
        description.neurons,
        description.activations.replace(", ", "-"),
        description.training_method,
    ]
    learning_rate = description.settings.get("learning_rate")
    if learning_rate is not None:
        parts.append(f"lr{learning_rate:g}")
    parts.append(f"seed{description.settings['seed']}")
    if tag:
        parts.append(tag)
    return "_".join(parts)


def save_run(
    run_id: str,
    model,
    training: TrainingResult,
    train_eval: Evaluation,
    test_eval: Evaluation,
    split: DataSplit,
    training_settings: dict | None = None,
    split_seed: int = config.RANDOM_SEED,
    results_dir: Path = config.RESULTS_DIR,
) -> Path:
    """Save everything about one run and return its folder.

    An existing folder with the same `run_id` is overwritten.

    Args:
        run_id: Name of the run, see `build_run_id`.
        model: A trained model of a supported type.
        training: Result of `model.fit`.
        train_eval: Evaluation on the training set.
        test_eval: Evaluation on the test set.
        split: The data the model was trained and tested on.
        training_settings: Arguments of `fit` (batch size, patience, ...).
        split_seed: Seed of the train/test split.
        results_dir: Root folder for the results.
    """
    description = describe_model(model)
    run_dir = results_dir / RUNS_DIRNAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_id": run_id,
        "kind": description.kind,
        "n_layers": description.n_layers,
        "neurons": description.neurons,
        "activations": description.activations,
        "training_method": description.training_method,
        "settings": description.settings,
        "training_settings": training_settings or {},
        "data": {
            "feature_columns": list(config.FEATURE_COLUMNS),
            "train_samples": len(split.X_train),
            "test_samples": len(split.X_test),
            "split_seed": split_seed,
            "test_size": config.TEST_SIZE,
            "scaling": "StandardScaler fitted on the training set",
        },
    }
    metrics = {
        "training_time_s": training.training_time,
        "epochs_run": training.epochs_run,
        "stopped_early": training.stopped_early,
        "final_training_loss": (
            training.loss_history[-1] if training.loss_history else None
        ),
        "train": {
            "error": train_eval.error,
            "classification_time_s": train_eval.classification_time,
        },
        "test": {
            "error": test_eval.error,
            "classification_time_s": test_eval.classification_time,
        },
    }
    _write_json(run_dir / "config.json", run_config)
    _write_json(run_dir / "metrics.json", metrics)

    if description.initial_parameters is not None:
        np.savez(run_dir / "initial_params.npz", **description.initial_parameters)
    np.savez(run_dir / "trained_params.npz", **description.parameters)

    _write_predictions(
        run_dir / "predictions_train.csv", split.X_train, split.y_train, train_eval
    )
    _write_predictions(
        run_dir / "predictions_test.csv", split.X_test, split.y_test, test_eval
    )
    return run_dir


def _write_predictions(
    path: Path, X: np.ndarray, y: np.ndarray, evaluation: Evaluation
) -> None:
    """Write the model inputs, true classes, outputs and predicted classes."""
    table = pd.DataFrame(X, columns=list(config.FEATURE_COLUMNS))
    table["true_class"] = y
    table["output"] = evaluation.outputs
    table["predicted_class"] = evaluation.labels
    table.to_csv(path, index=False)


def _write_json(path: Path, data: dict) -> None:
    """Write a dictionary as readable JSON (numpy values are converted)."""
    path.write_text(
        json.dumps(_to_jsonable(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _to_jsonable(value):
    """Convert numpy values to plain Python and non-finite numbers to None."""
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
