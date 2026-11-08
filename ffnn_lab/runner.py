"""One experiment run: train a model, evaluate it and save everything.

The experiment scripts for all three model types use `run_experiment`, so
the training time, the errors and the saved files are produced in exactly
the same way for every architecture.
"""

import statistics
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from ffnn_lab import config
from ffnn_lab.data import DataSplit
from ffnn_lab.evaluation import Evaluation, evaluate
from ffnn_lab.storage import ModelDescription, build_run_id, describe_model, save_run
from ffnn_lab.utils import TrainingResult


@dataclass(frozen=True)
class RunSummary:
    """Outcome of `run_experiment`.

    Attributes:
        run_id: Name of the run (and of its folder).
        description: Common description of the trained model.
        training: Result of the last training; its `training_time` is the
            mean over all repeats.
        training_times: Training time of every repeat, in seconds.
        train_eval: Evaluation on the training set.
        test_eval: Evaluation on the test set.
        run_dir: Folder with the saved files.
    """

    run_id: str
    description: ModelDescription
    training: TrainingResult
    training_times: list[float]
    train_eval: Evaluation
    test_eval: Evaluation
    run_dir: Path


def run_experiment(
    make_model: Callable[[], object],
    split: DataSplit,
    fit_kwargs: dict | None = None,
    repeats: int = config.N_REPEATS,
    tag: str = "",
    results_dir: Path = config.RESULTS_DIR,
) -> RunSummary:
    """Train a model `repeats` times, evaluate the last one and save the run.

    Every repeat builds a fresh model with the same seed, so the weights and
    the errors are the same and only the measured time differs. The mean
    time of all repeats is stored as the training time, because a single
    measurement is noisy.

    Args:
        make_model: Function without arguments that returns a new untrained
            model (use `lambda` or `functools.partial`).
        split: The prepared train/test data.
        fit_kwargs: Extra arguments of `model.fit` (batch size, patience, ...).
        repeats: How many times to train.
        tag: Optional suffix of the run ID (see `build_run_id`).
        results_dir: Root folder for the results.

    Raises:
        ValueError: If `repeats` is less than 1.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    fit_kwargs = dict(fit_kwargs or {})

    times: list[float] = []
    for _ in range(repeats):
        model = make_model()
        training = model.fit(split.X_train, split.y_train, **fit_kwargs)
        times.append(training.training_time)

    mean_training = replace(training, training_time=statistics.fmean(times))
    train_eval = evaluate(model, split.X_train, split.y_train)
    test_eval = evaluate(model, split.X_test, split.y_test)

    description = describe_model(model)
    run_id = build_run_id(description, tag)
    run_dir = save_run(
        run_id,
        model,
        mean_training,
        train_eval,
        test_eval,
        split,
        training_settings={**fit_kwargs, "training_time_repeats": repeats},
        results_dir=results_dir,
    )
    return RunSummary(
        run_id=run_id,
        description=description,
        training=mean_training,
        training_times=times,
        train_eval=train_eval,
        test_eval=test_eval,
        run_dir=run_dir,
    )
