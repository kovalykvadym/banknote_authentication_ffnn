import numpy as np

from ffnn_lab import config
from ffnn_lab.data import (
    DataSplit,
    convert_numeric_columns,
    load_dataset,
    prepare_dataset,
)

OTHER_SEED = 7
CLASS_SHARE_TOLERANCE = 0.001
SCALING_TOLERANCE = 1e-9


def print_header(title: str) -> None:
    """Print a section title surrounded by empty lines."""
    print()
    print(f"=== {title} ===")


def status(ok: bool) -> str:
    """Return a short text label for a passed or failed check."""
    return "OK" if ok else "CHECK"


def print_shapes(split: DataSplit) -> None:
    """Print the shapes of the four data arrays.

    Expected: (960, 4), (412, 4), (960,), (412,).
    """
    print_header("Array shapes")
    print(f"X_train:\t{split.X_train.shape}")
    print(f"X_test:\t\t{split.X_test.shape}")
    print(f"y_train:\t{split.y_train.shape}")
    print(f"y_test:\t\t{split.y_test.shape}")


def print_class_shares(split: DataSplit) -> None:
    """Print the share of class 1 in the train and test labels.

    The split is stratified, so both shares must be almost equal
    (about 0.4446 for this dataset).
    """
    print_header("Share of class 1")
    train_share = float(np.mean(split.y_train))
    test_share = float(np.mean(split.y_test))
    difference = abs(train_share - test_share)

    print(f"train:\t\t{train_share:.4f}")
    print(f"test:\t\t{test_share:.4f}")
    print(
        f"difference:\t{difference:.4f} "
        f"[{status(difference < CLASS_SHARE_TOLERANCE)}]"
    )


def print_scaling_stats(split: DataSplit) -> None:
    """Print the mean and standard deviation of every scaled feature.

    The scaler is fitted on the training part only. Therefore, the training
    features have mean 0 and std 1, while the test features are close to
    these values but not exactly equal. If the test part is exactly 0 and 1,
    the scaler has seen the test data, which is a data leakage.
    """
    print_header("Feature statistics after scaling")
    print(f"features:\t{list(config.FEATURE_COLUMNS)}")

    train_mean = split.X_train.mean(axis=0)
    train_std = split.X_train.std(axis=0)
    test_mean = split.X_test.mean(axis=0)
    test_std = split.X_test.std(axis=0)

    print(f"train_mean:\t{np.round(train_mean, 4)}")
    print(f"train_std:\t{np.round(train_std, 4)}")
    print(f"test_mean:\t{np.round(test_mean, 4)}")
    print(f"test_std:\t{np.round(test_std, 4)}")

    train_ok = bool(
        np.allclose(train_mean, 0.0, atol=SCALING_TOLERANCE)
        and np.allclose(train_std, 1.0, atol=SCALING_TOLERANCE)
    )
    test_exact = bool(
        np.allclose(test_mean, 0.0, atol=SCALING_TOLERANCE)
        and np.allclose(test_std, 1.0, atol=SCALING_TOLERANCE)
    )
    print(f"train is exactly 0 and 1:\t\t[{status(train_ok)}]")
    print(f"test is NOT exactly 0 and 1:\t[{status(not test_exact)}]")


def check_reproducibility() -> None:
    """Check that the same seed gives the same split and another seed does not."""
    print_header("Reproducibility")
    first = prepare_dataset(random_state=config.RANDOM_SEED)
    second = prepare_dataset(random_state=config.RANDOM_SEED)
    other = prepare_dataset(random_state=OTHER_SEED)

    same_seed_equal = np.array_equal(first.X_train, second.X_train)
    other_seed_equal = np.array_equal(first.X_train, other.X_train)

    print(
        f"seed {config.RANDOM_SEED} twice, X_train equal: {same_seed_equal} "
        f"[{status(same_seed_equal)}]"
    )
    print(
        f"seed {config.RANDOM_SEED} vs {OTHER_SEED}, X_train equal: "
        f"{other_seed_equal} [{status(not other_seed_equal)}]"
    )


def check_duplicates() -> None:
    """Print how many fully duplicated rows the dataset contains.

    The rows are not removed, but identical rows can end up in both
    the train and the test part. This must be mentioned in the report.
    """
    print_header("Duplicate rows")
    df = convert_numeric_columns(load_dataset())
    duplicates = int(df.duplicated().sum())

    print(f"rows in the dataset:\t{len(df)}")
    print(f"duplicated rows:\t\t{duplicates}")
    print(f"share of duplicates:\t{duplicates / len(df):.4f}")


def main() -> None:
    """Prepare the dataset and run all checks."""
    split = prepare_dataset()

    print_shapes(split)
    print_class_shares(split)
    print_scaling_stats(split)
    check_reproducibility()
    check_duplicates()


if __name__ == "__main__":
    main()
