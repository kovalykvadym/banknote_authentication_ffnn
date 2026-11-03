"""Dataset loading, validation, train/test splitting and feature scaling.

Every model receives data prepared by this module, so all models
are compared on exactly the same samples.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ffnn_lab import config


@dataclass(frozen=True)
class DataSplit:
    """Prepared train and test data.

    Attributes:
        X_train: Scaled training features, shape (n_train, n_features).
        X_test: Scaled test features, shape (n_test, n_features).
        y_train: Training labels (0 or 1), shape (n_train,).
        y_test: Test labels (0 or 1), shape (n_test,).
        scaler: Scaler fitted on the training features only.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def load_dataset(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV file and check that its structure is as expected.

    Args:
        path: Location of the CSV file with a header row.

    Returns:
        A table containing only the feature columns and the target column,
        in the order defined in the configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file, the file is empty or is not
            a valid UTF-8 CSV, required columns are missing, or there are
            no data rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"Dataset file has no readable content: {path}",
        ) from error
    except pd.errors.ParserError as error:
        raise ValueError(
            f"Dataset file is not a valid CSV: {path} ({error})",
        ) from error
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Dataset file is not UTF-8 text: {path}",
        ) from error

    df.columns = df.columns.str.strip()

    required = [*config.FEATURE_COLUMNS, config.TARGET_COLUMN]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}; found {list(df.columns)}. "
            "Does the first line of the file contain the header?"
        )
    if df.empty:
        raise ValueError(f"Dataset has a header but no data rows: {path}")

    return df[required]


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the feature and target columns to numeric types.

    Args:
        df: Table returned by `load_dataset`.

    Returns:
        A copy of the table with all required columns converted to numbers.
        The input table is not modified.

    Raises:
        ValueError: If a required column is missing or contains values
            that cannot be converted to numbers.
    """
    required_columns = [
        *config.FEATURE_COLUMNS,
        config.TARGET_COLUMN,
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns for numeric conversion: {missing_columns}",
        )

    df = df.copy()

    for column in required_columns:
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Column '{column}' contains non-numeric values: {error}"
            ) from error

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Check that the table is ready for training.

    The table must contain only numeric columns, with no missing or
    infinite values. The target column must contain only the labels 0 and 1,
    and both classes must be present. The table is not modified.

    Args:
        df: Table returned by `convert_numeric_columns`.

    Raises:
        ValueError: If any of the checks above fails.
    """
    non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns: {non_numeric}")

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(f"Missing values per column: {missing_counts.to_dict()}")

    infinite_counts = np.isinf(df.select_dtypes(include=["number"])).sum()
    infinite_counts = infinite_counts[infinite_counts > 0]
    if not infinite_counts.empty:
        raise ValueError(
            f"Infinite values per column: {infinite_counts.to_dict()}",
        )

    labels = set(df[config.TARGET_COLUMN].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"Labels must be 0 or 1, found: {sorted(labels)}")
    if len(labels) < 2:
        raise ValueError("Only one class is present in the dataset")


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and labels into train and test parts.

    The split is stratified, so both parts keep the same class proportions.

    Args:
        X: Feature table.
        y: Label series.
        random_state: Seed that makes the split reproducible.

    Returns:
        A tuple (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


def to_bipolar(labels: np.ndarray) -> np.ndarray:
    """Convert labels from {0, 1} to {-1, +1}.

    Useful for models whose output lies in [-1, 1] (tanh, threshold).
    """
    return 2 * labels - 1


def prepare_dataset(random_state: int = config.RANDOM_SEED) -> DataSplit:
    """Load, validate, split and scale the dataset.

    The scaler is fitted on the training part only, so no information
    from the test part leaks into training.

    Args:
        random_state: Seed that makes the split reproducible.

    Returns:
        A `DataSplit` with scaled features and integer labels (0 or 1).
    """
    df = load_dataset()
    df = convert_numeric_columns(df)
    validate_dataset(df)

    X = df[list(config.FEATURE_COLUMNS)]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.to_numpy(dtype=int),
        y_test=y_test.to_numpy(dtype=int),
        scaler=scaler,
    )
