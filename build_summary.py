"""Build the summary tables of all saved runs.

Reads `config.json` and `metrics.json` of every folder in `results/runs/`
and writes two files to `results/`:
    summary.csv     all runs, English column names (for further processing)
    summary.xlsx    Ukrainian column names (for the report), two sheets:
                    the main table and the runs with a stricter tolerance

Units: training time in seconds, classification time in milliseconds,
errors in percent.

Run from the project root: python build_summary.py
"""

import json
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font

from ffnn_lab import config
from ffnn_lab.storage import RUNS_DIRNAME

ARCHITECTURE_NAMES = {
    "single_layer": ("Single-layer perceptron", "Одношаровий перцептрон"),
    "mlp": ("Multilayer perceptron", "Багатошаровий перцептрон"),
    "rbf": ("RBF Network", "RBF-мережа"),
}
METHOD_NAMES_UK = {
    "rosenblatt": "Правило Розенблатта",
    "delta": "Дельта-правило",
    "sgd": "SGD",
    "sgd_momentum": "SGD з моментом",
    "rmsprop": "RMSprop",
    "adam": "Adam",
    "kmeans-lstsq": "K-means + метод найменших квадратів",
}

# (column in the CSV, column in the Excel file). The first ten columns are
# the table required by the lab methodology, the rest is extra information.
TABLE_COLUMNS = [
    ("architecture", "Назва архітектури"),
    ("n_layers", "Кількість шарів"),
    ("neurons", "Кількість нейронів у шарах"),
    ("activations", "Функції активації"),
    ("training_method", "Метод навчання"),
    ("training_time_s", "Час навчання, с"),
    ("train_classification_time_ms", "Час класифікації (train), мс"),
    ("train_error_pct", "Помилка (train), %"),
    ("test_classification_time_ms", "Час класифікації (test), мс"),
    ("test_error_pct", "Помилка (test), %"),
    ("epochs", "Кількість епох"),
    ("stopped_early", "Зупинка за критерієм"),
]
CSV_ONLY_COLUMNS = ["learning_rate", "min_delta", "run_id"]


def load_runs(results_dir: Path = config.RESULTS_DIR) -> pd.DataFrame:
    """Collect all saved runs into one table, one row per run.

    Raises:
        FileNotFoundError: If there are no saved runs.
    """
    runs_dir = results_dir / RUNS_DIRNAME
    folders = [p for p in runs_dir.iterdir() if p.is_dir()] if runs_dir.is_dir() else []
    if not folders:
        raise FileNotFoundError(f"No runs found in {runs_dir}")

    rows = []
    for folder in folders:
        run_config = _read_json(folder / "config.json")
        metrics = _read_json(folder / "metrics.json")
        # The RBF network has no epochs, so these two fields do not apply.
        has_epochs = run_config["kind"] != "rbf"
        rows.append(
            {
                "kind": run_config["kind"],
                "architecture": ARCHITECTURE_NAMES[run_config["kind"]][0],
                "n_layers": run_config["n_layers"],
                "neurons": run_config["neurons"],
                "activations": run_config["activations"],
                "training_method": run_config["training_method"],
                "training_time_s": round(metrics["training_time_s"], 4),
                "train_classification_time_ms": round(
                    metrics["train"]["classification_time_s"] * 1000, 3
                ),
                "train_error_pct": round(metrics["train"]["error"] * 100, 2),
                "test_classification_time_ms": round(
                    metrics["test"]["classification_time_s"] * 1000, 3
                ),
                "test_error_pct": round(metrics["test"]["error"] * 100, 2),
                "epochs": metrics["epochs_run"] if has_epochs else None,
                "stopped_early": metrics["stopped_early"] if has_epochs else None,
                "learning_rate": run_config["settings"].get("learning_rate"),
                "min_delta": run_config["training_settings"].get("min_delta"),
                "run_id": run_config["run_id"],
            }
        )

    table = pd.DataFrame(rows)
    # Whole numbers with gaps (the RBF network has no epochs).
    table["epochs"] = table["epochs"].astype("Int64")
    order = [_sort_key(row) for row in rows]
    return table.iloc[sorted(range(len(rows)), key=order.__getitem__)].reset_index(
        drop=True
    )


def split_main_and_strict(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the runs into the main ones and those with another `min_delta`."""
    is_strict = table["min_delta"].notna() & (table["min_delta"] != config.MIN_DELTA)
    return table[~is_strict], table[is_strict]


def write_csv(table: pd.DataFrame, path: Path) -> None:
    """Write all runs with English column names."""
    columns = [name for name, _ in TABLE_COLUMNS] + CSV_ONLY_COLUMNS
    table[columns].to_csv(path, index=False)


def write_excel(table: pd.DataFrame, path: Path) -> None:
    """Write the report table with Ukrainian names, one sheet per group."""
    main, strict = split_main_and_strict(table)
    sheets = {"Основна таблиця": main}
    if not strict.empty:
        sheets[f"Додатково, min_delta {strict['min_delta'].iloc[0]:g}"] = strict

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, part in sheets.items():
            view = _to_report_view(part)
            view.to_excel(writer, sheet_name=sheet_name, index=False)
            _format_sheet(writer.sheets[sheet_name], view)


def _to_report_view(part: pd.DataFrame) -> pd.DataFrame:
    """Return the rows with Ukrainian names and values for the report."""
    view = part.copy()
    view["architecture"] = view["kind"].map(lambda k: ARCHITECTURE_NAMES[k][1])
    view["training_method"] = view["training_method"].map(
        lambda m: METHOD_NAMES_UK.get(m, m)
    )
    view["stopped_early"] = view["stopped_early"].map({True: "Так", False: "Ні"})
    columns = dict(TABLE_COLUMNS)
    return view[list(columns)].rename(columns=columns)


def _format_sheet(sheet, view: pd.DataFrame) -> None:
    """Bold wrapped header, frozen first row and readable column widths."""
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    sheet.freeze_panes = "A2"
    for index, name in enumerate(view.columns, start=1):
        longest = max([len(str(name)) // 2] + [len(str(v)) for v in view[name]])
        letter = sheet.cell(row=1, column=index).column_letter
        sheet.column_dimensions[letter].width = min(max(longest + 2, 10), 40)


def _sort_key(row: dict) -> tuple:
    """Order of the rows: by model type, then in the order of the config."""
    kind_rank = list(ARCHITECTURE_NAMES).index(row["kind"])
    # Strict-tolerance runs go after the main ones of the same kind.
    strict_rank = int(
        row["min_delta"] is not None and row["min_delta"] != config.MIN_DELTA
    )
    if row["kind"] == "single_layer":
        setup = (row["activations"], row["training_method"])
        position = config.SINGLE_LAYER_SETUPS.index(setup)
    elif row["kind"] == "mlp":
        set_names = list(config.MLP_ACTIVATION_SETS)
        position = (
            set_names.index(row["activations"].replace(", ", "-")),
            config.MLP_OPTIMIZERS.index(row["training_method"]),
        )
    else:
        position = int(row["neurons"].split("-")[0])
    return (kind_rank, strict_rank, position)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(results_dir: Path = config.RESULTS_DIR) -> None:
    """Build `summary.csv` and `summary.xlsx` from the saved runs."""
    table = load_runs(results_dir)
    csv_path = results_dir / "summary.csv"
    xlsx_path = results_dir / "summary.xlsx"
    write_csv(table, csv_path)
    write_excel(table, xlsx_path)

    main_part, strict_part = split_main_and_strict(table)
    print(f"Runs: {len(table)} (main: {len(main_part)}, strict: {len(strict_part)})")
    print(f"Saved: {csv_path}")
    print(f"Saved: {xlsx_path}")


if __name__ == "__main__":
    main()
