import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "MSYData"
DEFAULT_OUTPUT_DIR = BASE_DIR / "cleaned"
DROP_TARGETS = {"sourcepage", "sourcetable"}


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in str(name) if ch.isalnum()).lower()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        column
        for column in df.columns
        if column and _normalize_column_name(column) in DROP_TARGETS
    ]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    return df


def load_workbook(path: Path) -> dict[str, pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=None)
    if suffix == ".csv":
        return {path.stem: pd.read_csv(path)}
    raise ValueError(f"Unsupported file type for {path}")


def clean_file(path: Path, output_dir: Path) -> Path:
    workbook = load_workbook(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for sheet_name, df in workbook.items():
        cleaned = clean_dataframe(df)
        output_path = output_dir / f"{path.stem}_{_sanitize_sheet_name(sheet_name)}_cleaned.csv"
        cleaned.to_csv(output_path, index=False)
        output_paths.append(output_path)

    return output_paths


def iter_data_files(data_dir: Path):
    for path in sorted(data_dir.iterdir()):
        if path.name.startswith("~$"):
            continue
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".csv"}:
            yield path


def _sanitize_sheet_name(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))
    return sanitized.strip("_") or "Sheet"


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Remove 'source_page' and 'source_table' columns from data files "
            "and export cleaned CSV outputs."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing input files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write cleaned CSV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(args)


def main(argv=None) -> int:
    args = parse_args(argv)

    data_dir = args.data_dir
    if not data_dir.exists():
        alternative = BASE_DIR.parent / "MSYData"
        if alternative.exists():
            data_dir = alternative
        else:
            print(f"Data directory not found: {args.data_dir}", file=sys.stderr)
            return 1

    output_dir = args.output_dir

    processed_files = 0
    processed_outputs = 0
    for file_path in iter_data_files(data_dir):
        try:
            output_paths = clean_file(file_path, output_dir)
            for output_path in output_paths:
                print(f"Cleaned {file_path.name} ({output_path.stem}) -> {output_path.name}")
                processed_outputs += 1
            processed_files += 1
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to clean {file_path}: {exc}", file=sys.stderr)

    if processed_files == 0:
        print("No data files processed. Ensure the directory contains .xlsx or .csv files.")
        return 1

    print(
        f"Finished cleaning {processed_files} files "
        f"into {output_dir} with {processed_outputs} CSV outputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

