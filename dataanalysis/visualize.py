import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = BASE_DIR / "MSYData"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR.parent / "MSYData"

def load_data(filename: str):
    file_path = DATA_DIR / filename
    df = pd.read_excel(file_path)
    if df.isnull().values.any():
        df = df.fillna(0)

    columns_to_drop = [
        col
        for col in df.columns
        if col and _normalize_column_name(col) in {"sourcepage", "sourcetable"}
    ]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum()).lower()


def visualize_data(data):
    if "Count" not in data.columns or "Amount" not in data.columns:
        available = ", ".join(sorted(map(str, data.columns))) or "none"
        missing = [
            column
            for column in ("Count", "Amount")
            if column not in data.columns
        ]
        raise ValueError(
            f"Missing required columns for plotting: {', '.join(missing)}. "
            f"Available columns: {available}"
        )

    counts = pd.to_numeric(data["Count"], errors="coerce")
    amounts = pd.to_numeric(data["Amount"], errors="coerce")

    valid_rows = ~(counts.isna() | amounts.isna())
    counts = counts[valid_rows]
    amounts = amounts[valid_rows]

    if counts.empty:
        raise ValueError(
            "No numeric values found in the 'Count' and 'Amount' columns "
            "after removing non-numeric entries."
        )

    preview_df = (
        pd.DataFrame({"Count": counts, "Amount": amounts})
        .head(10)
        .reset_index(drop=True)
    )
    print("Preview of data used for plotting:")
    print(preview_df.to_string(index=False))

    plt.figure()
    plt.bar(counts, amounts, width=0.8, align="center")

    plt.title("Count vs Amount")
    plt.xlabel("Count")
    plt.ylabel("Amount")
    plt.xticks(counts)
    plt.tight_layout()
    plt.show()

    return counts, amounts

if __name__ == "__main__":
    data = load_data("June_Data_Matrix.xlsx")
    visualize_data(data)