import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = BASE_DIR / "MSYData"

def load_data(filename: str):
    file_path = DATA_DIR / filename
    df = pd.read_excel(file_path)
    if df.isnull().values.any():
        df = df.fillna(0)
    return df

def visualize_data(data, output_path: Path = BASE_DIR / "visualization.png"):
    numeric_data = data.select_dtypes(include="number")
    if numeric_data.empty:
        raise ValueError("No numeric columns available for plotting.")

    plt.figure()
    for column in numeric_data.columns:
        plt.plot(numeric_data.index, numeric_data[column], label=column)

    plt.title("June Data Metrics")
    plt.xlabel("Row Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    data = load_data("June_Data_Matrix.xlsx")
    visualize_data(data)