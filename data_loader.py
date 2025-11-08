"""
Utilities for loading and lightly shaping Mai Shun Yun data assets.

These helpers keep raw-data handling in one place so the rest of the app
can assume tidy, ready-to-analyze DataFrames. Each function is cached to
avoid repeated I/O when the Streamlit dashboard reruns.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "MSYData"
CLEANED_DIR = BASE_DIR / "dataanalysis" / "cleaned"

_MONTH_ORDER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _clean_currency(series: pd.Series) -> pd.Series:
    """Convert a dollar-formatted string column into floats."""
    return (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", "0")
        .astype(float)
    )


def _clean_count(series: pd.Series) -> pd.Series:
    """Ensure counts are numeric even when commas are present."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .replace("", "0")
        .astype(float)
    )


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with snake_case column names."""
    df = df.copy()
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("/", "_") for col in df.columns
    ]
    return df


def _parse_period_from_filename(path: Path) -> Optional[pd.Timestamp]:
    """Infer a period (month) from filenames like 'June_Data_Matrix.xlsx'."""
    stem = path.stem.lower()
    for month, month_idx in _MONTH_ORDER.items():
        if month in stem:
            # Attempt to parse a 4-digit year; default to current year.
            tokens = [t for t in stem.replace("_", " ").split() if t.isdigit()]
            year = pd.Timestamp.today().year
            for token in tokens:
                if len(token) >= 4:
                    year_candidate = int(token[:4])
                    if 1900 <= year_candidate <= 2100:
                        year = year_candidate
                        break
            return pd.Timestamp(year=year, month=month_idx, day=1)
    return None


def _normalise_category_column(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the column name used for menu category (legacy Excel files)."""
    rename_map = {}
    if "Category" in df.columns and "Group" not in df.columns:
        rename_map["Category"] = "Group"
    return df.rename(columns=rename_map)


def _load_cleaned_category_sales() -> pd.DataFrame:
    if not CLEANED_DIR.exists():
        return pd.DataFrame(columns=["period", "category", "count", "amount"])

    frames: list[pd.DataFrame] = []
    for path in sorted(CLEANED_DIR.glob("*data_2_cleaned.csv")):
        df = pd.read_csv(path)
        df = _normalise_columns(df)
        required_cols = {"category", "count", "amount"}
        if not required_cols.issubset(df.columns):
            continue
        period = _parse_period_from_filename(path)
        df = df.assign(
            period=period,
            category=df["category"].astype(str).str.strip().str.title(),
            count=_clean_count(df["count"]),
            amount=_clean_currency(df["amount"]),
        )[["period", "category", "count", "amount"]]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["period", "category", "count", "amount"])

    sales = pd.concat(frames, ignore_index=True)
    sales = sales.sort_values(
        ["period", "category"],
        key=lambda col: col.fillna(pd.Timestamp.min) if col.name == "period" else col,
    )
    return sales.reset_index(drop=True)


def _load_legacy_category_sales() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not DATA_DIR.exists():
        return pd.DataFrame(columns=["period", "category", "count", "amount"])

    for path in sorted(DATA_DIR.glob("*Data_Matrix*.xlsx")):
        df = pd.read_excel(path)
        df = _normalise_category_column(df)
        period = _parse_period_from_filename(path)
        if "Group" not in df.columns:
            continue
        df = df.assign(
            period=period,
            category=df["Group"].astype(str).str.strip().str.title(),
            count=_clean_count(df["Count"]),
            amount=_clean_currency(df["Amount"]),
        )[["period", "category", "count", "amount"]]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["period", "category", "count", "amount"])

    sales = pd.concat(frames, ignore_index=True)
    sales = sales.sort_values(
        ["period", "category"],
        key=lambda col: col.fillna(pd.Timestamp.min) if col.name == "period" else col,
    )
    return sales.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_sales_data() -> pd.DataFrame:
    """
    Load monthly sales summaries aggregated by category.

    Returns columns: period (Timestamp), category, count (float), amount (float).
    """
    cleaned = _load_cleaned_category_sales()
    if not cleaned.empty:
        return cleaned
    return _load_legacy_category_sales()


def _load_cleaned_item_sales() -> pd.DataFrame:
    if not CLEANED_DIR.exists():
        return pd.DataFrame(columns=["period", "item_name", "count", "amount"])

    frames: list[pd.DataFrame] = []
    for path in sorted(CLEANED_DIR.glob("*data_3_cleaned.csv")):
        df = pd.read_csv(path)
        df = _normalise_columns(df)
        required_cols = {"item_name", "count", "amount"}
        if not required_cols.issubset(df.columns):
            continue
        period = _parse_period_from_filename(path)
        df = df.assign(
            period=period,
            item_name=df["item_name"].astype(str).str.strip().str.title(),
            count=_clean_count(df["count"]),
            amount=_clean_currency(df["amount"]),
        )[["period", "item_name", "count", "amount"]]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["period", "item_name", "count", "amount"])

    sales = pd.concat(frames, ignore_index=True)
    sales = sales.sort_values(
        ["period", "count"], key=lambda col: col.fillna(pd.Timestamp.min)
    )
    return sales.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_item_sales_data() -> pd.DataFrame:
    """
    Load item-level sales data derived from cleaned exports.

    Returns columns: period (Timestamp), item_name, count, amount.
    """
    return _load_cleaned_item_sales()


def _resolve_recipe_path() -> Path:
    cleaned_path = (
        CLEANED_DIR / "MSY Data - Ingredient_MSY_Data_-_Ingredient_cleaned.csv"
    )
    if cleaned_path.exists():
        return cleaned_path
    return DATA_DIR / "MSY Data - Ingredient.csv"


def _resolve_shipment_path() -> Path:
    cleaned_path = (
        CLEANED_DIR / "MSY Data - Shipment_MSY_Data_-_Shipment_cleaned.csv"
    )
    if cleaned_path.exists():
        return cleaned_path
    return DATA_DIR / "MSY Data - Shipment.csv"


@lru_cache(maxsize=1)
def load_recipe_book(melt: bool = True) -> pd.DataFrame:
    """
    Load the ingredient usage-per-menu-item reference sheet.

    Parameters
    ----------
    melt
        When True (default), returns a long-form table with one ingredient per row.
        When False, returns the original wide layout.
    """
    path = _resolve_recipe_path()
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df = df.rename(columns={"Item name": "item_name"})
    df["item_name"] = df["item_name"].str.strip()

    if not melt:
        return df

    long_df = df.melt(
        id_vars="item_name",
        var_name="ingredient",
        value_name="quantity_per_unit",
    )
    long_df["ingredient"] = long_df["ingredient"].str.strip().str.lower()
    long_df = long_df.replace({"quantity_per_unit": {"": 0, "nan": 0}})
    long_df["quantity_per_unit"] = (
        pd.to_numeric(long_df["quantity_per_unit"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    long_df = long_df[long_df["quantity_per_unit"] > 0]
    return long_df.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_shipments() -> pd.DataFrame:
    """
    Load recurring shipment schedules.

    Returns columns: ingredient, quantity_per_shipment, unit, shipments, frequency.
    """
    path = _resolve_shipment_path()
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df = df.rename(
        columns={
            "Ingredient": "ingredient",
            "Quantity per shipment": "quantity_per_shipment",
            "Unit of shipment": "unit",
            "Number of shipments": "shipments",
            "frequency": "frequency",
        }
    )
    df["ingredient"] = df["ingredient"].str.strip().str.lower()
    df["frequency"] = df["frequency"].str.strip().str.lower()
    df["shipments"] = pd.to_numeric(df["shipments"], errors="coerce").fillna(0).astype(int)
    df["quantity_per_shipment"] = pd.to_numeric(
        df["quantity_per_shipment"], errors="coerce"
    ).fillna(0.0)
    return df


def list_available_periods() -> Iterable[pd.Timestamp]:
    """Return the unique reporting periods present in the sales data."""
    sales = load_sales_data()
    periods = sales["period"].dropna().unique()
    return sorted(pd.to_datetime(periods)) if len(periods) else []


def latest_period() -> Optional[pd.Timestamp]:
    """Convenience helper returning the most recent period with data."""
    periods = list_available_periods()
    if not periods:
        return None
    return periods[-1]

