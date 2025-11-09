"""
Feature engineering helpers for the Mai Shun Yun inventory dashboard.

These functions operate on the tidy tables produced by `data_loader`
and generate higher-level insights used by the Streamlit UI.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from data_loader import load_item_sales_data, load_recipe_book, load_sales_data

# Heuristic mapping from menu categories to the recipe book item names.
CATEGORY_TO_ITEMS: Dict[str, Tuple[str, ...]] = {
    "ramen": ("Beef Ramen", "Pork Ramen", "Chicken Ramen"),
    "tossed ramen": ("Beef Tossed Ramen", "Pork Tossed Ramen", "Chicken Tossed Ramen"),
    "tossed rice noodle": (
        "Beef Tossed Rice Noodles",
        "Pork Tossed Rice Noodles",
        "Chicken Tossed Rice Noodles",
    ),
    "fried rice": ("Beef Fried Rice", "Pork Fried Rice", "Chicken Fried Rice"),
    "fried chicken": ("Fried Wings", "Chicken Cutlet"),
    "lunch menu": (
        "Beef Ramen",
        "Pork Ramen",
        "Chicken Ramen",
        "Beef Fried Rice",
        "Chicken Fried Rice",
        "Pork Fried Rice",
    ),
    "all day menu": (
        "Beef Ramen",
        "Pork Ramen",
        "Chicken Ramen",
        "Beef Tossed Ramen",
        "Pork Tossed Ramen",
        "Chicken Tossed Ramen",
        "Fried Wings",
        "Chicken Cutlet",
    ),
}


def summarise_sales(sales: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregate sales by month and category with tidy numeric columns.
    """
    if sales is None:
        sales = load_sales_data()
    if sales.empty:
        return sales

    summary = (
        sales.groupby(["period", "category"], dropna=False)[["count", "amount"]]
        .sum()
        .reset_index()
    )
    summary["amount"] = summary["amount"].astype(float)
    summary["count"] = summary["count"].astype(float)
    return summary


def monthly_revenue_trend(sales: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return total revenue and transactions per month."""
    summary = summarise_sales(sales)
    if summary.empty:
        return summary

    trend = (
        summary.groupby("period")[["amount", "count"]]
        .sum()
        .sort_index()
        .reset_index()
    )
    trend["amount_change_pct"] = trend["amount"].pct_change()
    trend["count_change_pct"] = trend["count"].pct_change()
    return trend


def estimate_item_mix(sales: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Estimate item-level counts by distributing category totals across recipes.

    When the source data already contains `item_name`, it will be used directly.
    Otherwise, the function falls back to a heuristic mapping defined above.
    """
    if sales is None:
        sales = load_item_sales_data()

    if sales.empty:
        return pd.DataFrame(columns=["period", "item_name", "estimated_count"])

    if "item_name" in sales.columns:
        df = sales.copy()
        df["period"] = pd.to_datetime(df["period"])
        df["item_name"] = df["item_name"].astype(str).str.strip().str.title()
        if "estimated_count" in df.columns:
            metric_col = "estimated_count"
        else:
            metric_col = "count"
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0)
        result = (
            df.groupby(["period", "item_name"])[metric_col]
            .sum()
            .rename("estimated_count")
            .reset_index()
        )
        return result

    rows = []
    for _, row in sales.iterrows():
        category = str(row["category"]).strip().lower()
        period = row["period"]
        count = row["count"]
        items = CATEGORY_TO_ITEMS.get(category)
        if not items:
            continue
        share = count / len(items) if count and len(items) else 0
        for item in items:
            rows.append({"period": period, "item_name": item, "estimated_count": share})
    return pd.DataFrame(rows)


def estimate_ingredient_usage(
    item_mix: Optional[pd.DataFrame] = None,
    recipe_book: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Combine estimated item counts with recipe multipliers to produce ingredient demand.
    """
    if item_mix is None or item_mix.empty:
        item_mix = estimate_item_mix(load_item_sales_data())
    if recipe_book is None:
        recipe_book = load_recipe_book(melt=True)

    if item_mix.empty or recipe_book.empty:
        return pd.DataFrame(
            columns=["period", "ingredient", "estimated_usage", "item_name"]
        )

    merged = item_mix.merge(
        recipe_book,
        how="left",
        left_on="item_name",
        right_on="item_name",
    )
    merged["ingredient"] = merged["ingredient"].str.lower()
    merged["estimated_usage"] = (
        merged["estimated_count"] * merged["quantity_per_unit"]
    )
    usage = (
        merged.groupby(["period", "ingredient"])["estimated_usage"]
        .sum()
        .reset_index()
    )
    return usage

