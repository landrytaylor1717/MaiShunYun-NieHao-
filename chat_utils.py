from __future__ import annotations

from typing import List

import pandas as pd


DESCRIPTOR_NAMES = {
    "additonal",
    "additional",
    "all day menu",
    "appetizer",
    "combo items",
    "combo items donot delete",
    "dessert",
    "desert",
    "drink",
    "drinks",
    "fruit tea",
    "lunch menu",
    "lunch special",
    "milk tea",
    "open food",
    "prep item",
    "signature drinks",
}

DRINK_KEYWORDS = [
    "tea",
    "drink",
    "juice",
    "soda",
    "ramune",
    "milk tea",
    "boba",
    "water",
    "coffee",
    "smoothie",
    "lemonade",
    "sparkling",
]

APPETIZER_KEYWORDS = [
    "appetizer",
    "dumpling",
    "bun",
    "roll",
    "wonton",
    "fries",
    "salad",
    "egg roll",
    "spring roll",
    "rangoon",
    "tempura",
    "wing",
    "nugget",
    "tender",
]

DESCRIPTOR_SET = {name.lower() for name in DESCRIPTOR_NAMES}


def filter_individual_items(item_sales: pd.DataFrame) -> pd.DataFrame:
    if item_sales.empty:
        return item_sales
    return item_sales[
        ~item_sales["item_name"].astype(str).str.strip().str.lower().isin(DESCRIPTOR_SET)
    ].copy()


def filter_descriptor_categories(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    return summary[
        summary["category"].astype(str).str.strip().str.lower().isin(DESCRIPTOR_SET)
    ].copy()


def filter_drink_items(item_sales: pd.DataFrame) -> pd.DataFrame:
    if item_sales.empty:
        return item_sales
    pattern = "|".join(DRINK_KEYWORDS)
    mask = item_sales["item_name"].astype(str).str.contains(pattern, case=False, na=False)
    exclude_mask = item_sales["item_name"].astype(str).str.contains(
        "bun|dumpling|soup", case=False, na=False
    )
    return item_sales[mask & ~exclude_mask].copy()


def filter_appetizer_items(item_sales: pd.DataFrame) -> pd.DataFrame:
    if item_sales.empty:
        return item_sales
    pattern = "|".join(APPETIZER_KEYWORDS)
    mask = item_sales["item_name"].astype(str).str.contains(pattern, case=False, na=False)
    drink_names = filter_drink_items(item_sales)["item_name"].unique()
    return item_sales[mask & ~item_sales["item_name"].isin(drink_names)].copy()


def compute_item_trend(item_sales: pd.DataFrame) -> pd.DataFrame:
    if item_sales.empty:
        return pd.DataFrame(
            columns=["period", "amount", "count", "amount_change_pct", "count_change_pct"]
        )
    trend = (
        item_sales.groupby("period")[["amount", "count"]]
        .sum()
        .sort_index()
        .reset_index()
    )
    trend["amount_change_pct"] = trend["amount"].pct_change()
    trend["count_change_pct"] = trend["count"].pct_change()
    return trend

