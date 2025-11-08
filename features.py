"""
Feature engineering helpers for the Mai Shun Yun inventory dashboard.

These functions operate on the tidy tables produced by `data_loader`
and generate higher-level insights used by the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from data_loader import load_recipe_book, load_sales_data, load_shipments
from forecast import ForecastResult, holt_winters_forecast, moving_average_forecast

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


@dataclass
class InventoryInsight:
    ingredient: str
    average_daily_usage: float
    projected_depletion_date: Optional[pd.Timestamp]
    days_on_hand: Optional[float]
    recommended_reorder_qty: Optional[float]
    forecast: ForecastResult


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


def estimate_item_mix(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate item-level counts by distributing category totals across recipes.

    When the source data already contains `item_name`, it will be used directly.
    Otherwise, the function falls back to a heuristic mapping defined above.
    """
    if sales.empty:
        return pd.DataFrame(columns=["period", "item_name", "estimated_count"])

    if "item_name" in sales.columns:
        result = (
            sales.groupby(["period", "item_name"])["count"]
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
        item_mix = estimate_item_mix(load_sales_data())
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


def compute_days_on_hand(
    usage: pd.DataFrame,
    shipments: Optional[pd.DataFrame] = None,
    lookback_periods: int = 2,
) -> pd.DataFrame:
    """
    Estimate days-on-hand by comparing recent usage to shipment quantities.
    """
    if usage.empty:
        return pd.DataFrame(
            columns=["ingredient", "average_daily_usage", "days_on_hand_estimate"]
        )
    if shipments is None:
        shipments = load_shipments()

    usage["period"] = pd.to_datetime(usage["period"])
    recent = (
        usage.sort_values("period")
        .groupby("ingredient")
        .tail(lookback_periods)
        .groupby("ingredient")["estimated_usage"]
        .mean()
        .rename("avg_period_usage")
        .reset_index()
    )

    # Assume monthly reporting periods; convert to daily usage.
    recent["average_daily_usage"] = recent["avg_period_usage"] / 30.0

    inventory = shipments.copy()
    inventory["ingredient"] = inventory["ingredient"].str.lower()
    inventory["monthly_inflow"] = inventory["quantity_per_shipment"] * inventory[
        "shipments"
    ]
    merged = recent.merge(inventory[["ingredient", "monthly_inflow"]], how="left")
    merged["monthly_inflow"] = merged["monthly_inflow"].fillna(0)
    merged["days_on_hand_estimate"] = np.where(
        merged["average_daily_usage"] > 0,
        merged["monthly_inflow"] / merged["average_daily_usage"],
        np.nan,
    )
    return merged[["ingredient", "average_daily_usage", "days_on_hand_estimate"]]


def build_inventory_insights(
    usage: pd.DataFrame,
    current_stock: Optional[pd.DataFrame] = None,
    horizon: int = 14,
) -> Iterable[InventoryInsight]:
    """
    Construct ingredient reorder insights using forecasts and stock figures.

    Parameters
    ----------
    usage
        DataFrame produced by `estimate_ingredient_usage`.
    current_stock
        Optional DataFrame with columns `ingredient` and `on_hand`.
        When omitted, shipment-based inflow is used as a loose proxy.
    horizon
        Forecast horizon in days.
    """
    if usage.empty:
        return []

    usage = usage.copy()
    usage["period"] = pd.to_datetime(usage["period"])
    usage = usage.sort_values("period")

    inventory_lookup = None
    if current_stock is not None and not current_stock.empty:
        inventory_lookup = (
            current_stock.assign(ingredient=lambda df: df["ingredient"].str.lower())
            .set_index("ingredient")["on_hand"]
        )

    insights: list[InventoryInsight] = []
    grouped = usage.groupby("ingredient", sort=False)
    for ingredient, group in grouped:
        series = (
            group.set_index("period")["estimated_usage"]
            .sort_index()
            .astype(float)
        )
        series = series.asfreq("MS", method="ffill").fillna(method="bfill")
        forecast = (
            holt_winters_forecast(series, horizon)
            if len(series) >= 4
            else moving_average_forecast(series, horizon)
        )

        current_qty = (
            inventory_lookup.get(ingredient)
            if inventory_lookup is not None and ingredient in inventory_lookup
            else np.nan
        )
        daily_usage = series.tail(1).iloc[0] / 30.0 if len(series) else np.nan
        depletion = (
            forecast.point_forecast.cumsum() / 30.0
            if isinstance(forecast.point_forecast.index, pd.DatetimeIndex)
            else forecast.point_forecast.cumsum()
        )

        reorder_date = None
        recommended_qty = None
        days_on_hand = None

        if not np.isnan(current_qty) and daily_usage > 0:
            days_on_hand = current_qty / daily_usage
            cumulative = forecast.point_forecast.cumsum()
            threshold = cumulative[cumulative >= current_qty]
            reorder_date = threshold.index[0] if not threshold.empty else None
            recommended_qty = daily_usage * horizon

        insight = InventoryInsight(
            ingredient=ingredient,
            average_daily_usage=daily_usage,
            projected_depletion_date=reorder_date,
            days_on_hand=days_on_hand,
            recommended_reorder_qty=recommended_qty,
            forecast=forecast,
        )
        insights.append(insight)
    return insights


def build_alert_table(insights: Iterable[InventoryInsight]) -> pd.DataFrame:
    """Convert InventoryInsight objects into a DataFrame for display/export."""
    rows = []
    for insight in insights:
        rows.append(
            {
                "ingredient": insight.ingredient,
                "avg_daily_usage": insight.average_daily_usage,
                "days_on_hand": insight.days_on_hand,
                "projected_depletion_date": insight.projected_depletion_date,
                "recommended_reorder_qty": insight.recommended_reorder_qty,
                "model": insight.forecast.model_name,
            }
        )
    return pd.DataFrame(rows)

