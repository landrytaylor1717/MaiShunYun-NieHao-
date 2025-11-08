"""
Streamlit dashboard for Mai Shun Yun inventory intelligence.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from collections.abc import Mapping

import altair as alt
import pandas as pd
import streamlit as st

from data_loader import (
    DATA_DIR,
    list_available_periods,
    load_item_sales_data,
    load_recipe_book,
    load_sales_data,
    load_shipments,
)
from features import (
    build_alert_table,
    build_inventory_insights,
    compute_days_on_hand,
    estimate_ingredient_usage,
    estimate_item_mix,
    monthly_revenue_trend,
    summarise_sales,
)
from mongo_utils import fetch_recent_alerts, save_alert


def _secrets_dict() -> Mapping[str, str]:
    try:
        return st.secrets  # type: ignore[return-value]
    except (AttributeError, RuntimeError):
        return {}


def get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a configuration value from Streamlit secrets or environment."""
    secrets = _secrets_dict()
    if key in secrets:
        return secrets[key]
    return os.getenv(key, default)


def bootstrap_env_from_secrets() -> None:
    """Populate os.environ with secrets so helper modules can read them."""
    secrets = _secrets_dict()
    for key in ["MONGO_URI", "MONGO_DB", "MONGO_COLLECTION"]:
        if key in secrets and not os.getenv(key):
            os.environ[key] = secrets[key]


@st.cache_data(show_spinner=False)
def get_sales_data() -> pd.DataFrame:
    return load_sales_data()


@st.cache_data(show_spinner=False)
def get_item_sales_data() -> pd.DataFrame:
    return load_item_sales_data()


@st.cache_data(show_spinner=False)
def get_recipe_book() -> pd.DataFrame:
    return load_recipe_book()


@st.cache_data(show_spinner=False)
def get_shipments() -> pd.DataFrame:
    return load_shipments()


def init_gemini():
    bootstrap_env_from_secrets()
    api_key = get_config_value("GOOGLE_API_KEY")
    if not api_key:
        return None, "Set GOOGLE_API_KEY to enable Gemini insights."
    try:
        import google.generativeai as genai  # type: ignore

        model_name = get_config_value("GEMINI_MODEL", "gemini-1.5-flash")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Gemini initialisation failed: {exc}"


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percentage(value: float) -> str:
    return f"{value * 100:,.1f}%"


def render_kpis(monthly_trend: pd.DataFrame, selected_period: Optional[pd.Timestamp] = None) -> None:
    if monthly_trend.empty:
        st.warning("No sales trend data available yet.")
        return

    trend_sorted = monthly_trend.sort_values("period")
    latest = trend_sorted.iloc[-1]
    prev = trend_sorted.iloc[-2] if len(trend_sorted) > 1 else None

    if selected_period is not None:
        match = trend_sorted[trend_sorted["period"] == selected_period]
        if not match.empty:
            latest = match.iloc[0]
            prev = trend_sorted[trend_sorted["period"] < selected_period].tail(1)
            prev = prev.iloc[0] if not prev.empty else None

    sales_delta = (
        format_percentage(latest["amount_change_pct"])
        if pd.notnull(latest["amount_change_pct"])
        else "n/a"
    )
    count_delta = (
        format_percentage(latest["count_change_pct"])
        if pd.notnull(latest["count_change_pct"])
        else "n/a"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Monthly Revenue",
        format_currency(latest["amount"]),
        sales_delta if prev is not None else None,
    )
    col2.metric("Items Sold", f"{latest['count']:,.0f}", count_delta if prev is not None else None)
    col3.metric("Reporting Period", latest["period"].strftime("%b %Y"))


def render_category_chart(summary: pd.DataFrame, selected_period: Optional[pd.Timestamp]) -> None:
    if summary.empty:
        st.info("Category breakdown will appear once data is available.")
        return

    if selected_period is None:
        top_categories = (
            summary.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .index
        )
        data = summary[summary["category"].isin(top_categories)]
        chart = (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(
                x="period:T",
                y="amount:Q",
                color="category:N",
                tooltip=["period:T", "category:N", "count:Q", "amount:Q"],
            )
            .interactive()
        )
    else:
        data = summary.assign(category=lambda df: df["category"])
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x="amount:Q",
                y=alt.Y("category:N", sort="-x"),
                color="category:N",
                tooltip=["category:N", "count:Q", "amount:Q"],
            )
            .interactive()
        )
    st.altair_chart(chart, use_container_width=True)


def render_item_chart(
    item_sales: pd.DataFrame, selected_period: Optional[pd.Timestamp], limit: int = 15
) -> None:
    if item_sales.empty:
        st.info("Item-level sales will display once cleaned data is available.")
        return
    if selected_period is None:
        data = (
            item_sales.groupby("item_name", as_index=False)[["amount", "count"]]
            .sum()
            .sort_values("amount", ascending=False)
            .head(limit)
        )
    else:
        data = (
            item_sales.sort_values("amount", ascending=False)
            .head(limit)
            .copy()
        )

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x="amount:Q",
            y=alt.Y("item_name:N", sort="-x", title="Item"),
            tooltip=["item_name:N", "count:Q", "amount:Q"],
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    table = data.assign(
        amount=lambda df: df["amount"].map(format_currency),
        count=lambda df: df["count"].round().astype(int),
    )
    st.dataframe(
        table[["item_name", "count", "amount"]]
        .rename(columns={"item_name": "Item", "count": "Units", "amount": "Revenue"}),
        hide_index=True,
        use_container_width=True,
    )


def render_ingredient_chart(
    usage: pd.DataFrame, selected_period: Optional[pd.Timestamp], limit: int = 15
) -> None:
    if usage.empty:
        st.info("Ingredient usage metrics will appear once data is available.")
        return

    display_usage = usage.assign(
        ingredient=lambda df: df["ingredient"].str.replace("_", " ").str.title()
    )

    if selected_period is None:
        data = (
            display_usage.groupby("ingredient", as_index=False)["estimated_usage"]
            .sum()
            .sort_values("estimated_usage", ascending=False)
            .head(limit)
        )
    else:
        data = (
            display_usage.sort_values("estimated_usage", ascending=False)
            .head(limit)
            .copy()
        )

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x="estimated_usage:Q",
            y=alt.Y("ingredient:N", sort="-x", title="Ingredient"),
            tooltip=["ingredient:N", "estimated_usage:Q"],
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(
        data.assign(estimated_usage=lambda df: df["estimated_usage"].round(2)).rename(
            columns={"estimated_usage": "Usage"}
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_usage_table(alert_table: pd.DataFrame) -> None:
    if alert_table.empty:
        st.info("Ingredient usage estimates will populate once recipe mapping is ready.")
        return
    table = alert_table.copy()
    table["ingredient"] = table["ingredient"].str.title()
    st.dataframe(
        table.sort_values("projected_depletion_date", na_position="last"),
        use_container_width=True,
    )


def handle_alert_submission(selected_row: Dict, notes: str) -> None:
    payload = {
        "ingredient": selected_row.get("ingredient"),
        "projected_depletion_date": selected_row.get("projected_depletion_date"),
        "recommended_reorder_qty": selected_row.get("recommended_reorder_qty"),
        "notes": notes,
    }
    success = save_alert(payload)
    if success:
        st.success("Alert saved to MongoDB.")
    else:
        st.warning("MongoDB not configured; set MONGO_URI to enable persistence.")


def render_gemini_assistant(context: str) -> None:
    model, error = init_gemini()
    st.subheader("Gemini Assistant")
    if error:
        st.info(error)
        return
    if "gemini_history" not in st.session_state:
        st.session_state.gemini_history = []

    for message in st.session_state.gemini_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about reorder priorities or demand trends…")
    if prompt and model:
        st.session_state.gemini_history.append({"role": "user", "content": prompt})
        try:
            response = model.generate_content(
                f"""You are assisting a restaurant inventory manager.
Context summary:
{context}

Question: {prompt}

Provide clear, actionable guidance and reference specific ingredients when possible."""
            )
            answer = response.text or "I could not generate an answer."
        except Exception as exc:  # noqa: BLE001
            answer = f"Gemini request failed: {exc}"
        st.session_state.gemini_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


def render_recent_alerts() -> None:
    bootstrap_env_from_secrets()
    st.subheader("Recent Saved Alerts (MongoDB)")
    alerts = fetch_recent_alerts()
    if not alerts:
        st.caption("Connect MongoDB via MONGO_URI to persist alerts.")
        return
    alert_df = pd.DataFrame(alerts)
    alert_df["created_at"] = pd.to_datetime(alert_df["created_at"])
    st.dataframe(alert_df[["created_at", "ingredient", "notes"]], hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Mai Shun Yun Inventory Intelligence",
        page_icon="🍜",
        layout="wide",
    )
    st.title("Mai Shun Yun | Inventory Intelligence Dashboard")
    st.caption("Track ingredient health, forecast demand, and surface actionable insights.")

    if not DATA_DIR.exists():
        st.error(f"Data directory not found: {DATA_DIR}")
        return

    sales = get_sales_data()
    item_sales = get_item_sales_data()
    recipe_book = get_recipe_book()
    shipments = get_shipments()

    periods = list_available_periods()
    period_map = {p.strftime("%b %Y"): p for p in periods}
    period_options = ["All Periods"] + list(period_map.keys())
    default_index = len(period_options) - 1 if periods else 0

    st.subheader("Sales Mix & Revenue Trends")
    select_col, _ = st.columns([1.5, 3])
    with select_col:
        selected_label = st.selectbox(
            "Reporting Period",
            period_options,
            index=default_index,
            key="period_select",
        )

    selected_period = period_map.get(selected_label)
    sales_view = sales if selected_period is None else sales[sales["period"] == selected_period]
    item_sales_view = (
        item_sales if selected_period is None else item_sales[item_sales["period"] == selected_period]
    )

    monthly_trend = monthly_revenue_trend(sales)
    render_kpis(monthly_trend, selected_period)

    summary = summarise_sales(sales_view if not sales_view.empty else sales)
    render_category_chart(summary, selected_period)

    st.markdown("#### Individual Meals")
    render_item_chart(item_sales_view, selected_period)

    item_mix_source = estimate_item_mix(item_sales)
    usage = estimate_ingredient_usage(item_mix_source, recipe_book)
    usage_view = usage if selected_period is None else usage[usage["period"] == selected_period]

    st.markdown("#### Ingredient Usage")
    render_ingredient_chart(usage_view, selected_period)

    st.divider()

    st.subheader("Ingredient Usage & Reorder Signals")
    days_on_hand = compute_days_on_hand(usage, shipments)
    insights = list(build_inventory_insights(usage, None))
    alert_table = build_alert_table(insights)

    if not days_on_hand.empty:
        st.dataframe(days_on_hand, use_container_width=True)

    render_usage_table(alert_table)

    if not alert_table.empty:
        st.markdown("#### Save a Reorder Alert")
        selected = st.selectbox(
            "Choose ingredient",
            alert_table["ingredient"],
            index=0,
            key="alert_select",
        )
        notes = st.text_area("Notes / instructions", placeholder="e.g., expedite delivery")
        if st.button("Save Alert"):
            selected_row = alert_table[alert_table["ingredient"] == selected].iloc[0]
            handle_alert_submission(selected_row.to_dict(), notes)
        render_recent_alerts()

    st.divider()

    if not monthly_trend.empty:
        if selected_period is not None:
            revenue_row = monthly_trend[monthly_trend["period"] == selected_period]
            if revenue_row.empty:
                revenue_row = monthly_trend.iloc[[-1]]
        else:
            revenue_row = monthly_trend.iloc[[-1]]
        revenue_value = revenue_row.iloc[0]["amount"]
        revenue_label = revenue_row.iloc[0]["period"].strftime("%b %Y")
        revenue_context = f"Revenue for {revenue_label}: {format_currency(revenue_value)}"
    else:
        revenue_context = "Revenue data unavailable."

    if not alert_table.empty:
        top_risk = alert_table.sort_values("days_on_hand").head(1)["ingredient"].iloc[0]
        risk_context = f"Top ingredient at risk: {top_risk}"
    else:
        risk_context = "No ingredient risk computed."

    context_bits = [revenue_context, risk_context]
    render_gemini_assistant("\n".join(context_bits))


if __name__ == "__main__":
    main()
