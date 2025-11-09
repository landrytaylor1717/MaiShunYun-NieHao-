"""
Streamlit dashboard for Mai Shun Yun inventory intelligence.
"""

from __future__ import annotations

from typing import Optional


import altair as alt
import pandas as pd
import streamlit as st
from contextlib import contextmanager


from data_loader import (
    DATA_DIR,
    list_available_periods,
    load_item_sales_data,
    load_recipe_book,
    load_sales_data,
)
from features import (
    estimate_ingredient_usage,
    estimate_item_mix,
    summarise_sales,
)
from chat_utils import (
    filter_individual_items,
    filter_descriptor_categories,
    filter_drink_items,
    filter_appetizer_items,
    compute_item_trend,
)


def style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(
            labelColor="#ffffff",
            titleColor="#ffffff",
            gridColor="rgba(148, 163, 184, 0.2)",
            domainColor="rgba(255, 255, 255, 0.35)",
            tickColor="rgba(255, 255, 255, 0.5)",
        )
        .configure_legend(
            labelColor="#ffffff",
            titleColor="#ffffff",
            symbolFillColor="rgba(255,255,255,0.85)",
        )
        .configure_view(stroke="rgba(255, 255, 255, 0.12)")
    )


GLOBAL_STYLE = """
<style>
:root {
    --msy-indigo: #4c51bf;
    --msy-amber: #f97316;
    --msy-slate: #f8fafc;
}
.stApp {
    background-color: #0f172a;
    color: #ffffff;
}
.stApp [class*="stMarkdown"] * {
    color: #ffffff !important;
}
[data-testid="stMarkdownContainer"] * {
    color: #ffffff !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #ffffff !important;
}
.hero-card {
    background: linear-gradient(135deg, rgba(79,70,229,0.96), rgba(249,115,22,0.92));
    padding: 32px 38px;
    border-radius: 28px;
    color: #ffffff;
    margin-bottom: 28px;
    box-shadow: 0 22px 48px rgba(79, 70, 229, 0.35);
}
.hero-card h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 6px;
    color: #ffffff;
}
.hero-card p {
    font-size: 1.05rem;
    opacity: 0.95;
    margin: 0;
    color: #ffffff;
}
.section-card {
    background: rgba(15, 23, 42, 0.55);
    border-radius: 22px;
    padding: 26px 30px;
    margin-bottom: 26px;
    box-shadow: 0 18px 40px rgba(8, 145, 178, 0.18);
    border: 1px solid rgba(148, 163, 184, 0.28);
    color: #ffffff;
}
.section-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 18px;
    color: #ffffff;
}
.section-header h3 {
    font-size: 1.25rem;
    margin: 0;
    font-weight: 700;
    color: #ffffff;
}
.section-subtitle {
    font-size: 0.96rem;
    margin: 0;
    color: rgba(255,255,255,0.72);
}
.chart-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #ffffff;
    margin: 4px 0 14px;
}
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.65);
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.32);
    box-shadow: 0 16px 36px rgba(15,23,42,0.28);
}
.stDataFrame [data-testid="stDataFrameContainer"] {
    color: #ffffff;
}
.stDataFrame table {
    color: #ffffff;
}
.stDataFrame table thead th {
    color: #ffffff;
}
.stDataFrame table tbody td {
    color: #ffffff;
}
.stSelectbox label {
    color: #ffffff;
}
.stSelectbox [data-baseweb="select"] * {
    color: #0f172a;
}
.stSelectbox [data-baseweb="select"] {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.45);
}
.stSelectbox [data-baseweb="select"] [class*="control"] {
    background-color: transparent !important;
    border-radius: 12px;
    border: none;
}
.stSelectbox [data-baseweb="select"] [class*="value"] {
    color: #0f172a !important;
}
.stSelectbox [data-baseweb="select"] [class*="placeholder"] {
    color: rgba(15, 23, 42, 0.55) !important;
}
.stSelectbox [data-baseweb="select"] [class*="value-label"] {
    color: #0f172a !important;
}
.stSelectbox [data-baseweb="select"] [class*="singleValue"] {
    color: #0f172a !important;
}
.stSelectbox [data-baseweb="select"] [class*="selected"] {
    color: #0f172a;
}
.stSelectbox div[role="listbox"] span {
    color: #0f172a;
}
</style>
"""


@contextmanager
def section_block(title: str, subtitle: Optional[str] = None) -> None:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    header_html = f"<div class='section-header'><h3>{title}</h3>"
    if subtitle:
        header_html += f"<p class='section-subtitle'>{subtitle}</p>"
    header_html += "</div>"
    st.markdown(header_html, unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_sales_data() -> pd.DataFrame:
    return load_sales_data()


@st.cache_data(show_spinner=False)
def get_item_sales_data() -> pd.DataFrame:
    return load_item_sales_data()


@st.cache_data(show_spinner=False)
def get_recipe_book() -> pd.DataFrame:
    return load_recipe_book()


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percentage(value: float) -> str:
    return f"{value * 100:,.1f}%"




def render_kpis(
    monthly_trend: pd.DataFrame,
    selected_period: Optional[pd.Timestamp] = None,
    selected_label: Optional[str] = None,
) -> None:
    if monthly_trend.empty:
        st.warning("No sales trend data available yet.")
        return

    all_periods_selected = selected_period is None and (
        selected_label is None or selected_label.lower() == "all periods"
    )

    if all_periods_selected:
        total_revenue = monthly_trend["amount"].sum()
        total_items = monthly_trend["count"].sum()
        period_label = (
            f"{selected_label} ({len(monthly_trend)} periods)"
            if selected_label
            else f"All Periods ({len(monthly_trend)} periods)"
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", format_currency(total_revenue))
        col2.metric("Items Sold", f"{total_items:,.0f}")
        col3.metric("Reporting Period", period_label)
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
    col3.metric(
        "Reporting Period",
        selected_label if selected_label else latest["period"].strftime("%b %Y"),
    )


def render_item_chart(
    item_sales: pd.DataFrame,
    selected_period: Optional[pd.Timestamp],
    *,
    title: str = "Top Menu Items by Revenue",
    limit: int = 15,
) -> None:
    st.markdown(f"<h4 class='chart-title'>{title}</h4>", unsafe_allow_html=True)
    if item_sales.empty:
        st.info(f"{title} will display once data is available.")
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
    data["item_name"] = data["item_name"].astype(str).str.title()

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("amount:Q", title="Amount ($)"),
            y=alt.Y("item_name:N", sort="-x", title="Item"),
            color=alt.Color("item_name:N", legend=None),
            tooltip=[
                alt.Tooltip("item_name:N", title="Item"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("amount:Q", title="Amount ($)"),
            ],
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(style_chart(chart), width="stretch")


def render_descriptor_chart(
    descriptor_sales: pd.DataFrame, selected_period: Optional[pd.Timestamp], limit: int = 12
) -> None:
    st.markdown(
        "<h4 class='chart-title'>Group Descriptor Performance</h4>",
        unsafe_allow_html=True,
    )
    if descriptor_sales.empty:
        st.info("Descriptor metrics will appear once data is available.")
        return

    if selected_period is None:
        data = (
            descriptor_sales.groupby("category", as_index=False)[["amount", "count"]]
            .sum()
            .sort_values("amount", ascending=False)
            .head(limit)
        )
    else:
        data = (
            descriptor_sales[descriptor_sales["period"] == selected_period]
            .sort_values("amount", ascending=False)
            .head(limit)
            .copy()
        )

    data["category"] = data["category"].astype(str).str.title()
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("amount:Q", title="Amount ($)"),
            y=alt.Y(
                "category:N",
                sort="-x",
                title="Descriptor",
                axis=alt.Axis(labelLimit=0, labelOverlap=False),
            ),
            color=alt.Color("category:N", legend=None),
            tooltip=[
                alt.Tooltip("category:N", title="Descriptor"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("amount:Q", title="Amount ($)"),
            ],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(style_chart(chart), width="stretch")


def render_drink_chart(
    drink_sales: pd.DataFrame, selected_period: Optional[pd.Timestamp], limit: int = 15
) -> None:
    st.markdown("<h4 class='chart-title'>Drinks by Revenue</h4>", unsafe_allow_html=True)
    if drink_sales.empty:
        st.info("Drink metrics will appear once data is available.")
        return

    if selected_period is None:
        data = (
            drink_sales.groupby("item_name", as_index=False)[["amount", "count"]]
            .sum()
            .sort_values("amount", ascending=False)
            .head(limit)
        )
    else:
        data = (
            drink_sales[drink_sales["period"] == selected_period]
            .groupby("item_name", as_index=False)[["amount", "count"]]
            .sum()
            .sort_values("amount", ascending=False)
            .head(limit)
            .copy()
        )

    data["item_name"] = data["item_name"].astype(str).str.title()
    y_field = alt.Y(
        "item_name:N",
        sort="-x",
        title="Drink",
        axis=alt.Axis(labelLimit=0, labelOverlap=False),
    )

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("amount:Q", title="Amount ($)"),
            y=y_field,
            color=alt.Color("item_name:N", legend=None),
            tooltip=[
                alt.Tooltip("item_name:N", title="Drink"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("amount:Q", title="Amount ($)"),
            ],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(style_chart(chart), width="stretch")


def render_appetizer_chart(
    appetizer_sales: pd.DataFrame, selected_period: Optional[pd.Timestamp], limit: int = 15
) -> None:
    st.markdown("<h4 class='chart-title'>Appetizers by Revenue</h4>", unsafe_allow_html=True)
    if appetizer_sales.empty:
        st.info("Appetizer metrics will appear once data is available.")
        return

    if selected_period is None:
        data = (
            appetizer_sales.groupby("item_name", as_index=False)[["amount", "count"]]
            .sum()
            .sort_values("amount", ascending=False)
            .head(limit)
        )
    else:
        data = (
            appetizer_sales[appetizer_sales["period"] == selected_period]
            .groupby("item_name", as_index=False)[["amount", "count"]]
            .sum()
            .sort_values("amount", ascending=False)
            .head(limit)
            .copy()
        )

    data["item_name"] = data["item_name"].astype(str).str.title()
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("amount:Q", title="Amount ($)"),
            y=alt.Y(
                "item_name:N",
                sort="-x",
                title="Appetizer",
                axis=alt.Axis(labelLimit=0, labelOverlap=False),
            ),
            color=alt.Color("item_name:N", legend=None),
            tooltip=[
                alt.Tooltip("item_name:N", title="Appetizer"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("amount:Q", title="Amount ($)"),
            ],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(style_chart(chart), width="stretch")


def render_ingredient_chart(
    usage: pd.DataFrame,
    selected_period: Optional[pd.Timestamp],
    *,
    title: str = "Top Ingredients by Usage",
    limit: int = 15,
) -> None:
    st.markdown(f"<h4 class='chart-title'>{title}</h4>", unsafe_allow_html=True)
    if usage.empty:
        st.info("Ingredient usage metrics will appear once data is available.")
        return

    usage = usage[
        ~usage["ingredient"].str.contains("ramen", case=False, na=False)
        & ~usage["ingredient"].str.contains("egg", case=False, na=False)
    ]
    usage = usage.assign(
        ingredient=lambda df: df["ingredient"].str.replace("_", " ").str.title()
    )
    if selected_period is None:
        data = (
            usage.groupby("ingredient", as_index=False)["estimated_usage"]
            .sum()
            .sort_values("estimated_usage", ascending=False)
            .head(limit)
        )
    else:
        data = (
            usage[usage["period"] == selected_period]
            .sort_values("estimated_usage", ascending=False)
            .head(limit)
            .copy()
        )

    data = data.rename(columns={"estimated_usage": "Amount"})
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("Amount:Q", title="Amount"),
            y=alt.Y("ingredient:N", sort="-x", title="Ingredient"),
            color=alt.Color("ingredient:N", legend=None),
            tooltip=[
                alt.Tooltip("ingredient:N", title="Ingredient"),
                alt.Tooltip("Amount:Q", title="Amount"),
            ],
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(style_chart(chart), width="stretch")


def render_ramen_egg_chart(
    usage: pd.DataFrame,
    selected_period: Optional[pd.Timestamp],
) -> None:
    subset = usage[
        usage["ingredient"].str.contains("ramen", case=False, na=False)
        | usage["ingredient"].str.contains("egg", case=False, na=False)
    ]
    st.markdown("<h4 class='chart-title'>Ramen & Egg Usage</h4>", unsafe_allow_html=True)
    if subset.empty:
        st.info("Ramen and egg usage will appear once data is available.")
        return

    subset = subset.assign(
        ingredient=lambda df: df["ingredient"].str.replace("_", " ").str.title()
    )
    data = (
        subset.groupby(["period", "ingredient"], as_index=False)["estimated_usage"]
        .sum()
        .sort_values("period")
    )
    data = data.rename(columns={"estimated_usage": "Amount"})

    if selected_period is None:
        chart = (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(
                x="period:T",
                y=alt.Y("Amount:Q", title="Amount"),
                color=alt.Color("ingredient:N", legend=None),
                tooltip=[
                    alt.Tooltip("period:T", title="Period"),
                    alt.Tooltip("ingredient:N", title="Ingredient"),
                    alt.Tooltip("Amount:Q", title="Amount"),
                ],
            )
            .interactive()
        )
    else:
        data = data[data["period"] == selected_period]
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("Amount:Q", title="Amount"),
                y=alt.Y("ingredient:N", sort="-x"),
                color=alt.Color("ingredient:N", legend=None),
                tooltip=[
                    alt.Tooltip("ingredient:N", title="Ingredient"),
                    alt.Tooltip("Amount:Q", title="Amount"),
                ],
            )
        )
    st.altair_chart(style_chart(chart), width="stretch")

def main() -> None:
    st.set_page_config(
        page_title="Mai Shun Yun Inventory Intelligence",
        page_icon="🍜",
        layout="wide",
    )
    st.markdown(GLOBAL_STYLE, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card">
            <h1>Mai Shun Yun · Operations Pulse</h1>
            <p>Monitor sales momentum, spotlight top-performing dishes, and keep ingredient usage on your radar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_DIR.exists():
        st.error(f"Data directory not found: {DATA_DIR}")
        return

    sales = get_sales_data()
    item_sales = get_item_sales_data()
    recipe_book = get_recipe_book()
    periods = list_available_periods()
    period_map = {p.strftime("%b %Y"): p for p in periods}
    period_options = ["All Periods"] + list(period_map.keys())
    default_index = len(period_options) - 1 if periods else 0

    st.subheader("Sales Performance Overview")
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

    filtered_item_sales = filter_individual_items(item_sales)
    filtered_item_sales_view = filter_individual_items(item_sales_view)
    drink_sales = filter_drink_items(filtered_item_sales)
    drink_sales_view = (
        filter_drink_items(filtered_item_sales_view)
        if selected_period is not None
        else drink_sales
    )
    appetizer_sales = filter_appetizer_items(filtered_item_sales)
    appetizer_sales_view = (
        filter_appetizer_items(filtered_item_sales_view)
        if selected_period is not None
        else appetizer_sales
    )
    monthly_trend = compute_item_trend(filtered_item_sales)

    with section_block(
        "Sales Snapshot",
        "Select a reporting period to drill in or choose All Periods for the aggregate story.",
    ):
        render_kpis(monthly_trend, selected_period, selected_label)

    summary = summarise_sales(sales_view if not sales_view.empty else sales)
    descriptor_sales = filter_descriptor_categories(summary)

    with section_block(
        "Menu Performance Highlights",
        "Understand how your menu mix contributes to revenue across categories and hero dishes.",
    ):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            render_descriptor_chart(descriptor_sales, selected_period)
        with col2:
            render_item_chart(
                filtered_item_sales_view,
                selected_period,
                title="Top Individual Meals by Revenue",
            )

    with section_block(
        "Beverage & Appetizer Dynamics",
        "Keep tabs on supporting items that boost average checks and guest satisfaction.",
    ):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            render_drink_chart(drink_sales, selected_period)
        with col2:
            render_appetizer_chart(appetizer_sales, selected_period)

    item_mix_source = estimate_item_mix(item_sales)
    usage = estimate_ingredient_usage(item_mix_source, recipe_book)

    with section_block(
        "Ingredient Usage Spotlight",
        "Translate demand into prep needs so your pantry stays balanced.",
    ):
        col1, col2 = st.columns((2, 1), gap="large")
        with col1:
            render_ingredient_chart(
                usage.copy(), selected_period, title="Most Utilized Ingredients"
            )
        with col2:
            render_ramen_egg_chart(usage, selected_period)


if __name__ == "__main__":
    main()
