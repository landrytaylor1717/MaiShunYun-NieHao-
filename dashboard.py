"""
Streamlit dashboard for Mai Shun Yun inventory intelligence.
"""

from __future__ import annotations

import os
from typing import Optional

from collections.abc import Mapping
import html

import altair as alt
import pandas as pd
import streamlit as st

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
    summarise_sales,
)


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
]

DESCRIPTOR_SET = {name.lower() for name in DESCRIPTOR_NAMES}


def filter_individual_items(item_sales: pd.DataFrame) -> pd.DataFrame:
    if item_sales.empty:
        return item_sales
    mask = ~item_sales["item_name"].astype(str).str.strip().str.lower().isin(DESCRIPTOR_SET)
    return item_sales[mask].copy()


def filter_descriptor_categories(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    mask = summary["category"].astype(str).str.strip().str.lower().isin(DESCRIPTOR_SET)
    return summary[mask].copy()


def filter_drink_items(item_sales: pd.DataFrame) -> pd.DataFrame:
    if item_sales.empty:
        return item_sales
    pattern = "|".join(DRINK_KEYWORDS)
    mask = item_sales["item_name"].astype(str).str.contains(pattern, case=False, na=False)
    exclude_mask = item_sales["item_name"].astype(str).str.contains("bun|dumpling|soup", case=False, na=False)
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


def render_item_chart(
    item_sales: pd.DataFrame,
    selected_period: Optional[pd.Timestamp],
    *,
    title: str = "Top Menu Items by Revenue",
    limit: int = 15,
) -> None:
    if item_sales.empty:
        st.info(f"{title} will display once data is available.")
        return
    st.markdown(f"### {title}")
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
            x="amount:Q",
            y=alt.Y("item_name:N", sort="-x", title="Item"),
            tooltip=["item_name:N", "count:Q", "amount:Q"],
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def render_descriptor_chart(
    descriptor_sales: pd.DataFrame, selected_period: Optional[pd.Timestamp], limit: int = 12
) -> None:
    st.markdown("### Group Descriptor Performance")
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
            x="amount:Q",
            y=alt.Y("category:N", sort="-x", title="Descriptor"),
            tooltip=["category:N", "count:Q", "amount:Q"],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def render_drink_chart(
    drink_sales: pd.DataFrame, selected_period: Optional[pd.Timestamp], limit: int = 15
) -> None:
    st.markdown("### Drinks by Revenue")
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
    y_field = alt.Y("item_name:N", sort="-x", title="Drink")

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x="amount:Q",
            y=y_field,
            color=alt.Color("item_name:N", legend=None),
            tooltip=["item_name:N", "count:Q", "amount:Q"],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def render_appetizer_chart(
    appetizer_sales: pd.DataFrame, selected_period: Optional[pd.Timestamp], limit: int = 15
) -> None:
    st.markdown("### Appetizers by Revenue")
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
            x="amount:Q",
            y=alt.Y("item_name:N", sort="-x", title="Appetizer"),
            color=alt.Color("item_name:N", legend=None),
            tooltip=["item_name:N", "count:Q", "amount:Q"],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def render_ingredient_chart(
    usage: pd.DataFrame,
    selected_period: Optional[pd.Timestamp],
    *,
    title: str = "Top Ingredients by Usage",
    limit: int = 15,
) -> None:
    st.markdown(f"### {title}")
    if usage.empty:
        st.info("Ingredient usage metrics will appear once data is available.")
        return

    usage = usage[
        ~usage["ingredient"].str.contains(r"\b(ramen|egg)\b", case=False, na=False)
        & ~usage["ingredient"].str.contains(r"\(count\)", case=False, na=False)
    ]
    if usage.empty:
        st.info("Ingredient usage metrics will appear once data is available.")
        return

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


def render_ramen_egg_chart(
    usage: pd.DataFrame,
    selected_period: Optional[pd.Timestamp],
) -> None:
    keys = ["ramen", "egg"]
    subset = usage[
        usage["ingredient"].str.contains("|".join(keys), case=False, na=False)
    ]
    st.markdown("### Ramen & Egg Usage")
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

    if selected_period is None:
        chart = (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(
                x="period:T",
                y="estimated_usage:Q",
                color="ingredient:N",
                tooltip=["period:T", "ingredient:N", "estimated_usage:Q"],
            )
            .interactive()
        )
    else:
        data = data[data["period"] == selected_period]
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x="estimated_usage:Q",
                y=alt.Y("ingredient:N", sort="-x"),
                tooltip=["ingredient:N", "estimated_usage:Q"],
            )
        )
    st.altair_chart(chart, use_container_width=True)


def render_descriptor_chart(
    descriptor_sales: pd.DataFrame, selected_period: Optional[pd.Timestamp], limit: int = 12
) -> None:
    st.markdown("### Group Descriptor Performance")
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
            x="amount:Q",
            y=alt.Y("category:N", sort="-x", title="Descriptor"),
            tooltip=["category:N", "count:Q", "amount:Q"],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def render_ingredient_chart(
    usage: pd.DataFrame,
    selected_period: Optional[pd.Timestamp],
    *,
    title: str = "Top Ingredients by Usage",
    limit: int = 15,
) -> None:
    st.markdown(f"### {title}")
    if usage.empty:
        st.info("Ingredient usage metrics will appear once data is available.")
        return

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
def render_usage_table(alert_table: pd.DataFrame) -> None:
    if alert_table.empty:
        st.info("Ingredient usage estimates will populate once data is available.")
        return
    table = alert_table.copy()
    table["ingredient"] = table["ingredient"].str.title()
    st.dataframe(
        table.sort_values("projected_depletion_date", na_position="last"),
        use_container_width=True,
    )

def render_gemini_assistant(context: str) -> None:
    if "gemini_open" not in st.session_state:
        st.session_state["gemini_open"] = False
    if "gemini_history" not in st.session_state:
        st.session_state["gemini_history"] = []

    model, error = init_gemini()
    st.markdown(
        """
        <style>
        .gemini-fab-container {
            position: fixed;
            bottom: 24px;
            right: 24px;
            z-index: 1000;
        }
        .gemini-fab-container button {
            width: 56px;
            height: 56px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            background: linear-gradient(135deg, #6750a4, #9c6ce0);
            color: white;
            font-size: 28px;
            box-shadow: 0 6px 24px rgba(103, 80, 164, 0.35);
        }
        .gemini-popup {
            position: fixed;
            bottom: 92px;
            right: 24px;
            width: 320px;
            max-height: 60vh;
            background: rgba(255, 255, 255, 0.97);
            border-radius: 16px;
            box-shadow: 0 24px 48px rgba(15, 23, 42, 0.25);
            padding: 16px;
            z-index: 1100;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .gemini-popup h4 {
            margin: 0;
            font-size: 1rem;
            font-weight: 600;
        }
        .gemini-messages {
            overflow-y: auto;
            max-height: 32vh;
            padding-right: 4px;
        }
        .gemini-msg {
            border-radius: 10px;
            padding: 8px 10px;
            margin-bottom: 6px;
            font-size: 0.9rem;
            line-height: 1.3;
        }
        .gemini-msg.user {
            background: #e8def8;
            align-self: flex-end;
        }
        .gemini-msg.assistant {
            background: #f5f5f5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="gemini-fab-container">', unsafe_allow_html=True)
        if st.button("🤖", key="gemini_toggle"):
            st.session_state["gemini_open"] = not st.session_state["gemini_open"]
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if not st.session_state["gemini_open"]:
        return

    if error:
        st.session_state["gemini_open"] = False
        st.warning(error)
        return

    messages = st.session_state["gemini_history"]
    messages_html = "".join(
        f"<div class='gemini-msg {'user' if msg['role']=='user' else 'assistant'}'>{html.escape(msg['content'])}</div>"
        for msg in messages
    ) or "<div class='gemini-msg assistant'>Ask a question to get started.</div>"

    with st.container():
        st.markdown(
            f"<div class='gemini-popup'><h4>Gemini Assistant</h4><div class='gemini-messages'>{messages_html}</div>",
            unsafe_allow_html=True,
        )
        with st.form("gemini_form", clear_on_submit=True):
            prompt = st.text_input(
                "Ask Gemini",
                key="gemini_prompt",
                label_visibility="collapsed",
                placeholder="Ask about inventory insights…",
            )
            col_send, col_close = st.columns([3, 1])
            send = col_send.form_submit_button("Send")
            close = col_close.form_submit_button("Close")
        st.markdown("</div>", unsafe_allow_html=True)

    if close:
        st.session_state["gemini_open"] = False
        st.experimental_rerun()

    if send:
        if not prompt:
            st.experimental_rerun()
        st.session_state["gemini_history"].append({"role": "user", "content": prompt})
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
        st.session_state["gemini_history"].append({"role": "assistant", "content": answer})
        st.experimental_rerun()
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
    appetizer_sales = filter_appetizer_items(filtered_item_sales)
    monthly_trend = compute_item_trend(filtered_item_sales)
    render_kpis(monthly_trend, selected_period)

    summary = summarise_sales(sales_view if not sales_view.empty else sales)
    descriptor_sales = filter_descriptor_categories(summary)
    render_descriptor_chart(descriptor_sales, selected_period)

    render_item_chart(
        filtered_item_sales_view, selected_period, title="Top Individual Meals by Revenue"
    )
    render_drink_chart(drink_sales, selected_period)
    render_appetizer_chart(appetizer_sales, selected_period)

    item_mix_source = estimate_item_mix(item_sales)
    usage = estimate_ingredient_usage(item_mix_source, recipe_book)
    render_ingredient_chart(usage, selected_period, title="Top Ingredients by Usage")
    render_ramen_egg_chart(usage, selected_period)

    st.divider()

    st.subheader("Forecasts & Reorder Signals")
    days_on_hand = compute_days_on_hand(usage, shipments)
    insights = list(build_inventory_insights(usage, None))
    alert_table = build_alert_table(insights)

    if not days_on_hand.empty:
        st.markdown("### Estimated Days on Hand")
        st.dataframe(days_on_hand, use_container_width=True)

    st.markdown("### Forecasted Reorder Signals")
    render_usage_table(alert_table)

    st.divider()

    context = ""
    if not monthly_trend.empty:
        ref_period = (
            monthly_trend[monthly_trend["period"] == selected_period]
            if selected_period is not None
            else monthly_trend.iloc[[-1]]
        )
        if ref_period.empty:
            ref_period = monthly_trend.iloc[[-1]]
        revenue_value = ref_period.iloc[0]["amount"]
        revenue_label = ref_period.iloc[0]["period"].strftime("%b %Y")
        context = f"Revenue for {revenue_label}: {format_currency(revenue_value)}."
    if not alert_table.empty:
        top_risk = (
            alert_table.sort_values("days_on_hand").head(1)["ingredient"].iloc[0]
        )
        context += f" Ingredient at risk: {top_risk}."

    render_gemini_assistant(context.strip())


if __name__ == "__main__":
    main()
