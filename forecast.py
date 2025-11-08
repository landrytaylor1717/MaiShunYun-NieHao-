"""Time-series forecasting utilities for inventory demand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class ForecastResult:
    """Container for forecast outputs and metadata."""

    point_forecast: pd.Series
    lower: pd.Series
    upper: pd.Series
    model_name: str


def moving_average_forecast(
    y: pd.Series,
    horizon: int,
    window: int = 7,
    alpha: float = 1.96,
) -> ForecastResult:
    """
    Forecast future demand using a trailing moving average.

    Parameters
    ----------
    y
        Historical series with a DatetimeIndex.
    horizon
        Number of periods to forecast.
    window
        Size of the trailing window used to compute the average.
    alpha
        Z-multiplier for approximate confidence bounds (default 95%).
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if window <= 0:
        raise ValueError("window must be positive")

    history = y.dropna().astype(float)
    if history.empty:
        idx = pd.RangeIndex(start=0, stop=horizon, step=1)
        zeros = pd.Series(np.zeros(horizon), index=idx)
        return ForecastResult(zeros, zeros, zeros, model_name="moving_average")

    window = min(window, len(history))
    tail = history.tail(window)
    mean = tail.mean()
    std = tail.std(ddof=1) if len(tail) > 1 else 0.0

    future_index = _build_future_index(history.index, horizon)
    forecast = pd.Series(np.full(horizon, mean), index=future_index)
    lower = forecast - alpha * std
    upper = forecast + alpha * std
    return ForecastResult(
        point_forecast=forecast,
        lower=lower.clip(lower=0),
        upper=upper,
        model_name="moving_average",
    )


def holt_winters_forecast(
    y: pd.Series,
    horizon: int,
    seasonal_periods: Optional[int] = 7,
    trend: Optional[str] = "add",
    seasonal: Optional[str] = "add",
) -> ForecastResult:
    """
    Forecast future demand using Holt-Winters exponential smoothing.

    Falls back to a moving-average style forecast if the model cannot converge.
    """
    history = y.dropna().astype(float)
    if history.empty:
        return moving_average_forecast(y, horizon)

    history = history.asfreq(history.index.inferred_freq or "D")
    try:
        model = ExponentialSmoothing(
            history,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        pred = fitted.forecast(horizon)
        sigma = np.sqrt(fitted.sse / max(len(history) - fitted.params.shape[0], 1))
        future_index = pred.index
    except Exception:  # noqa: BLE001
        return moving_average_forecast(history, horizon)

    lower = pred - 1.96 * sigma
    upper = pred + 1.96 * sigma
    return ForecastResult(
        point_forecast=pd.Series(pred.values, index=future_index),
        lower=pd.Series(lower.values, index=future_index).clip(lower=0),
        upper=pd.Series(upper.values, index=future_index),
        model_name="holt_winters",
    )


def project_reorder_date(
    current_stock: float,
    forecast: ForecastResult,
    safety_stock: float = 0.0,
) -> Optional[pd.Timestamp]:
    """
    Estimate when inventory will dip below safety stock given a forecast.
    """
    available = current_stock - safety_stock
    if available <= 0:
        return forecast.point_forecast.index[0] if len(forecast.point_forecast) else None

    cumulative = forecast.point_forecast.cumsum()
    depletion = cumulative[cumulative >= available]
    if depletion.empty:
        return None
    return depletion.index[0]


def _build_future_index(index: pd.Index, horizon: int) -> pd.Index:
    """Generate a future index aligned with the input series frequency."""
    if isinstance(index, pd.DatetimeIndex) and index.freq:
        return pd.date_range(
            start=index[-1] + index.freq,
            periods=horizon,
            freq=index.freq,
        )
    if isinstance(index, pd.DatetimeIndex):
        inferred = index.inferred_freq or "D"
        return pd.date_range(start=index[-1], periods=horizon + 1, freq=inferred)[1:]
    return pd.RangeIndex(start=0, stop=horizon, step=1)
