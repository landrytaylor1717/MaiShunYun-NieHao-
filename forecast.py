import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def moving_avg(y: pd.Series, window: int) -> pd.Series:
    return y.rolling(window=window).mean()

def exponential_smoothing(y, alpha=0.1):
    return ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=7).fit(smoothing_level=alpha).forecast(len(y))

def double_exponential_smoothing(y, alpha=0.1, beta=0.1):
    return ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=7).fit(smoothing_level=alpha, smoothing_slope=beta).forecast(len(y))

def triple_exponential_smoothing(y, alpha=0.1, beta=0.1, gamma=0.1):
    return ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=7).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(len(y))