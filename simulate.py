"""
Scenario simulation utilities for stress-testing inventory decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ScenarioResult:
    name: str
    description: str
    projected_usage: pd.Series
    notes: Optional[str] = None


def apply_demand_shock(
    baseline: pd.Series,
    shock_factor: float,
    duration: int,
    start_offset: int = 0,
) -> pd.Series:
    """
    Apply a temporary multiplicative shock to a demand series.
    """
    adjusted = baseline.copy().astype(float)
    idx = adjusted.index[start_offset : start_offset + duration]
    adjusted.loc[idx] = adjusted.loc[idx] * shock_factor
    return adjusted


def simulate_menu_launch(
    baseline_usage: pd.Series,
    incremental_units: float,
    ramp_days: int = 7,
) -> pd.Series:
    """
    Model the effect of a new menu item contributing incremental ingredient usage.
    """
    ramp = np.linspace(0, incremental_units, num=ramp_days, dtype=float)
    incremental = pd.Series(ramp, index=baseline_usage.index[:ramp_days]).reindex(
        baseline_usage.index, fill_value=incremental_units
    )
    return baseline_usage + incremental


def run_scenarios(
    baseline: pd.Series,
    scenarios: Dict[str, Callable[[pd.Series], pd.Series]],
) -> Dict[str, ScenarioResult]:
    """
    Execute a dictionary of scenario callables returning new usage projections.
    """
    results: Dict[str, ScenarioResult] = {}
    for name, fn in scenarios.items():
        projected = fn(baseline)
        results[name] = ScenarioResult(
            name=name,
            description=f"Custom scenario: {name}",
            projected_usage=projected,
        )
    return results

