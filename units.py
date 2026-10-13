"""
units.py
========
Unit conversion helpers for demand and population display.
Calculations always stay in base units (MW for demand, raw number for population).
Only display values are converted.
"""

from typing import Literal

# ── Demand units ────────────────────────────────────────────────────
DemandUnit = Literal["kW", "MW", "GW", "MU", "TWh"]

DEMAND_UNIT_OPTIONS: list[str] = ["kW", "MW", "GW", "MU", "TWh"]


def convert_demand(value_mw: float, unit: DemandUnit) -> float:
    """Convert demand from MW to the selected unit."""
    if unit == "kW":
        return value_mw * 1_000.0
    elif unit == "MW":
        return value_mw
    elif unit == "GW":
        return value_mw / 1_000.0
    elif unit == "MU":
        # 1 MW running for 8760 hours = 8.76 MU (million units / GWh)
        return value_mw * 8.76
    elif unit == "TWh":
        # 1 MW * 8760 h = 8760 MWh = 0.00876 TWh
        return value_mw * 0.00876
    return value_mw


def demand_label(unit: DemandUnit) -> str:
    """Return a human-readable label for the demand unit."""
    return unit


# ── Population units ────────────────────────────────────────────────
PopUnit = Literal["Number", "Lakhs", "Millions"]

POP_UNIT_OPTIONS: list[str] = ["Number", "Lakhs", "Millions"]


def convert_population(value: float, unit: PopUnit) -> float:
    """Convert population from raw number to the selected unit."""
    if unit == "Number":
        return value
    elif unit == "Lakhs":
        return value / 1_00_000.0
    elif unit == "Millions":
        return value / 1_000_000.0
    return value


def population_label(unit: PopUnit) -> str:
    """Return a human-readable label for the population unit."""
    if unit == "Number":
        return "Population"
    return f"Population ({unit})"


def format_demand(value_mw: float, unit: DemandUnit) -> str:
    """Format a demand value with its unit for display."""
    converted = convert_demand(value_mw, unit)
    if converted >= 1000:
        return f"{converted:,.1f} {unit}"
    return f"{converted:,.2f} {unit}"


def format_population(value: float, unit: PopUnit) -> str:
    """Format a population value with its unit for display."""
    converted = convert_population(value, unit)
    if unit == "Number":
        return f"{converted:,.0f}"
    return f"{converted:,.2f} {unit}"
