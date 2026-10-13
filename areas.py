"""
areas.py
========
Area-level demand calculation and helpers.
Area demand = Coimbatore total demand × area share_percent / 100.
"""

import pandas as pd
from typing import Optional


def get_area_list(areas_df: pd.DataFrame) -> list[str]:
    """Return the list of area names from the areas DataFrame."""
    return areas_df["area"].tolist()


def get_area_share(areas_df: pd.DataFrame, area_name: str) -> float:
    """Return the share_percent for a given area."""
    row = areas_df[areas_df["area"] == area_name]
    if row.empty:
        return 0.0
    return float(row["share_percent"].iloc[0])


def get_area_type(areas_df: pd.DataFrame, area_name: str) -> str:
    """Return the area_type for a given area."""
    row = areas_df[areas_df["area"] == area_name]
    if row.empty:
        return "Unknown"
    return str(row["area_type"].iloc[0])


def compute_area_demand(total_demand_mw: float, share_percent: float) -> float:
    """Calculate area demand from total Coimbatore demand and area share."""
    return total_demand_mw * share_percent / 100.0


def compute_all_area_demands(
    total_demand_mw: float, areas_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute demand for all areas given a total Coimbatore demand.
    Returns a DataFrame with columns: area, area_type, share_percent, demand_mw.
    """
    result = areas_df.copy()
    result["demand_mw"] = result["share_percent"].apply(
        lambda s: compute_area_demand(total_demand_mw, s)
    )
    return result


def get_demand_for_selection(
    total_demand_mw: float,
    areas_df: pd.DataFrame,
    selected_area: Optional[str],
) -> tuple[float, str]:
    """
    Get the demand value and label for the current selection.
    If selected_area is None or 'All Coimbatore', returns total demand.
    Otherwise returns the area-specific demand.
    Returns (demand_mw, label_string).
    """
    if selected_area is None or selected_area == "All Coimbatore":
        return total_demand_mw, "All Coimbatore"

    share = get_area_share(areas_df, selected_area)
    area_demand = compute_area_demand(total_demand_mw, share)
    return area_demand, selected_area
