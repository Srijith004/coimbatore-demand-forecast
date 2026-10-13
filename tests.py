"""
tests.py
========
Pytest tests for the Coimbatore Demand Forecast app.
Run with:  pytest tests.py -v

Tests:
  1. Area shares add up to 100
  2. Area demand sums to district (total) demand
  3. R² sanity (> 0)
  4. Unit round-trips
  5. CSV validation
"""

import pytest
import pandas as pd
import numpy as np

from data_loader import validate_data_csv, validate_areas_csv
from forecast import build_forecast, get_value_for_year
from areas import (
    get_area_share, compute_area_demand, compute_all_area_demands,
    get_demand_for_selection,
)
from units import convert_demand, convert_population


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def data_df() -> pd.DataFrame:
    """Load the default data CSV."""
    df = pd.read_csv("data/coimbatore_data.csv")
    df["year"] = df["year"].astype(int)
    df["population"] = df["population"].astype(float)
    df["demand_mw"] = df["demand_mw"].astype(float)
    df["data_type"] = df["data_type"].str.strip().str.lower()
    df = df.sort_values("year").reset_index(drop=True)
    return df


@pytest.fixture
def areas_df() -> pd.DataFrame:
    """Load the default areas CSV."""
    df = pd.read_csv("data/coimbatore_areas.csv")
    df["share_percent"] = df["share_percent"].astype(float)
    df["data_type"] = df["data_type"].str.strip().str.lower()
    return df


@pytest.fixture
def forecast_result(data_df):
    """Build forecast from data."""
    return build_forecast(data_df)


# ── Test 1: Area shares add to 100 ─────────────────────────────────

def test_area_shares_sum_to_100(areas_df: pd.DataFrame):
    """Area share_percent values must add up to exactly 100."""
    total = areas_df["share_percent"].sum()
    assert abs(total - 100.0) < 0.01, f"Shares sum to {total}, expected 100"


# ── Test 2: Area demand sums to district demand ────────────────────

def test_area_demand_sums_to_total(areas_df: pd.DataFrame):
    """Sum of all area demands must equal the total Coimbatore demand."""
    total_demand = 1000.0  # test value
    area_demands = compute_all_area_demands(total_demand, areas_df)
    area_sum = area_demands["demand_mw"].sum()
    assert abs(area_sum - total_demand) < 0.01, (
        f"Area demands sum to {area_sum}, expected {total_demand}"
    )


def test_area_demand_single(areas_df: pd.DataFrame):
    """Single area demand = total × share / 100."""
    total_demand = 500.0
    share = get_area_share(areas_df, "Gandhipuram")
    expected = total_demand * share / 100.0
    actual = compute_area_demand(total_demand, share)
    assert abs(actual - expected) < 0.01


# ── Test 3: R² sanity ──────────────────────────────────────────────

def test_r2_scores_positive(forecast_result):
    """R² scores for the best models should be > 0 (at least better than mean)."""
    assert forecast_result.best_pop_model.r2_full > 0, "Pop model R² full <= 0"
    assert forecast_result.best_demand_model.r2_full > 0, "Demand model R² full <= 0"


def test_r2_backtest_reasonable(forecast_result):
    """Back-test R² should be a reasonable number (> -1 at minimum)."""
    assert forecast_result.best_pop_model.r2_backtest > -1.0
    assert forecast_result.best_demand_model.r2_backtest > -1.0


# ── Test 4: Unit round-trips ───────────────────────────────────────

def test_demand_unit_mw_identity():
    """Converting to MW should return the same value."""
    assert convert_demand(100.0, "MW") == 100.0


def test_demand_unit_kw():
    """1 MW = 1000 kW."""
    assert convert_demand(1.0, "kW") == 1000.0


def test_demand_unit_gw():
    """1000 MW = 1 GW."""
    assert abs(convert_demand(1000.0, "GW") - 1.0) < 0.001


def test_population_unit_identity():
    """Converting to 'Number' should return the same value."""
    assert convert_population(1000000, "Number") == 1000000


def test_population_unit_lakhs():
    """1,00,000 = 1 lakh."""
    assert abs(convert_population(100000, "Lakhs") - 1.0) < 0.001


def test_population_unit_millions():
    """1,000,000 = 1 million."""
    assert abs(convert_population(1000000, "Millions") - 1.0) < 0.001


# ── Test 5: CSV validation ─────────────────────────────────────────

def test_valid_data_csv(data_df: pd.DataFrame):
    """The default data CSV should pass validation."""
    errors = validate_data_csv(data_df)
    assert errors == [], f"Unexpected errors: {errors}"


def test_valid_areas_csv(areas_df: pd.DataFrame):
    """The default areas CSV should pass validation."""
    errors = validate_areas_csv(areas_df)
    assert errors == [], f"Unexpected errors: {errors}"


def test_invalid_data_csv_missing_col():
    """Should flag missing columns."""
    bad_df = pd.DataFrame({"year": [2020], "population": [1000]})
    errors = validate_data_csv(bad_df)
    assert len(errors) > 0
    assert "Missing columns" in errors[0]


def test_invalid_areas_csv_shares_wrong():
    """Should flag shares not adding to 100."""
    bad_df = pd.DataFrame({
        "area": ["A", "B"],
        "area_type": ["X", "Y"],
        "share_percent": [40, 40],
        "data_type": ["dummy", "dummy"],
    })
    errors = validate_areas_csv(bad_df)
    assert len(errors) > 0
    assert "100" in errors[0]


def test_invalid_data_csv_duplicate_years():
    """Should flag duplicate years."""
    bad_df = pd.DataFrame({
        "year": [2020, 2020],
        "population": [1000, 2000],
        "demand_mw": [100, 200],
        "data_type": ["dummy", "dummy"],
    })
    errors = validate_data_csv(bad_df)
    assert any("Duplicate" in e for e in errors)


# ── Test 6: Forecast value retrieval ────────────────────────────────

def test_get_actual_year(data_df, forecast_result):
    """Getting a value for a year in the CSV should return 'actual'."""
    result = get_value_for_year(2020, data_df, forecast_result)
    assert result["source"] == "actual"
    assert result["demand_mw"] > 0


def test_get_forecast_year(data_df, forecast_result):
    """Getting a value for a future year should return 'forecast'."""
    result = get_value_for_year(2035, data_df, forecast_result)
    assert result["source"] == "forecast"
    assert result["demand_mw"] > 0


# ── Test 7: get_demand_for_selection ────────────────────────────────

def test_all_coimbatore_selection(areas_df):
    """'All Coimbatore' should return total demand."""
    demand, label = get_demand_for_selection(1000.0, areas_df, "All Coimbatore")
    assert demand == 1000.0
    assert label == "All Coimbatore"


def test_area_selection(areas_df):
    """Selecting an area should return area-specific demand."""
    demand, label = get_demand_for_selection(1000.0, areas_df, "Gandhipuram")
    expected = 1000.0 * 12.0 / 100.0  # 12% share
    assert abs(demand - expected) < 0.01
    assert label == "Gandhipuram"
