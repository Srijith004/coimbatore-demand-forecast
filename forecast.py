"""
forecast.py
===========
Forecast population and electricity demand for Coimbatore up to 2040.
Uses scikit-learn LinearRegression and PolynomialFeatures(degree=2).
Compares models with R² score and back-testing (train on earlier years,
test on last 3 years). Picks the better model automatically.

Note: Simple regression on limited yearly data.
      The forecast is indicative only.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from typing import NamedTuple

from config import FORECAST_END_YEAR, CONFIDENCE_MARGIN


class ModelResult(NamedTuple):
    """Holds the result of a trained model."""
    name: str
    r2_full: float       # R² on full data
    r2_backtest: float   # R² on held-out last 3 years
    model: LinearRegression
    poly: PolynomialFeatures | None  # None for linear


class ForecastResult(NamedTuple):
    """Holds full forecast output."""
    forecast_df: pd.DataFrame        # year, population, demand_mw, per_capita, data_type
    best_pop_model: ModelResult
    best_demand_model: ModelResult
    method2_df: pd.DataFrame         # demand = forecast_pop × forecast_per_capita
    all_pop_models: list[ModelResult]
    all_demand_models: list[ModelResult]


def _train_model(
    X: np.ndarray, y: np.ndarray, degree: int, name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> ModelResult:
    """Train a linear or polynomial model and compute R² scores."""
    poly = None
    if degree > 1:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_full = poly.fit_transform(X)
        X_tr = poly.transform(X_train)
        X_te = poly.transform(X_test)
    else:
        X_full = X
        X_tr = X_train
        X_te = X_test

    # Full-data model
    model_full = LinearRegression()
    model_full.fit(X_full, y)
    r2_full = r2_score(y, model_full.predict(X_full))

    # Back-test model (train on earlier, test on last 3)
    model_bt = LinearRegression()
    model_bt.fit(X_tr, y_train)
    y_pred_test = model_bt.predict(X_te)
    r2_bt = r2_score(y_test, y_pred_test)

    return ModelResult(name=name, r2_full=r2_full, r2_backtest=r2_bt,
                       model=model_full, poly=poly)


def _predict(model_result: ModelResult, years: np.ndarray) -> np.ndarray:
    """Predict using a ModelResult for given years."""
    X = years.reshape(-1, 1)
    if model_result.poly is not None:
        X = model_result.poly.transform(X)
    return model_result.model.predict(X)


def build_forecast(data_df: pd.DataFrame) -> ForecastResult:
    """
    Build forecast models and generate predictions up to FORECAST_END_YEAR.

    Steps:
      1. Train linear and polynomial(deg=2) models for population and demand.
      2. Back-test each (train on all but last 3, test on last 3).
      3. Select the model with higher back-test R².
      4. Forecast population and demand for future years.
      5. Also compute Method 2: demand = forecast_pop × forecast_per_capita.
    """
    years = data_df["year"].values.reshape(-1, 1)
    pop = data_df["population"].values
    demand = data_df["demand_mw"].values

    # Split for back-testing
    n_test = 3
    X_train, X_test = years[:-n_test], years[-n_test:]
    pop_train, pop_test = pop[:-n_test], pop[-n_test:]
    dem_train, dem_test = demand[:-n_test], demand[-n_test:]

    # ── Population models ───────────────────────────────────────────
    pop_linear = _train_model(years, pop, 1, "Linear Regression",
                              X_train, pop_train, X_test, pop_test)
    pop_poly = _train_model(years, pop, 2, "Polynomial (degree 2)",
                            X_train, pop_train, X_test, pop_test)
    pop_models = [pop_linear, pop_poly]
    best_pop = max(pop_models, key=lambda m: m.r2_backtest)

    # ── Demand models ───────────────────────────────────────────────
    dem_linear = _train_model(years, demand, 1, "Linear Regression",
                              X_train, dem_train, X_test, dem_test)
    dem_poly = _train_model(years, demand, 2, "Polynomial (degree 2)",
                            X_train, dem_train, X_test, dem_test)
    dem_models = [dem_linear, dem_poly]
    best_dem = max(dem_models, key=lambda m: m.r2_backtest)

    # ── Per-capita demand model (for Method 2) ──────────────────────
    per_capita = demand / pop
    pc_linear = _train_model(years, per_capita, 1, "Linear Regression",
                             X_train, per_capita[:-n_test],
                             X_test, per_capita[-n_test:])
    pc_poly = _train_model(years, per_capita, 2, "Polynomial (degree 2)",
                           X_train, per_capita[:-n_test],
                           X_test, per_capita[-n_test:])
    best_pc = max([pc_linear, pc_poly], key=lambda m: m.r2_backtest)

    # ── Generate forecasts ──────────────────────────────────────────
    last_year = int(data_df["year"].max())
    future_years = np.arange(last_year + 1, FORECAST_END_YEAR + 1)

    if len(future_years) == 0:
        # No future years to forecast
        forecast_df = pd.DataFrame(
            columns=["year", "population", "demand_mw", "per_capita",
                     "pop_lower", "pop_upper", "demand_lower", "demand_upper",
                     "data_type"]
        )
        method2_df = forecast_df.copy()
        return ForecastResult(forecast_df, best_pop, best_dem, method2_df,
                              pop_models, dem_models)

    # Method 1: independent forecast of population and demand
    pop_forecast = _predict(best_pop, future_years)
    dem_forecast = _predict(best_dem, future_years)

    # Ensure non-negative
    pop_forecast = np.maximum(pop_forecast, 0)
    dem_forecast = np.maximum(dem_forecast, 0)

    forecast_df = pd.DataFrame({
        "year": future_years.astype(int),
        "population": pop_forecast,
        "demand_mw": dem_forecast,
        "per_capita": dem_forecast / np.maximum(pop_forecast, 1),
        "pop_lower": pop_forecast * (1 - CONFIDENCE_MARGIN),
        "pop_upper": pop_forecast * (1 + CONFIDENCE_MARGIN),
        "demand_lower": dem_forecast * (1 - CONFIDENCE_MARGIN),
        "demand_upper": dem_forecast * (1 + CONFIDENCE_MARGIN),
        "data_type": "forecast",
    })

    # Method 2: demand = forecast_pop × forecast_per_capita
    pc_forecast = _predict(best_pc, future_years)
    pc_forecast = np.maximum(pc_forecast, 0)
    dem_method2 = pop_forecast * pc_forecast

    method2_df = pd.DataFrame({
        "year": future_years.astype(int),
        "population": pop_forecast,
        "demand_mw": dem_method2,
        "per_capita": pc_forecast,
        "pop_lower": pop_forecast * (1 - CONFIDENCE_MARGIN),
        "pop_upper": pop_forecast * (1 + CONFIDENCE_MARGIN),
        "demand_lower": dem_method2 * (1 - CONFIDENCE_MARGIN),
        "demand_upper": dem_method2 * (1 + CONFIDENCE_MARGIN),
        "data_type": "forecast",
    })

    return ForecastResult(forecast_df, best_pop, best_dem, method2_df,
                          pop_models, dem_models)


def get_value_for_year(
    year: int, data_df: pd.DataFrame, forecast_result: ForecastResult
) -> dict:
    """
    Get population and demand for any year.
    Returns dict with keys: year, population, demand_mw, per_capita,
    source ('actual' or 'forecast'), model_name, r2_score,
    pop_lower, pop_upper, demand_lower, demand_upper.
    """
    # Check if year is in actual data
    actual = data_df[data_df["year"] == year]
    if not actual.empty:
        row = actual.iloc[0]
        pop = float(row["population"])
        dem = float(row["demand_mw"])
        return {
            "year": year,
            "population": pop,
            "demand_mw": dem,
            "per_capita": dem / pop if pop > 0 else 0,
            "source": "actual",
            "data_type": str(row["data_type"]),
            "model_name": "N/A (actual data)",
            "r2_score": None,
            "pop_lower": pop,
            "pop_upper": pop,
            "demand_lower": dem,
            "demand_upper": dem,
        }

    # Check if year is in forecast
    fc = forecast_result.forecast_df
    fc_row = fc[fc["year"] == year]
    if not fc_row.empty:
        row = fc_row.iloc[0]
        return {
            "year": year,
            "population": float(row["population"]),
            "demand_mw": float(row["demand_mw"]),
            "per_capita": float(row["per_capita"]),
            "source": "forecast",
            "data_type": "forecast",
            "model_name": forecast_result.best_demand_model.name,
            "r2_score": forecast_result.best_demand_model.r2_backtest,
            "pop_lower": float(row["pop_lower"]),
            "pop_upper": float(row["pop_upper"]),
            "demand_lower": float(row["demand_lower"]),
            "demand_upper": float(row["demand_upper"]),
        }

    # Year is beyond forecast range — predict on the fly
    year_arr = np.array([year])
    pop_val = float(_predict(forecast_result.best_pop_model, year_arr)[0])
    dem_val = float(_predict(forecast_result.best_demand_model, year_arr)[0])
    pop_val = max(pop_val, 0)
    dem_val = max(dem_val, 0)

    return {
        "year": year,
        "population": pop_val,
        "demand_mw": dem_val,
        "per_capita": dem_val / pop_val if pop_val > 0 else 0,
        "source": "forecast",
        "data_type": "forecast",
        "model_name": forecast_result.best_demand_model.name,
        "r2_score": forecast_result.best_demand_model.r2_backtest,
        "pop_lower": pop_val * (1 - CONFIDENCE_MARGIN),
        "pop_upper": pop_val * (1 + CONFIDENCE_MARGIN),
        "demand_lower": dem_val * (1 - CONFIDENCE_MARGIN),
        "demand_upper": dem_val * (1 + CONFIDENCE_MARGIN),
    }


def get_combined_df(data_df: pd.DataFrame, forecast_result: ForecastResult) -> pd.DataFrame:
    """
    Combine actual data and forecast into a single DataFrame for charting.
    Columns: year, population, demand_mw, per_capita, data_type.
    """
    actual = data_df[["year", "population", "demand_mw", "data_type"]].copy()
    actual["per_capita"] = actual["demand_mw"] / actual["population"]

    fc = forecast_result.forecast_df[["year", "population", "demand_mw", "per_capita", "data_type"]].copy()

    combined = pd.concat([actual, fc], ignore_index=True)
    combined = combined.sort_values("year").reset_index(drop=True)
    return combined
