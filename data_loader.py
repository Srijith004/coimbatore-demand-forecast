"""
data_loader.py
==============
Loads, validates, and manages the CSV data files.
Provides template generation for missing files and handles user uploads.
"""

import pandas as pd
import streamlit as st
import io
from typing import Optional


# ── Required columns ────────────────────────────────────────────────
DATA_REQUIRED_COLS: list[str] = ["year", "population", "demand_mw", "data_type"]
AREAS_REQUIRED_COLS: list[str] = ["area", "area_type", "share_percent", "data_type"]


def generate_data_template() -> str:
    """Return a template CSV string for coimbatore_data.csv."""
    return "year,population,demand_mw,data_type\n2020,4000000,800,dummy\n"


def generate_areas_template() -> str:
    """Return a template CSV string for coimbatore_areas.csv."""
    lines = [
        "area,area_type,share_percent,data_type",
        "Gandhipuram,Commercial,12,dummy",
        "RS Puram,Residential + Commercial,9,dummy",
        "Peelamedu,Industrial + Educational,14,dummy",
        "Saibaba Colony,Residential,7,dummy",
        "Singanallur,Industrial,13,dummy",
        "Ukkadam,Commercial + Market,8,dummy",
        "Saravanampatti,IT + Residential,11,dummy",
        "Vadavalli,Residential,7,dummy",
        "Thudiyalur,Residential + Small Industry,10,dummy",
        "Podanur,Industrial + Railway,9,dummy",
    ]
    return "\n".join(lines) + "\n"


def validate_data_csv(df: pd.DataFrame) -> list[str]:
    """
    Validate the main data CSV.
    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []

    # Check required columns
    missing = [c for c in DATA_REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
        return errors  # can't continue without columns

    # Check for non-numeric values in numeric columns
    for col in ["year", "population", "demand_mw"]:
        if not pd.to_numeric(df[col], errors="coerce").notna().all():
            errors.append(f"Column '{col}' contains non-numeric values.")

    # Check for duplicate years
    if df["year"].duplicated().any():
        dup_years = df.loc[df["year"].duplicated(), "year"].tolist()
        errors.append(f"Duplicate years found: {dup_years}")

    # Check data_type values
    valid_types = {"dummy", "official"}
    invalid = set(df["data_type"].str.strip().str.lower()) - valid_types
    if invalid:
        errors.append(f"Invalid data_type values: {invalid}. Use 'dummy' or 'official'.")

    return errors


def validate_areas_csv(df: pd.DataFrame) -> list[str]:
    """
    Validate the areas CSV.
    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []

    # Check required columns
    missing = [c for c in AREAS_REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
        return errors

    # Check for non-numeric share_percent
    if not pd.to_numeric(df["share_percent"], errors="coerce").notna().all():
        errors.append("Column 'share_percent' contains non-numeric values.")
    else:
        total = pd.to_numeric(df["share_percent"]).sum()
        if abs(total - 100.0) > 0.01:
            errors.append(f"Area shares sum to {total:.2f}%, not 100%.")

    # Check area count
    if len(df) < 1:
        errors.append("Areas CSV is empty.")

    return errors


def load_csv_file(
    path: str,
    required_cols: list[str],
    validator,
    template_fn,
    label: str,
) -> Optional[pd.DataFrame]:
    """
    Load a CSV from disk, validate it, and return the DataFrame.
    If the file is missing, shows a message with a downloadable template.
    Returns None on failure.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"📁 **{label}** file not found at `{path}`.")
        template = template_fn()
        st.download_button(
            f"⬇️ Download {label} template",
            data=template,
            file_name=path.split("/")[-1],
            mime="text/csv",
        )
        return None
    except Exception as e:
        st.error(f"❌ Error reading {label}: {e}")
        return None

    errors = validator(df)
    if errors:
        for err in errors:
            st.error(f"❌ {label} validation error: {err}")
        return None

    return df


def load_data(data_path: str, areas_path: str) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load both CSVs with validation.
    Returns (data_df, areas_df) – either may be None on failure.
    """
    data_df = load_csv_file(
        data_path, DATA_REQUIRED_COLS, validate_data_csv,
        generate_data_template, "Coimbatore Data",
    )
    areas_df = load_csv_file(
        areas_path, AREAS_REQUIRED_COLS, validate_areas_csv,
        generate_areas_template, "Coimbatore Areas",
    )

    # Clean up types if loaded
    if data_df is not None:
        data_df["year"] = data_df["year"].astype(int)
        data_df["population"] = data_df["population"].astype(float)
        data_df["demand_mw"] = data_df["demand_mw"].astype(float)
        data_df["data_type"] = data_df["data_type"].str.strip().str.lower()
        data_df = data_df.sort_values("year").reset_index(drop=True)

    if areas_df is not None:
        areas_df["share_percent"] = areas_df["share_percent"].astype(float)
        areas_df["data_type"] = areas_df["data_type"].str.strip().str.lower()

    return data_df, areas_df


def parse_uploaded_csv(uploaded_file, validator) -> Optional[pd.DataFrame]:
    """
    Parse an uploaded CSV file and validate it.
    Returns the DataFrame or None on failure.
    """
    try:
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        errors = validator(df)
        if errors:
            for err in errors:
                st.error(f"❌ Upload validation error: {err}")
            return None
        return df
    except Exception as e:
        st.error(f"❌ Error parsing upload: {e}")
        return None


def has_dummy_data(data_df: pd.DataFrame, areas_df: pd.DataFrame) -> bool:
    """Check if any row in either CSV has data_type = 'dummy'."""
    dummy_in_data = (data_df["data_type"] == "dummy").any()
    dummy_in_areas = (areas_df["data_type"] == "dummy").any()
    return dummy_in_data or dummy_in_areas
