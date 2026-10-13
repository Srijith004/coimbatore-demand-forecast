"""
ai_assistant.py
===============
Gemini API integration for AI reports and Q&A.
Uses the google-genai SDK: from google import genai.
API key from GEMINI_API_KEY env var or st.secrets.
NEVER hard-codes the key. NEVER executes AI-generated code.
"""

import os
import json
import streamlit as st
from typing import Optional

from config import GEMINI_MODEL


def _get_api_key() -> Optional[str]:
    """Retrieve the Gemini API key from env or Streamlit secrets."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return None


def _get_client():
    """Create a Gemini client. Returns None if no API key is available."""
    key = _get_api_key()
    if not key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        return client
    except Exception as e:
        st.error(f"❌ Failed to create Gemini client: {e}")
        return None


def generate_ai_report(
    year: int,
    area: str,
    demand_mw: float,
    source: str,
    population: float,
    per_capita: float,
    compare_year1: int,
    compare_year1_demand: float,
    compare_year2: int,
    compare_year2_demand: float,
    is_dummy: bool,
    model_name: str = "",
    r2_score_val: Optional[float] = None,
) -> str:
    """
    Send data to Gemini and get a short summary report (under 120 words).
    Uses only the given numbers; says the forecast is an estimate;
    mentions if dummy data is used.
    """
    client = _get_client()
    if client is None:
        return "⚠️ Gemini API key not configured. Set GEMINI_API_KEY in environment or .streamlit/secrets.toml."

    dummy_note = "The data used is DUMMY/SAMPLE data for demonstration purposes. Mention this." if is_dummy else ""

    prompt = f"""You are an electricity demand analyst for Coimbatore, India.
Write a short summary report (UNDER 120 words) using ONLY the numbers provided below.
Do NOT invent or look up any additional data.

Data:
- Year: {year}
- Area: {area}
- Demand: {demand_mw:.2f} MW
- Source: {source} {"(this is a forecast/estimate, clearly say it is an estimate)" if source == "forecast" else "(actual recorded data)"}
- Population: {population:,.0f}
- Per capita demand: {per_capita:.6f} MW/person
- Comparison: {compare_year1} demand = {compare_year1_demand:.2f} MW vs {compare_year2} demand = {compare_year2_demand:.2f} MW
- Forecast model: {model_name}, R² = {r2_score_val if r2_score_val is not None else 'N/A'}
{dummy_note}

Write concisely. Include the key numbers. If forecast, say it is an estimate based on regression."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"


def ask_question(
    question: str,
    data_summary: str,
) -> dict:
    """
    Send a user question to Gemini and get structured JSON back.
    Gemini returns JSON only with keys:
      year, area, population_growth_change_pct, per_capita_change_pct
    Python then calculates the result — we NEVER exec AI code.

    Returns a dict with the parsed values, or {"error": "message"}.
    """
    client = _get_client()
    if client is None:
        return {"error": "Gemini API key not configured. Set GEMINI_API_KEY in environment or .streamlit/secrets.toml."}

    prompt = f"""You are an assistant for the Coimbatore Demand Forecast app.
The user asked: "{question}"

Context data:
{data_summary}

You MUST respond with ONLY valid JSON (no markdown, no explanation) with these keys:
- "year": integer (the year the user is asking about, default 2030)
- "area": string (one of the 10 Coimbatore areas, or "All Coimbatore")
- "population_growth_change_pct": float (percentage change to add to population growth rate, default 0.0)
- "per_capita_change_pct": float (percentage change to add to per-capita demand growth, default 0.0)

Examples:
- "what is the demand of Peelamedu in 2032?" → {{"year": 2032, "area": "Peelamedu", "population_growth_change_pct": 0.0, "per_capita_change_pct": 0.0}}
- "what if population grows 1% faster?" → {{"year": 2030, "area": "All Coimbatore", "population_growth_change_pct": 1.0, "per_capita_change_pct": 0.0}}

Respond with JSON ONLY."""

    try:
        from google import genai

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()

        # Parse JSON
        parsed = json.loads(raw)

        # Validate required keys
        required_keys = ["year", "area", "population_growth_change_pct", "per_capita_change_pct"]
        for key in required_keys:
            if key not in parsed:
                return {"error": f"Missing key in AI response: {key}"}

        # Validate types
        parsed["year"] = int(parsed["year"])
        parsed["population_growth_change_pct"] = float(parsed["population_growth_change_pct"])
        parsed["per_capita_change_pct"] = float(parsed["per_capita_change_pct"])
        parsed["area"] = str(parsed["area"])

        return parsed

    except json.JSONDecodeError as e:
        return {"error": f"Gemini returned invalid JSON: {e}"}
    except Exception as e:
        return {"error": f"Gemini API error: {e}"}
