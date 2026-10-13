"""
config.py
=========
Central configuration for the Coimbatore Demand Forecast app.
Defines colour palette, Gemini model name, file paths, and Plotly template.
"""

# ── Colour Palette ──────────────────────────────────────────────────
SKY_BLUE: str   = "#87CEEB"
OLIVE_GREEN: str = "#6B8E23"
BLACK: str       = "#111111"
WHITE: str       = "#FFFFFF"
LIGHT_BLUE: str  = "#D6EEF9"   # cloud card background

# Extended palette for charts (10 colours for the 10 areas)
AREA_COLORS: list[str] = [
    "#87CEEB", "#6B8E23", "#4682B4", "#2E8B57", "#5F9EA0",
    "#3CB371", "#66CDAA", "#48D1CC", "#20B2AA", "#008B8B",
]

# ── Gemini API ──────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.0-flash"

# ── Data file paths ─────────────────────────────────────────────────
DATA_CSV_PATH: str  = "data/coimbatore_data.csv"
AREAS_CSV_PATH: str = "data/coimbatore_areas.csv"

# ── Forecast settings ───────────────────────────────────────────────
FORECAST_END_YEAR: int = 2040
CONFIDENCE_MARGIN: float = 0.05   # ±5 % for upper/lower range

# ── Plotly template (reusable) ──────────────────────────────────────
import plotly.graph_objects as go  # noqa: E402

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, sans-serif", color=BLACK),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        colorway=AREA_COLORS,
        title_font=dict(size=18, color=BLACK),
        xaxis=dict(gridcolor="#E0E0E0", linecolor=BLACK, zerolinecolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0", linecolor=BLACK, zerolinecolor="#E0E0E0"),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor=BLACK, borderwidth=0.5),
    )
)
