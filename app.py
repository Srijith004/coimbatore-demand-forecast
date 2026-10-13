"""
app.py
======
Main Streamlit application for Coimbatore Demand Forecast.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import io

from config import (
    SKY_BLUE, OLIVE_GREEN, BLACK, WHITE, LIGHT_BLUE,
    AREA_COLORS, PLOTLY_TEMPLATE, DATA_CSV_PATH, AREAS_CSV_PATH,
    FORECAST_END_YEAR, GEMINI_MODEL,
)
from data_loader import (
    load_data, parse_uploaded_csv, validate_data_csv, validate_areas_csv,
    has_dummy_data, generate_data_template, generate_areas_template,
)
from forecast import build_forecast, get_value_for_year, get_combined_df
from areas import (
    get_area_list, get_area_share, get_area_type,
    compute_area_demand, compute_all_area_demands, get_demand_for_selection,
)
from units import (
    convert_demand, convert_population, format_demand, format_population,
    demand_label, population_label, DEMAND_UNIT_OPTIONS, POP_UNIT_OPTIONS,
)
from ai_assistant import generate_ai_report, ask_question

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Page config
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
import base64, pathlib

_logo_path = pathlib.Path(__file__).parent / "assets" / "logo.png"
_logo_b64 = ""
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()

st.set_page_config(
    page_title="Coimbatore Demand Forecast",
    page_icon="âš¡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Custom CSS with animations & icons
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
    html, body, [class*="st-"] {{ font-family: 'Inter', sans-serif; }}

    /* â”€â”€ Keyframe Animations â”€â”€ */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(24px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulseGlow {{
        0%, 100% {{ box-shadow: 0 0 12px rgba(135,206,235,0.3); }}
        50%      {{ box-shadow: 0 0 28px rgba(135,206,235,0.65); }}
    }}
    @keyframes shimmer {{
        0%   {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}
    @keyframes floatBolt {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        25%      {{ transform: translateY(-6px) rotate(3deg); }}
        75%      {{ transform: translateY(4px) rotate(-2deg); }}
    }}
    @keyframes gradientShift {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-30px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}

    /* â”€â”€ Header bar with animated gradient â”€â”€ */
    .main-header {{
        background: linear-gradient(135deg, {BLACK}, #1a1a2e, #16213e, {BLACK});
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        padding: 22px 30px 18px 30px;
        border-bottom: 4px solid {SKY_BLUE};
        margin: -1rem -1rem 1.5rem -1rem;
        border-radius: 0 0 14px 14px;
        display: flex;
        align-items: center;
        gap: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    .header-logo {{
        width: 56px; height: 56px;
        border-radius: 12px;
        animation: floatBolt 4s ease-in-out infinite;
        box-shadow: 0 0 16px rgba(135,206,235,0.4);
    }}
    .main-header h1 {{
        color: {WHITE}; margin: 0; font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, {WHITE}, {SKY_BLUE}, {WHITE});
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 4s linear infinite;
    }}
    .main-header p {{
        color: {SKY_BLUE}; margin: 4px 0 0 0; font-size: 1rem;
        letter-spacing: 0.5px;
    }}

    /* â”€â”€ Result card â”€â”€ */
    .result-card {{
        background: linear-gradient(135deg, {SKY_BLUE}22, {SKY_BLUE}44);
        border: 2px solid {SKY_BLUE};
        border-radius: 18px;
        padding: 28px 32px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(135,206,235,0.2);
        animation: fadeInUp 0.6s ease-out;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .result-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(135,206,235,0.35);
    }}
    .result-card .label {{
        font-size: 1rem; color: #555; margin-bottom: 6px;
    }}
    .result-card .value {{
        font-size: 2.6rem; font-weight: 700; color: {BLACK};
    }}
    .result-card .forecast-value {{
        color: {OLIVE_GREEN};
        animation: pulseGlow 2.5s ease-in-out infinite;
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
    }}
    .result-card .source-tag {{
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; margin-top: 8px;
        transition: transform 0.2s;
    }}
    .result-card .source-tag:hover {{ transform: scale(1.08); }}
    .tag-actual {{ background: {SKY_BLUE}; color: {BLACK}; }}
    .tag-forecast {{ background: {OLIVE_GREEN}; color: {WHITE}; }}

    /* â”€â”€ Cloud card â”€â”€ */
    .cloud-card {{
        position: relative;
        background: linear-gradient(145deg, {LIGHT_BLUE}, #c8e6f5);
        border-radius: 24px;
        padding: 22px 20px 18px 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        font-size: 0.85rem;
        color: {BLACK};
        line-height: 1.6;
        margin-top: 10px;
        animation: fadeInUp 0.8s ease-out;
        transition: box-shadow 0.3s ease;
    }}
    .cloud-card:hover {{
        box-shadow: 0 8px 30px rgba(0,0,0,0.14);
    }}
    .cloud-card::before {{
        content: '';
        position: absolute;
        width: 50px; height: 50px;
        background: {LIGHT_BLUE};
        border-radius: 50%;
        top: -20px; left: 25px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .cloud-card::after {{
        content: '';
        position: absolute;
        width: 70px; height: 70px;
        background: {LIGHT_BLUE};
        border-radius: 50%;
        top: -30px; left: 60px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .cloud-card h4 {{
        margin: 0 0 10px 0; color: {BLACK}; font-size: 1rem;
    }}
    .cloud-step {{
        margin: 3px 0; font-size: 0.82rem;
        animation: slideInLeft 0.5s ease-out backwards;
    }}
    .cloud-step:nth-child(2) {{ animation-delay: 0.1s; }}
    .cloud-step:nth-child(3) {{ animation-delay: 0.2s; }}
    .cloud-step:nth-child(4) {{ animation-delay: 0.3s; }}
    .cloud-step:nth-child(5) {{ animation-delay: 0.4s; }}

    /* â”€â”€ KPI cards â”€â”€ */
    .kpi-card {{
        background: {WHITE};
        border: 1.5px solid #ddd;
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        animation: fadeInUp 0.5s ease-out;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 6px 24px rgba(0,0,0,0.10);
        border-color: {SKY_BLUE};
    }}
    .kpi-card .kpi-label {{ font-size: 0.82rem; color: #777; }}
    .kpi-card .kpi-value {{ font-size: 1.6rem; font-weight: 700; color: {BLACK}; }}
    .kpi-card .kpi-change {{ font-size: 0.85rem; margin-top: 4px; }}
    .kpi-positive {{ color: {OLIVE_GREEN}; }}
    .kpi-negative {{ color: #c0392b; }}

    /* â”€â”€ Dummy banner â”€â”€ */
    .dummy-banner {{
        background: linear-gradient(90deg, #FFF3CD, #FFECB5, #FFF3CD);
        background-size: 200% auto;
        animation: shimmer 3s linear infinite;
        border: 1px solid #FFECB5;
        border-radius: 10px;
        padding: 12px 20px;
        color: #856404;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }}

    /* â”€â”€ Buttons â”€â”€ */
    .stButton > button {{
        background: linear-gradient(135deg, {OLIVE_GREEN}, #7a9e2a);
        color: {WHITE};
        border: none;
        border-radius: 10px;
        padding: 8px 22px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(107,142,35,0.3);
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, {SKY_BLUE}, #6bb5d9);
        color: {BLACK};
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(135,206,235,0.4);
    }}

    /* â”€â”€ Tabs â”€â”€ */
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {SKY_BLUE};
    }}

    /* â”€â”€ Sidebar styling â”€â”€ */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0d1117, #161b22);
        border-right: 2px solid {SKY_BLUE}33;
    }}
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h5 {{
        color: {SKY_BLUE};
    }}

    /* â”€â”€ Scroll fade effect â”€â”€ */
    .stApp > header {{ background: transparent; }}

    /* â”€â”€ Footer â”€â”€ */
    .app-footer {{
        text-align: center;
        padding: 20px;
        color: #888;
        font-size: 0.8rem;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }}
    .app-footer a {{ color: {SKY_BLUE}; text-decoration: none; }}
    .app-footer a:hover {{ text-decoration: underline; }}
</style>
""", unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Header with logo
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
_logo_html = f'<img src="data:image/png;base64,{_logo_b64}" class="header-logo" alt="Logo"/>' if _logo_b64 else 'âš¡'
st.markdown(f"""
<div class="main-header">
    {_logo_html}
    <div>
        <h1>âš¡ Coimbatore Demand Forecast</h1>
        <p>ðŸ™ï¸ Population vs Power Demand  Â·  ðŸ“Š ML Forecast  Â·  ðŸ¤– AI Insights</p>
    </div>
</div>
""", unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Session state defaults
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if "demand_unit" not in st.session_state:
    st.session_state.demand_unit = "MW"
if "pop_unit" not in st.session_state:
    st.session_state.pop_unit = "Number"

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Load data
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
data_path = DATA_CSV_PATH
areas_path = AREAS_CSV_PATH

data_df, areas_df = load_data(data_path, areas_path)

if data_df is None or areas_df is None:
    st.stop()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Sidebar
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with st.sidebar:
    # Sidebar logo
    if _logo_path.exists():
        st.image(str(_logo_path), width=80)
    st.markdown(f"""
    <div style="text-align:center;margin-bottom:12px;">
        <span style="background:{SKY_BLUE};color:{BLACK};padding:3px 12px;
        border-radius:20px;font-size:0.72rem;font-weight:600;">v1.0 Â· ML Forecast</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"### âš™ï¸ Controls")

    min_year = int(data_df["year"].min())
    max_year = FORECAST_END_YEAR

    selected_year = st.number_input(
        "ðŸ“… Year", min_value=min_year, max_value=max_year, value=2030, step=1
    )
    selected_year_slider = st.slider(
        "Year slider", min_value=min_year, max_value=max_year, value=int(selected_year)
    )
    # Sync slider â†” number input
    if selected_year_slider != int(selected_year):
        selected_year = selected_year_slider

    area_options = ["All Coimbatore"] + get_area_list(areas_df)
    selected_area = st.selectbox("ðŸ“ Area", area_options, index=0)

    st.markdown("---")
    st.markdown("##### ðŸ“ Display Units")
    st.session_state.demand_unit = st.selectbox(
        "Demand unit", DEMAND_UNIT_OPTIONS,
        index=DEMAND_UNIT_OPTIONS.index(st.session_state.demand_unit),
    )
    st.session_state.pop_unit = st.selectbox(
        "Population unit", POP_UNIT_OPTIONS,
        index=POP_UNIT_OPTIONS.index(st.session_state.pop_unit),
    )
    if st.button("ðŸ”„ Reset Units"):
        st.session_state.demand_unit = "MW"
        st.session_state.pop_unit = "Number"
        st.rerun()

    st.markdown("---")
    st.markdown("##### ðŸ“ Upload Your Data")
    up_data = st.file_uploader("Upload coimbatore_data.csv", type="csv", key="up_data")
    up_areas = st.file_uploader("Upload coimbatore_areas.csv", type="csv", key="up_areas")

    if up_data is not None:
        new_df = parse_uploaded_csv(up_data, validate_data_csv)
        if new_df is not None:
            data_df = new_df
            data_df["year"] = data_df["year"].astype(int)
            data_df["population"] = data_df["population"].astype(float)
            data_df["demand_mw"] = data_df["demand_mw"].astype(float)
            data_df["data_type"] = data_df["data_type"].str.strip().str.lower()
            data_df = data_df.sort_values("year").reset_index(drop=True)
            st.success("âœ… Data CSV uploaded successfully!")

    if up_areas is not None:
        new_areas = parse_uploaded_csv(up_areas, validate_areas_csv)
        if new_areas is not None:
            areas_df = new_areas
            areas_df["share_percent"] = areas_df["share_percent"].astype(float)
            areas_df["data_type"] = areas_df["data_type"].str.strip().str.lower()
            st.success("âœ… Areas CSV uploaded successfully!")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Dummy data banner
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
is_dummy = has_dummy_data(data_df, areas_df)
if is_dummy:
    st.markdown(
        '<div class="dummy-banner">âš ï¸ <b>Demo mode:</b> this project uses '
        'sample (dummy) data for demonstration.</div>',
        unsafe_allow_html=True,
    )

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Build forecast
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
forecast_result = build_forecast(data_df)
combined_df = get_combined_df(data_df, forecast_result)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Get values for the selected year
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
year_data = get_value_for_year(int(selected_year), data_df, forecast_result)
total_demand = year_data["demand_mw"]
area_demand, area_label = get_demand_for_selection(total_demand, areas_df, selected_area)
area_share = get_area_share(areas_df, selected_area) if selected_area != "All Coimbatore" else 100.0

du = st.session_state.demand_unit
pu = st.session_state.pop_unit

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TABS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
tab_overview, tab_charts, tab_old, tab_forecast, tab_ai, tab_data_tab = st.tabs(
    ["ðŸ“Š Overview", "ðŸ“ˆ Charts", "ðŸ“‹ Old Data", "ðŸ”® Forecast", "ðŸ¤– AI Assistant", "ðŸ’¾ Data"]
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 1: OVERVIEW
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab_overview:
    col_result, col_cloud = st.columns([3, 2])

    with col_result:
        source = year_data["source"]
        value_class = "forecast-value" if source == "forecast" else ""
        tag_class = "tag-forecast" if source == "forecast" else "tag-actual"
        tag_text = "Forecast" if source == "forecast" else "Actual data"
        data_type_note = f' <span style="color:#999;font-size:0.75rem;">({year_data["data_type"]})</span>'

        st.markdown(f"""
        <div class="result-card">
            <div class="label">Demand in {int(selected_year)} for {area_label}</div>
            <div class="value {value_class}">{format_demand(area_demand, du)}</div>
            <span class="source-tag {tag_class}">{tag_text}</span>{data_type_note}
        </div>
        """, unsafe_allow_html=True)

        # KPI row
        st.markdown("")
        kc1, kc2, kc3 = st.columns(3)
        with kc1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{population_label(pu)}</div>
                <div class="kpi-value">{format_population(year_data["population"], pu)}</div>
                <div class="kpi-label">{year_data["data_type"]}</div>
            </div>""", unsafe_allow_html=True)
        with kc2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Per Capita Demand</div>
                <div class="kpi-value">{year_data["per_capita"]*1000:.2f} W/person</div>
                <div class="kpi-label">{year_data["data_type"]}</div>
            </div>""", unsafe_allow_html=True)
        with kc3:
            total_formatted = format_demand(total_demand, du)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Coimbatore Demand</div>
                <div class="kpi-value">{total_formatted}</div>
                <div class="kpi-label">{year_data["data_type"]}</div>
            </div>""", unsafe_allow_html=True)

    # Cloud card
    with col_cloud:
        if selected_area == "All Coimbatore":
            step4 = ""
        else:
            step4 = f'<div class="cloud-step"><b>Step 4:</b> Area demand = {format_demand(total_demand, du)} Ã— {area_share:.0f}% = <b>{format_demand(area_demand, du)}</b></div>'

        method_line = (
            f'ðŸ“Œ Actual data from CSV'
            if source == "actual"
            else f'ðŸ“Œ Forecast using {year_data["model_name"]}, RÂ² = {year_data["r2_score"]:.4f}'
        )

        st.markdown(f"""
        <div class="cloud-card">
            <h4>â˜ï¸ How is it calculated?</h4>
            <div class="cloud-step"><b>Step 1:</b> Population ({int(selected_year)}) = {format_population(year_data["population"], pu)}</div>
            <div class="cloud-step"><b>Step 2:</b> Per capita demand = {year_data["per_capita"]*1000:.2f} W/person</div>
            <div class="cloud-step"><b>Step 3:</b> Coimbatore demand = Pop Ã— Per capita = <b>{format_demand(total_demand, du)}</b></div>
            {step4}
            <div style="border-top:1px solid #a8c8d8;margin:8px 0;"></div>
            <div class="cloud-step">{method_line}</div>
        </div>
        """, unsafe_allow_html=True)

    # Compare years section
    st.markdown("---")
    st.markdown("##### ðŸ”„ Compare Years")
    cc1, cc2 = st.columns(2)
    with cc1:
        compare_y1 = st.selectbox("Year 1", list(range(min_year, max_year + 1)), index=0, key="cy1")
    with cc2:
        default_y2 = min(2020, max_year)
        idx2 = default_y2 - min_year if default_y2 >= min_year else 0
        compare_y2 = st.selectbox("Year 2", list(range(min_year, max_year + 1)), index=min(idx2, max_year - min_year), key="cy2")

    cy1_data = get_value_for_year(int(compare_y1), data_df, forecast_result)
    cy2_data = get_value_for_year(int(compare_y2), data_df, forecast_result)
    cy1_dem, _ = get_demand_for_selection(cy1_data["demand_mw"], areas_df, selected_area)
    cy2_dem, _ = get_demand_for_selection(cy2_data["demand_mw"], areas_df, selected_area)

    comp1, comp2, comp3 = st.columns(3)
    dem_change = ((cy2_dem - cy1_dem) / cy1_dem * 100) if cy1_dem else 0
    pop_change = ((cy2_data["population"] - cy1_data["population"]) / cy1_data["population"] * 100) if cy1_data["population"] else 0

    with comp1:
        chg_class = "kpi-positive" if dem_change >= 0 else "kpi-negative"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{area_label} Demand</div>
            <div class="kpi-value">{format_demand(cy1_dem, du)} â†’ {format_demand(cy2_dem, du)}</div>
            <div class="kpi-change {chg_class}">{"ðŸ“ˆ" if dem_change >= 0 else "ðŸ“‰"} {dem_change:+.1f}%</div>
            <div class="kpi-label">{cy1_data["data_type"]} â†’ {cy2_data["data_type"]}</div>
        </div>""", unsafe_allow_html=True)
    with comp2:
        chg_class2 = "kpi-positive" if pop_change >= 0 else "kpi-negative"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Population</div>
            <div class="kpi-value">{format_population(cy1_data["population"], pu)} â†’ {format_population(cy2_data["population"], pu)}</div>
            <div class="kpi-change {chg_class2}">{"ðŸ“ˆ" if pop_change >= 0 else "ðŸ“‰"} {pop_change:+.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with comp3:
        pc_change = ((cy2_data["per_capita"] - cy1_data["per_capita"]) / cy1_data["per_capita"] * 100) if cy1_data["per_capita"] else 0
        chg_class3 = "kpi-positive" if pc_change >= 0 else "kpi-negative"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Per Capita Demand</div>
            <div class="kpi-value">{cy1_data["per_capita"]*1000:.2f} â†’ {cy2_data["per_capita"]*1000:.2f} W/person</div>
            <div class="kpi-change {chg_class3}">{"ðŸ“ˆ" if pc_change >= 0 else "ðŸ“‰"} {pc_change:+.1f}%</div>
        </div>""", unsafe_allow_html=True)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 2: CHARTS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab_charts:

    # --- 1. Line chart: Population & Demand over time (dual axis) ---
    st.markdown("##### ðŸ“ˆ Population & Demand Over Time")
    fig_line = go.Figure()

    actual_mask = combined_df["data_type"] != "forecast"
    forecast_mask = combined_df["data_type"] == "forecast"
    dummy_mask = combined_df["data_type"] == "dummy"

    # Population traces
    fig_line.add_trace(go.Scatter(
        x=combined_df.loc[actual_mask, "year"],
        y=combined_df.loc[actual_mask, "population"].apply(lambda v: convert_population(v, pu)),
        mode="lines+markers", name=population_label(pu),
        line=dict(color=SKY_BLUE, width=2),
        marker=dict(size=6),
        yaxis="y",
    ))
    if forecast_mask.any():
        # connect forecast to last actual point
        last_actual_yr = combined_df.loc[actual_mask, "year"].max()
        bridge = combined_df[combined_df["year"] >= last_actual_yr]
        fig_line.add_trace(go.Scatter(
            x=bridge["year"],
            y=bridge["population"].apply(lambda v: convert_population(v, pu)),
            mode="lines", name=f"{population_label(pu)} (forecast)",
            line=dict(color=SKY_BLUE, width=2, dash="dash"),
            yaxis="y", showlegend=True,
        ))

    # Demand traces
    fig_line.add_trace(go.Scatter(
        x=combined_df.loc[actual_mask, "year"],
        y=combined_df.loc[actual_mask, "demand_mw"].apply(lambda v: convert_demand(v, du)),
        mode="lines+markers", name=f"Demand ({du})",
        line=dict(color=OLIVE_GREEN, width=2),
        marker=dict(
            size=7,
            symbol=combined_df.loc[actual_mask, "data_type"].apply(
                lambda d: "diamond" if d == "dummy" else "circle"
            ),
        ),
        yaxis="y2",
    ))
    if forecast_mask.any():
        bridge_d = combined_df[combined_df["year"] >= last_actual_yr]
        fig_line.add_trace(go.Scatter(
            x=bridge_d["year"],
            y=bridge_d["demand_mw"].apply(lambda v: convert_demand(v, du)),
            mode="lines", name=f"Demand forecast ({du})",
            line=dict(color=OLIVE_GREEN, width=2, dash="dash"),
            yaxis="y2", showlegend=True,
        ))

    # Selected year vertical line
    fig_line.add_vline(x=int(selected_year), line_dash="dot", line_color="red",
                       annotation_text=str(int(selected_year)))

    fig_line.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis=dict(title=population_label(pu), side="left"),
        yaxis2=dict(title=f"Demand ({du})", side="right", overlaying="y"),
        height=420,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_line, width='stretch')
    st.caption("**Columns:** year, population, demand_mw | Dashed = forecast, Diamond markers = dummy data")

    # --- 2. Bar chart: Area-wise demand ---
    st.markdown("##### ðŸ˜ï¸ Area-wise Demand for Selected Year")
    area_demands = compute_all_area_demands(total_demand, areas_df)
    colors_bar = [OLIVE_GREEN if a == selected_area else SKY_BLUE for a in area_demands["area"]]

    fig_bar = go.Figure(go.Bar(
        x=area_demands["area"],
        y=area_demands["demand_mw"].apply(lambda v: convert_demand(v, du)),
        marker_color=colors_bar,
        text=area_demands["demand_mw"].apply(lambda v: f"{convert_demand(v, du):.1f}"),
        textposition="outside",
    ))
    fig_bar.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis_title=f"Demand ({du})",
        xaxis_title="Area",
        height=400,
    )
    st.plotly_chart(fig_bar, width='stretch')
    st.caption("**Columns:** area, demand_mw | Green = selected area")

    # --- Grouped bar: yearly growth % ---
    st.markdown("##### ðŸ“Š Yearly Growth % (Population & Demand)")
    growth_df = combined_df.copy()
    growth_df["pop_growth"] = growth_df["population"].pct_change() * 100
    growth_df["dem_growth"] = growth_df["demand_mw"].pct_change() * 100
    growth_df = growth_df.dropna(subset=["pop_growth"])

    fig_growth = go.Figure()
    fig_growth.add_trace(go.Bar(
        x=growth_df["year"], y=growth_df["pop_growth"],
        name="Population Growth %", marker_color=SKY_BLUE,
    ))
    fig_growth.add_trace(go.Bar(
        x=growth_df["year"], y=growth_df["dem_growth"],
        name="Demand Growth %", marker_color=OLIVE_GREEN,
    ))
    fig_growth.update_layout(
        template=PLOTLY_TEMPLATE, barmode="group", height=380,
        yaxis_title="Growth (%)", xaxis_title="Year",
    )
    st.plotly_chart(fig_growth, width='stretch')
    st.caption("**Columns:** year, pop_growth%, dem_growth%")

    # --- 3. Scatter: population vs demand ---
    st.markdown("##### ðŸ”µ Population vs Demand (Scatter)")
    fig_scatter = px.scatter(
        combined_df, x="population", y="demand_mw",
        color="year", hover_data=["year", "population", "demand_mw"],
        color_continuous_scale="Viridis",
        labels={"population": population_label(pu), "demand_mw": f"Demand ({du})"},
    )
    # Regression line
    from sklearn.linear_model import LinearRegression as LR
    _lr = LR()
    _lr.fit(combined_df[["population"]], combined_df["demand_mw"])
    x_range = np.linspace(combined_df["population"].min(), combined_df["population"].max(), 100)
    y_pred = _lr.predict(x_range.reshape(-1, 1))
    fig_scatter.add_trace(go.Scatter(
        x=x_range, y=y_pred, mode="lines", name="Regression line",
        line=dict(color="red", dash="dash"),
    ))
    # Highlight selected year
    sel_row = combined_df[combined_df["year"] == int(selected_year)]
    if not sel_row.empty:
        fig_scatter.add_trace(go.Scatter(
            x=sel_row["population"], y=sel_row["demand_mw"],
            mode="markers", name=f"Selected: {int(selected_year)}",
            marker=dict(size=16, color="red", symbol="star"),
        ))
    fig_scatter.update_layout(template=PLOTLY_TEMPLATE, height=420)
    st.plotly_chart(fig_scatter, width='stretch')
    st.caption("**Columns:** population, demand_mw | Colored by year, red star = selected year")

    # --- 4. Pie charts ---
    st.markdown("##### ðŸ¥§ Area-wise Share of Demand")
    pie_col1, pie_col2 = st.columns(2)

    with pie_col1:
        pull_vals = [0.1 if a == selected_area else 0 for a in area_demands["area"]]
        fig_pie = go.Figure(go.Pie(
            labels=area_demands["area"],
            values=area_demands["demand_mw"],
            pull=pull_vals,
            marker=dict(colors=AREA_COLORS),
            textinfo="label+percent",
        ))
        fig_pie.update_layout(template=PLOTLY_TEMPLATE, height=380,
                              title=f"Area Shares ({int(selected_year)})")
        st.plotly_chart(fig_pie, width='stretch')
        st.caption("**Columns:** area, demand_mw share")

    with pie_col2:
        # Growth split pie: population vs per-capita contribution
        if cy1_data["population"] > 0 and cy1_data["per_capita"] > 0:
            pop_contrib = pop_change
            pc_contrib = pc_change
            total_contrib = abs(pop_contrib) + abs(pc_contrib)
            if total_contrib > 0:
                fig_pie2 = go.Figure(go.Pie(
                    labels=["Population Growth", "Per Capita Growth"],
                    values=[abs(pop_contrib), abs(pc_contrib)],
                    marker=dict(colors=[SKY_BLUE, OLIVE_GREEN]),
                    textinfo="label+percent",
                ))
                fig_pie2.update_layout(
                    template=PLOTLY_TEMPLATE, height=380,
                    title=f"Growth Split: {compare_y1}â†’{compare_y2}",
                )
                st.plotly_chart(fig_pie2, width='stretch')
                st.caption(f"Pop growth: {pop_contrib:+.1f}%, Per-capita growth: {pc_contrib:+.1f}%")
            else:
                st.info("No growth difference between the two years.")
        else:
            st.info("Insufficient data for growth split.")

    # --- 5. Network diagrams ---
    st.markdown("##### ðŸ”— Network Diagrams")
    net_mode = st.radio("Network view", ["Cause-and-Effect", "Area Network"], horizontal=True)

    if net_mode == "Cause-and-Effect":
        G = nx.DiGraph()
        nodes = {
            "Population": year_data["population"],
            "Per Capita\nDemand": year_data["per_capita"] * 1000,
            "Total\nDemand": total_demand,
            "Forecast\nDemand": year_data["demand_mw"],
            "Required\nSupply": year_data["demand_mw"] * 1.1,
        }
        for n, v in nodes.items():
            G.add_node(n, value=v)
        edges = [
            ("Population", "Total\nDemand", f"{pop_change:+.1f}%"),
            ("Per Capita\nDemand", "Total\nDemand", f"{pc_change:+.1f}%"),
            ("Total\nDemand", "Forecast\nDemand", f"{dem_change:+.1f}%"),
            ("Forecast\nDemand", "Required\nSupply", "+10% margin"),
        ]
        for u, v, lbl in edges:
            G.add_edge(u, v, label=lbl)

        pos = {
            "Population": (0, 1),
            "Per Capita\nDemand": (0, -1),
            "Total\nDemand": (1, 0),
            "Forecast\nDemand": (2, 0),
            "Required\nSupply": (3, 0),
        }
        node_vals = [nodes[n] for n in G.nodes()]
        max_val = max(node_vals) if node_vals else 1
        node_sizes = [max(15, v / max_val * 50) for v in node_vals]

        edge_traces = []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines", line=dict(width=2, color="#888"),
                hoverinfo="none", showlegend=False,
            ))
            edge_traces.append(go.Scatter(
                x=[(x0+x1)/2], y=[(y0+y1)/2+0.15],
                mode="text", text=[d.get("label", "")],
                textfont=dict(size=10, color=OLIVE_GREEN),
                showlegend=False,
            ))

        node_trace = go.Scatter(
            x=[pos[n][0] for n in G.nodes()],
            y=[pos[n][1] for n in G.nodes()],
            mode="markers+text",
            marker=dict(size=node_sizes, color=SKY_BLUE, line=dict(width=2, color=BLACK)),
            text=list(G.nodes()),
            textposition="bottom center",
            textfont=dict(size=10),
            hovertext=[f"{n}: {nodes[n]:,.1f}" for n in G.nodes()],
            hoverinfo="text", showlegend=False,
        )
        fig_net = go.Figure(data=edge_traces + [node_trace])
        fig_net.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title="Cause-and-Effect Network",
        )
        st.plotly_chart(fig_net, width='stretch')
        st.caption("Node size = value magnitude; edge labels = growth %")

    else:  # Area Network
        G2 = nx.Graph()
        G2.add_node("Coimbatore", value=total_demand)
        area_dem_dict = {}
        for _, row in area_demands.iterrows():
            aname = row["area"]
            adem = row["demand_mw"]
            G2.add_node(aname, value=adem)
            G2.add_edge("Coimbatore", aname)
            area_dem_dict[aname] = adem

        # Star layout
        pos2 = {"Coimbatore": (0, 0)}
        n_areas = len(area_demands)
        for i, aname in enumerate(area_demands["area"]):
            angle = 2 * np.pi * i / n_areas
            pos2[aname] = (np.cos(angle) * 2, np.sin(angle) * 2)

        edge_traces2 = []
        for u, v in G2.edges():
            x0, y0 = pos2[u]
            x1, y1 = pos2[v]
            edge_traces2.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines", line=dict(width=1.5, color="#ccc"),
                hoverinfo="none", showlegend=False,
            ))

        node_names = list(G2.nodes())
        node_x = [pos2[n][0] for n in node_names]
        node_y = [pos2[n][1] for n in node_names]
        node_v = [G2.nodes[n]["value"] for n in node_names]
        max_nv = max(node_v) if node_v else 1
        node_sz = [max(20, v / max_nv * 55) for v in node_v]
        node_col = [OLIVE_GREEN if n == selected_area else (BLACK if n == "Coimbatore" else SKY_BLUE) for n in node_names]

        node_trace2 = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=node_sz, color=node_col, line=dict(width=1.5, color=BLACK)),
            text=node_names, textposition="top center",
            textfont=dict(size=9),
            hovertext=[f"{n}: {convert_demand(G2.nodes[n]['value'], du):.1f} {du}" for n in node_names],
            hoverinfo="text", showlegend=False,
        )
        fig_net2 = go.Figure(data=edge_traces2 + [node_trace2])
        fig_net2.update_layout(
            template=PLOTLY_TEMPLATE, height=450,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title=f"Area Network ({int(selected_year)})",
        )
        st.plotly_chart(fig_net2, width='stretch')
        st.caption("Node size = area demand; green = selected area")

    # --- 6. Per capita demand line chart ---
    st.markdown("##### ðŸ“‰ Per Capita Demand Over Time")
    fig_pc = go.Figure()
    fig_pc.add_trace(go.Scatter(
        x=combined_df.loc[actual_mask, "year"],
        y=combined_df.loc[actual_mask, "per_capita"] * 1000,
        mode="lines+markers", name="Per Capita (W/person)",
        line=dict(color=SKY_BLUE, width=2),
        marker=dict(
            size=6,
            symbol=combined_df.loc[actual_mask, "data_type"].apply(
                lambda d: "diamond" if d == "dummy" else "circle"
            ),
        ),
    ))
    if forecast_mask.any():
        bridge_pc = combined_df[combined_df["year"] >= last_actual_yr]
        fig_pc.add_trace(go.Scatter(
            x=bridge_pc["year"],
            y=bridge_pc["per_capita"] * 1000,
            mode="lines", name="Forecast",
            line=dict(color=OLIVE_GREEN, width=2, dash="dash"),
        ))
    fig_pc.add_vline(x=int(selected_year), line_dash="dot", line_color="red",
                     annotation_text=str(int(selected_year)))
    fig_pc.update_layout(
        template=PLOTLY_TEMPLATE, height=350,
        yaxis_title="Per Capita Demand (W/person)",
        xaxis_title="Year",
    )
    st.plotly_chart(fig_pc, width='stretch')
    st.caption("**Columns:** year, per_capita_demand | Diamond markers = dummy data")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 3: OLD DATA
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab_old:
    st.markdown("##### ðŸ“‹ Historical Data")

    display_df = data_df.copy()
    display_df["per_capita_w"] = (display_df["demand_mw"] / display_df["population"]) * 1000
    display_df["growth_pct"] = display_df["demand_mw"].pct_change() * 100

    # Apply unit conversions for display
    display_df[f"demand_{du}"] = display_df["demand_mw"].apply(lambda v: round(convert_demand(v, du), 2))
    display_df[f"population_{pu}"] = display_df["population"].apply(lambda v: round(convert_population(v, pu), 2))

    if selected_area != "All Coimbatore":
        share = get_area_share(areas_df, selected_area)
        display_df[f"area_demand_{du}"] = display_df[f"demand_{du}"] * share / 100
        st.info(f"Showing data for **{selected_area}** (share = {share}%)")

    st.dataframe(display_df, width='stretch', hide_index=True)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 4: FORECAST
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab_forecast:
    st.markdown("##### ðŸ”® Forecast Details")
    st.info("âš ï¸ Simple regression on limited yearly data. The forecast is indicative only. No deep learning (small dataset).")

    # Model comparison
    st.markdown("**Population Models**")
    pop_rows = []
    for m in forecast_result.all_pop_models:
        pop_rows.append({
            "Model": m.name,
            "RÂ² (full data)": f"{m.r2_full:.4f}",
            "RÂ² (back-test, last 3 yrs)": f"{m.r2_backtest:.4f}",
            "Selected": "âœ…" if m.name == forecast_result.best_pop_model.name else "",
        })
    st.table(pd.DataFrame(pop_rows))

    st.markdown("**Demand Models**")
    dem_rows = []
    for m in forecast_result.all_demand_models:
        dem_rows.append({
            "Model": m.name,
            "RÂ² (full data)": f"{m.r2_full:.4f}",
            "RÂ² (back-test, last 3 yrs)": f"{m.r2_backtest:.4f}",
            "Selected": "âœ…" if m.name == forecast_result.best_demand_model.name else "",
        })
    st.table(pd.DataFrame(dem_rows))

    # Forecast table: Method 1 vs Method 2
    st.markdown("---")
    st.markdown("##### ðŸ“Š Forecast: Method 1 (Direct) vs Method 2 (Pop Ã— Per Capita)")

    fc1 = forecast_result.forecast_df.copy()
    fc2 = forecast_result.method2_df.copy()

    if not fc1.empty:
        comparison = pd.DataFrame({
            "Year": fc1["year"],
            f"M1 Population": fc1["population"].apply(lambda v: format_population(v, pu)),
            f"M1 Demand ({du})": fc1["demand_mw"].apply(lambda v: f"{convert_demand(v, du):.2f}"),
            f"M1 Lower ({du})": fc1["demand_lower"].apply(lambda v: f"{convert_demand(v, du):.2f}"),
            f"M1 Upper ({du})": fc1["demand_upper"].apply(lambda v: f"{convert_demand(v, du):.2f}"),
            f"M2 Demand ({du})": fc2["demand_mw"].apply(lambda v: f"{convert_demand(v, du):.2f}"),
        })
        st.dataframe(comparison, width='stretch', hide_index=True)
    else:
        st.info("No future years to forecast.")

    # Forecast chart
    if not fc1.empty:
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=fc1["year"], y=fc1["demand_mw"].apply(lambda v: convert_demand(v, du)),
            mode="lines+markers", name="Method 1 (Direct)",
            line=dict(color=SKY_BLUE, width=2),
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc2["year"], y=fc2["demand_mw"].apply(lambda v: convert_demand(v, du)),
            mode="lines+markers", name="Method 2 (Pop Ã— Per Capita)",
            line=dict(color=OLIVE_GREEN, width=2),
        ))
        # Confidence band for Method 1
        fig_fc.add_trace(go.Scatter(
            x=list(fc1["year"]) + list(fc1["year"][::-1]),
            y=list(fc1["demand_upper"].apply(lambda v: convert_demand(v, du))) +
              list(fc1["demand_lower"].apply(lambda v: convert_demand(v, du)))[::-1],
            fill="toself", fillcolor=f"rgba(135,206,235,0.15)",
            line=dict(width=0), name="Â±5% range", showlegend=True,
        ))
        fig_fc.update_layout(
            template=PLOTLY_TEMPLATE, height=400,
            yaxis_title=f"Demand ({du})", xaxis_title="Year",
            title="Forecast Comparison",
        )
        st.plotly_chart(fig_fc, width='stretch')
        st.caption("**Columns:** year, method1_demand, method2_demand, confidence_band")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 5: AI ASSISTANT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab_ai:
    st.markdown("##### ðŸ¤– AI Assistant (Powered by Gemini)")
    st.caption(f"Model: {GEMINI_MODEL}")

    # AI Report
    st.markdown("**ðŸ“ AI Summary Report**")
    if st.button("Generate AI Report", key="gen_report"):
        with st.spinner("Generating report with Gemini..."):
            report = generate_ai_report(
                year=int(selected_year),
                area=area_label,
                demand_mw=area_demand,
                source=year_data["source"],
                population=year_data["population"],
                per_capita=year_data["per_capita"],
                compare_year1=int(compare_y1),
                compare_year1_demand=cy1_dem,
                compare_year2=int(compare_y2),
                compare_year2_demand=cy2_dem,
                is_dummy=is_dummy,
                model_name=year_data.get("model_name", ""),
                r2_score_val=year_data.get("r2_score"),
            )
        st.markdown(f"""<div style="background:{LIGHT_BLUE};padding:16px;border-radius:12px;
                    border:1px solid {SKY_BLUE};color:{BLACK};">{report}</div>""",
                    unsafe_allow_html=True)

    st.markdown("---")

    # Ask a question
    st.markdown("**ðŸ’¬ Ask a Question**")
    st.caption('Examples: "What is the demand of Peelamedu in 2032?" or "What if population grows 1% faster?"')
    user_q = st.text_input("Your question:", key="user_question")
    if user_q and st.button("Ask Gemini", key="ask_btn"):
        data_summary = (
            f"Available areas: {get_area_list(areas_df)}. "
            f"Data years: {int(data_df['year'].min())}-{int(data_df['year'].max())}. "
            f"Forecast up to {FORECAST_END_YEAR}. "
            f"Current selection: year={int(selected_year)}, area={selected_area}."
        )
        with st.spinner("Asking Gemini..."):
            result = ask_question(user_q, data_summary)

        if "error" in result:
            st.error(f"âš ï¸ {result['error']}")
        else:
            st.success(f"Gemini parsed: Year={result['year']}, Area={result['area']}, "
                       f"Pop growth change={result['population_growth_change_pct']}%, "
                       f"Per-capita change={result['per_capita_change_pct']}%")

            # Calculate with the AI parameters
            ai_year_data = get_value_for_year(result["year"], data_df, forecast_result)
            adj_pop = ai_year_data["population"] * (1 + result["population_growth_change_pct"] / 100)
            adj_pc = ai_year_data["per_capita"] * (1 + result["per_capita_change_pct"] / 100)
            adj_demand = adj_pop * adj_pc
            ai_area_dem, ai_area_lbl = get_demand_for_selection(adj_demand, areas_df, result["area"])

            st.markdown(f"""
            <div class="result-card">
                <div class="label">AI-Adjusted Demand in {result["year"]} for {ai_area_lbl}</div>
                <div class="value forecast-value">{format_demand(ai_area_dem, du)}</div>
                <span class="source-tag tag-forecast">AI What-if</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Adjusted population: {format_population(adj_pop, pu)} | "
                       f"Adjusted per-capita: {adj_pc*1000:.2f} W/person | "
                       f"Source: {ai_year_data['data_type']}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 6: DATA
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab_data_tab:
    st.markdown("##### ðŸ’¾ Data Management")

    st.markdown("**Current Data (coimbatore_data.csv)**")
    st.dataframe(data_df, width='stretch', hide_index=True)

    st.markdown("**Current Areas (coimbatore_areas.csv)**")
    st.dataframe(areas_df, width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("**â¬‡ï¸ Download Results**")

    # Prepare download data
    dl_data = {
        "Year": [int(selected_year)],
        "Area": [area_label],
        f"Demand ({du})": [round(convert_demand(area_demand, du), 2)],
        "Source": [year_data["source"]],
        "Data Type": [year_data["data_type"]],
        "Population": [round(year_data["population"], 0)],
        "Per Capita (W/person)": [round(year_data["per_capita"] * 1000, 2)],
    }
    dl_df = pd.DataFrame(dl_data)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv_str = dl_df.to_csv(index=False)
        st.download_button("ðŸ“¥ Download as CSV", csv_str, "result.csv", "text/csv")
    with dl_col2:
        buf = io.BytesIO()
        dl_df.to_excel(buf, index=False, engine="openpyxl")
        st.download_button("ðŸ“¥ Download as Excel", buf.getvalue(), "result.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Full forecast download
    if not forecast_result.forecast_df.empty:
        st.markdown("**â¬‡ï¸ Download Full Forecast**")
        full_dl = get_combined_df(data_df, forecast_result)
        full_csv = full_dl.to_csv(index=False)
        st.download_button("ðŸ“¥ Download full forecast CSV", full_csv,
                           "coimbatore_forecast_full.csv", "text/csv")

    # Template downloads
    st.markdown("---")
    st.markdown("**ðŸ“„ Download CSV Templates**")
    t1, t2 = st.columns(2)
    with t1:
        st.download_button("ðŸ“¥ Data template", generate_data_template(),
                           "coimbatore_data_template.csv", "text/csv")
    with t2:
        st.download_button("ðŸ“¥ Areas template", generate_areas_template(),
                           "coimbatore_areas_template.csv", "text/csv")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Footer
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("""
<div class="app-footer">
    <p>âš¡ <b>Coimbatore Demand Forecast</b> Â· Built with
    <a href="https://streamlit.io" target="_blank">Streamlit</a>,
    <a href="https://plotly.com" target="_blank">Plotly</a> &
    <a href="https://ai.google.dev" target="_blank">Gemini AI</a></p>
    <p>ðŸ“Š ML-powered electricity demand forecasting Â· ðŸ™ï¸ 10 Coimbatore areas Â· ðŸ¤– AI insights</p>
</div>
""", unsafe_allow_html=True)
