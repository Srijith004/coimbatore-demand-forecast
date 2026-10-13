# ⚡ Coimbatore Demand Forecast

> **AI-Powered Electricity Demand Forecasting for Coimbatore**
>
> A Streamlit web application that visualises historical electricity demand
> and forecasts future demand for Coimbatore and its 10 key areas using
> machine-learning regression models and Google Gemini AI.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    subgraph Data Layer
        A["📄 coimbatore_data.csv<br/>(year, population, demand, type)"]
        B["📄 coimbatore_areas.csv<br/>(area, share_percent)"]
    end

    subgraph Processing Layer
        C["📦 data_loader.py<br/>Load · Validate · Upload"]
        D["📦 forecast.py<br/>Linear & Poly Regression<br/>Back-test · R² Score"]
        E["📦 areas.py<br/>Area Demand = Total × Share%"]
        F["📦 units.py<br/>kW/MW/GW/MU/TWh<br/>Number/Lakhs/Millions"]
        G["📦 ai_assistant.py<br/>Gemini API · Reports · Q&A"]
    end

    subgraph Presentation Layer
        H["🖥️ app.py — Streamlit App"]
        I["📊 Plotly Charts"]
        J["🧠 AI Assistant"]
        K["📥 CSV / Excel Export"]
    end

    A --> C
    B --> C
    C --> D
    C --> E
    D --> H
    E --> H
    F --> H
    G --> J
    H --> I
    H --> J
    H --> K

    style A fill:#87CEEB,stroke:#333,color:#111
    style B fill:#87CEEB,stroke:#333,color:#111
    style H fill:#6B8E23,stroke:#333,color:#FFF
    style I fill:#D6EEF9,stroke:#333,color:#111
    style J fill:#D6EEF9,stroke:#333,color:#111
    style K fill:#D6EEF9,stroke:#333,color:#111
```

---

## 🔮 Forecast Pipeline

The app uses **two independent forecasting methods** and compares them:

```mermaid
flowchart LR
    A["Historical Data<br/>2001 – 2024"] --> B["Train / Test Split<br/>(last 3 years held out)"]

    B --> C["Method 1: Direct Forecast<br/>Predict demand directly<br/>from year"]
    B --> D["Method 2: Component Forecast<br/>Predict population & per-capita<br/>separately, then multiply"]

    C --> E{"Model Selection<br/>Linear vs Polynomial(deg=2)<br/>Pick higher R²"}
    D --> F{"Model Selection<br/>Linear vs Polynomial(deg=2)<br/>Pick higher R²"}

    E --> G["Forecast demand<br/>for selected year"]
    F --> H["Forecast population ×<br/>per-capita demand"]

    G --> I["Compare M1 vs M2<br/>± 5% confidence band"]
    H --> I

    style A fill:#87CEEB,stroke:#333,color:#111
    style E fill:#6B8E23,stroke:#333,color:#FFF
    style F fill:#6B8E23,stroke:#333,color:#FFF
    style I fill:#D6EEF9,stroke:#333,color:#111
```

**Back-testing:** Models are trained on all data except the last 3 years,
then tested on those years. The model (Linear or Polynomial degree 2) with
the higher R² score is automatically selected.

---

## 🤖 AI Assistant Flow

```mermaid
flowchart TD
    A["User enters question<br/>e.g. 'What is Peelamedu<br/>demand in 2032?'"] --> B["Gemini API<br/>Structured JSON Output"]
    B --> C{"Parse response:<br/>year, area,<br/>pop_growth_change,<br/>per_capita_change"}
    C --> D["Fetch base forecast<br/>for parsed year"]
    D --> E["Apply adjustments<br/>Pop × (1 + change%)<br/>PerCapita × (1 + change%)"]
    E --> F["Calculate adjusted<br/>area demand"]
    F --> G["Display result card<br/>with breakdown"]

    H["User clicks<br/>'Generate AI Report'"] --> I["Gemini API<br/>≤ 120 word summary"]
    I --> J["Display formatted<br/>report card"]

    style A fill:#87CEEB,stroke:#333,color:#111
    style B fill:#6B8E23,stroke:#333,color:#FFF
    style H fill:#87CEEB,stroke:#333,color:#111
    style I fill:#6B8E23,stroke:#333,color:#FFF
```

---

## ✨ Features

### 📊 Six Interactive Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | Result card showing demand for the selected year/area, cloud calculation card with step-by-step breakdown, and year-over-year KPI comparisons |
| **Charts** | Dual-axis line chart (population + demand), area bar chart, scatter plot (pop vs demand), pie charts (area shares & growth split), network diagrams (cause-effect & area connections), per-capita trend line |
| **Old Data** | Full historical data table with year-over-year growth percentages |
| **Forecast** | Model comparison table (Linear vs Polynomial with R² scores), Method 1 vs Method 2 forecast comparison, confidence band visualization |
| **AI Assistant** | Gemini-powered summary reports (≤120 words) and structured what-if Q&A with parameter adjustments |
| **Data** | View/upload CSV data, download results as CSV or Excel, download blank templates |

### 🔧 Key Capabilities

- **Smart Unit Selector** — Switch between kW / MW / GW / MU / TWh for demand and Number / Lakhs / Millions for population. All backend calculations stay in base units (MW, raw count).
- **10 Coimbatore Areas** — Gandhipuram, RS Puram, Peelamedu, Saibaba Colony, Singanallur, Ukkadam, Saravanampatty, Ganapathy, Vadavalli, Sulur — each with configurable demand share percentages summing to 100%.
- **Dummy Data Banner** — Automatically displays a warning banner when viewing data marked as `dummy` in the CSV.
- **CSV Upload** — Upload your own data/areas CSVs (validated on upload).
- **Export** — Download results and full forecasts as CSV or Excel.
- **20 Automated Tests** — pytest suite covering area shares, demand sums, R² sanity, unit round-trips, CSV validation, and more.

---

## 🗂️ Project Structure

```
coimbatore-demand-forecast/
├── app.py                  # Main Streamlit app (6 tabs, sidebar, charts)
├── config.py               # Colour palette, Gemini model, Plotly template
├── data_loader.py          # CSV loading, validation, templates, uploads
├── forecast.py             # sklearn Linear & Poly regression, back-testing
├── areas.py                # Area demand = total × share%
├── units.py                # Demand & population unit converters
├── ai_assistant.py         # Gemini API integration (reports & Q&A)
├── tests.py                # 20 pytest test cases
├── requirements.txt        # Python dependencies
├── .gitignore
├── .streamlit/
│   └── secrets.toml        # Gemini API key (git-ignored)
└── data/
    ├── coimbatore_data.csv  # Historical population & demand data
    └── coimbatore_areas.csv # 10 areas with share percentages
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Srijith004/coimbatore-demand-forecast.git
cd coimbatore-demand-forecast
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

Or use an environment variable:

```bash
# Windows
set GEMINI_API_KEY=your-key

# Linux / macOS
export GEMINI_API_KEY=your-key
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 🧪 Running Tests

```bash
pytest tests.py -v
```

**20 tests** covering:

| Category | Tests |
|----------|-------|
| Area shares | Sum to 100%, individual share validation |
| Demand calculation | Area demand sums match total demand |
| Forecasting | R² score sanity checks, model selection logic |
| Unit conversion | Round-trip accuracy for all demand/population units |
| CSV validation | Schema validation, required columns, data types |
| Data integrity | Year range checks, population/demand positivity |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core language |
| **Streamlit** | Web application framework |
| **Plotly** | Interactive charts and visualisations |
| **pandas** | Data manipulation and analysis |
| **scikit-learn** | Linear & Polynomial regression models |
| **NetworkX** | Network/graph diagrams (cause-effect, area connections) |
| **Google Gemini API** | AI-powered reports and structured Q&A |
| **openpyxl** | Excel export support |
| **pytest** | Automated testing |

---

## 📈 Data Format

### `data/coimbatore_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Year (e.g. 2001, 2024) |
| `population` | float | Population count |
| `demand_mw` | float | Electricity demand in MW |
| `data_type` | string | `"dummy"` or `"official"` |

### `data/coimbatore_areas.csv`

| Column | Type | Description |
|--------|------|-------------|
| `area` | string | Area name (e.g. Gandhipuram) |
| `area_type` | string | Area classification (e.g. commercial, residential) |
| `share_percent` | float | Percentage share of total demand |
| `data_type` | string | `"dummy"` or `"official"` |

> **Note:** `share_percent` values across all areas must sum to **100.0**.

---

## ⚠️ Disclaimer

This application uses **simple regression models** trained on limited yearly
data. The forecasts are **indicative only** and should not be used for
official planning or policy decisions without validation against
authoritative sources.

---

## 📝 License

This project is developed as an academic assignment. Feel free to use and
modify for educational purposes.

---

<p align="center">
  Built with ❤️ using Streamlit & Gemini AI
</p>
