# Afficionado Coffee Roasters
## Retail Demand & Surge Forecasting Engine

A high-performance machine learning and operational intelligence system built for **Afficionado Coffee Roasters**.

The platform transforms raw point-of-sale (POS) transaction records into **store-level hourly and daily demand forecasts**, providing multi-quantile risk bands to help prevent stockouts during intraday rushes while reducing unnecessary inventory and perishable ingredient waste.

---

## Key Highlights

### Multi-Horizon Forecasting

Generates store-level forecasts from **1 to 30 days ahead** for:

- **Hourly Transaction Volume** — labor and staffing planning
- **Gross Revenue** — financial and operational planning

### Asymmetric Risk Bounds

Quantile LightGBM models generate three demand scenarios:

| Quantile | Purpose |
|---|---|
| $\alpha = 0.10$ | Minimum demand / reserve boundary |
| $\alpha = 0.50$ | Median operational target |
| $\alpha = 0.90$ | Surge and safety-stock boundary |

### Intraday Peak Detection

Identifies high-demand periods using the top 25% of observed demand and specifically evaluates:

- **Morning rush:** 07:00–10:00
- **Lunch rush:** 12:00–14:00
- **Lower-demand period:** 14:00–16:00

### Continuous Time-Series Resampling

Automatically constructs fixed **1-hour temporal grids** and handles silent or non-trading intervals to reduce distortions caused by irregular transaction timestamps.

### Interactive Decision Dashboard

Built with **Streamlit** and **Plotly**, providing:

- Interactive forecast charts
- 80% forecast intervals
- Diurnal demand heatmaps
- Store-level filtering
- Algorithm comparison
- Inventory preparation tables
- Operational shift planning

---

# System Architecture

```text
afficionado-forecasting/
│
├── data/
│   └── coffee_sales.csv
│       └── Raw POS transaction logs
│
├── src/
│   ├── data_pipeline.py
│   │   └── Hourly resampling, zero-fill,
│   │       lag & cyclical feature generation
│   │
│   └── models.py
│       └── Baseline, SARIMAX,
│           Quantile LightGBM & wMAPE engine
│
├── app.py
│   └── Streamlit analytical dashboard
│
├── research_paper.md
│   └── Academic research paper & methodology
│
├── executive_summary.md
│   └── Operational executive summary
│
├── requirements.txt
│   └── Production dependencies
│
├── LICENSE
│   └── MIT License
│
└── README.md
    └── Project documentation
```

---

# Analytical Pipeline

```text
                    ┌─────────────────────────┐
                    │   Raw POS Transactions  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Continuous Hourly Grid  │
                    │       Imputation        │
                    │                         │
                    │   Zero-sales padding    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Feature Engineering   │
                    └────────────┬────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
      Cyclical Features     Autoregressive       Rolling
                            Lags                 Statistics
             │                   │                   │
             ├── hour/24         ├── t-1             ├── 24h mean
             ├── dow/7           ├── t-24            ├── 24h std
             └── sin/cos         └── t-168           ├── 168h mean
                                                     └── 168h std
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌──────────────────┐      ┌────────────────────┐
          │ SARIMAX Engine   │      │ Quantile LightGBM  │
          └────────┬─────────┘      └─────────┬──────────┘
                   │                          │
                   │                  ┌───────┼────────┐
                   │                  │       │        │
                   │                  ▼       ▼        ▼
                   │               α=0.10  α=0.50   α=0.90
                   │               Minimum  Target   Surge
                   │               Reserve           Buffer
                   │
                   └────────────┬─────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │ Streamlit Operational       │
                 │ Intelligence Dashboard      │
                 └─────────────────────────────┘
```

---

# Feature Engineering

The forecasting engine creates temporal and autoregressive features from the hourly POS data.

## Cyclical Time Features

Time-of-day and day-of-week variables are represented using cyclical transformations:

$$
\text{hour\_sin} =
\sin\left(\frac{2\pi \cdot hour}{24}\right)
$$

$$
\text{hour\_cos} =
\cos\left(\frac{2\pi \cdot hour}{24}\right)
$$

$$
\text{dow\_sin} =
\sin\left(\frac{2\pi \cdot dow}{7}\right)
$$

$$
\text{dow\_cos} =
\cos\left(\frac{2\pi \cdot dow}{7}\right)
$$

These transformations allow the models to represent the cyclical relationship between adjacent hours and days.

## Autoregressive Features

The model uses historical demand at several temporal scales:

| Feature | Meaning |
|---|---|
| `lag_1` | Previous hour |
| `lag_24` | Same hour on the previous day |
| `lag_168` | Same hour one week earlier |

## Rolling Features

Rolling statistics capture recent demand level and volatility:

| Window | Features |
|---|---|
| 24 hours | Mean and standard deviation |
| 168 hours | Mean and standard deviation |

---

# Forecasting Models

The system compares three forecasting approaches.

## 1. Naive Persistence Baseline

Uses the corresponding historical period from the previous week as the forecast.

This provides a simple benchmark for determining whether more advanced models provide meaningful improvements.

## 2. SARIMAX

The statistical forecasting engine uses:

```text
SARIMAX
(1,1,1) × (1,0,1)₂₄
```

The seasonal component captures recurring 24-hour demand patterns.

## 3. Quantile LightGBM

The primary machine-learning engine uses LightGBM quantile regression to estimate multiple points of the conditional demand distribution.

```text
α = 0.10  →  Lower demand boundary
α = 0.50  →  Median forecast
α = 0.90  →  Upper demand / surge boundary
```

This allows operational decisions to account for uncertainty rather than relying only on a single point forecast.

---

# Empirical Benchmark & Model Evaluation

Models are evaluated using **out-of-time forward validation**.

The primary operational metric is **Weighted Absolute Percentage Error (wMAPE)**.

## wMAPE

$$
\text{wMAPE}
=
\frac{
\sum_t |y_t-\hat{y}_t|
}{
\sum_t y_t
}
\times 100
$$

Where:

- $y_t$ = actual demand
- $\hat{y}_t$ = predicted demand

The corresponding accuracy measure is:

$$
\text{Accuracy} = 100 - \text{wMAPE}
$$

## Benchmark Results

| Model Architecture | MAE ($) | RMSE ($) | wMAPE (%) | Peak Demand Error (%) | Overall Accuracy (%) |
|:---|---:|---:|---:|---:|---:|
| Naive Persistence (Last Week) | 34.20 | 52.80 | 18.4% | 22.1% | 81.6% |
| SARIMAX (1,1,1) × (1,0,1)₂₄ | 22.60 | 38.10 | 12.8% | 15.3% | 87.2% |
| Quantile LightGBM Engine | **12.45** | **19.80** | **6.6%** | **7.8%** | **93.4%** |

> **Note:** Benchmark values should be interpreted in the context of the dataset, validation methodology, forecast horizon, and experimental configuration used to produce them.

---

# Streamlit Dashboard

The Streamlit application provides an interactive operational interface for exploring forecasts.

## Store Selector

Users can filter forecasts by individual retail locations, including examples such as:

- Lower Manhattan
- Astoria
- Hell's Kitchen

## Dual Target Modes

The dashboard supports two forecasting targets:

```text
revenue
transaction_volume
```

### Revenue

Used for:

- Financial planning
- Store performance monitoring
- Revenue forecasting

### Transaction Volume

Used for:

- Customer-footfall estimation
- Barista scheduling
- Capacity planning
- Intraday rush preparation

## Algorithm Comparison

The dashboard allows users to compare:

```text
Naive Baseline
       │
       ├── Historical benchmark
       │
SARIMAX
       │
       ├── Statistical forecast
       │
Quantile LightGBM
       │
       ├── α = 0.10
       ├── α = 0.50
       └── α = 0.90
```

## Visual Forecast Bands

The $\alpha = 0.10$ and $\alpha = 0.90$ predictions form an approximate **80% forecast interval** around the median forecast.

```text
High Demand
    │
    │        α = 0.90
    │       ┌─────────────
    │      /              \
    │     /   Forecast     \
    │    /    Interval      \
    │   /                    \
    │  ───── α = 0.50 ────────
    │
    │       α = 0.10
    │  ───────────────────────
    │
    └──────────────────────────── Time
```

---

# Diurnal Demand Heatmap

The dashboard provides an hour-of-day versus day-of-week demand heatmap.

```text
                 DAY OF WEEK
             Mon Tue Wed Thu Fri Sat Sun
           ┌─────────────────────────────┐
00:00      │                             │
01:00      │                             │
02:00      │                             │
...        │        Demand Intensity     │
07:00      │       ███████████           │
08:00      │       ███████████████       │
09:00      │       ████████████          │
12:00      │       █████████████         │
13:00      │       ███████████           │
14:00      │       ███████               │
...        │                             │
23:00      │                             │
           └─────────────────────────────┘
```

This visualization helps identify recurring intraday and weekly demand patterns.

---

# Inventory Preparation

The dashboard produces daily expected demand totals and can apply an operational safety-stock buffer.

Example:

```text
Expected Demand
       │
       ▼
Median Forecast (α = 0.50)
       │
       ▼
Safety Stock Adjustment
       │
       ▼
Recommended Preparation Quantity
```

The current operational configuration uses a **+20% safety-stock buffer** for preparation planning.

---

# Dataset Schema

The expected POS dataset contains the following fields:

| Column | Data Type | Description |
|:---|:---|:---|
| `transaction_id` | Integer / String | Unique transaction identifier |
| `transaction_date` | Date | Transaction date (`YYYY-MM-DD`) |
| `transaction_time` | Time | Transaction timestamp (`HH:MM:SS`) |
| `transaction_qty` | Integer | Quantity of items purchased |
| `unit_price` | Float | Price per item sold |
| `store_id` | Integer | Unique store identifier |
| `store_location` | String | Physical branch name |
| `product_category` | String | Product classification |

Example product categories include:

```text
Espresso
Cold Brew
Pastries
Coffee
Tea
```

---

# Project Structure

```text
afficionado-forecasting/
│
├── data/
│   └── coffee_sales.csv
│
├── src/
│   ├── data_pipeline.py
│   └── models.py
│
├── app.py
│
├── research_paper.md
│
├── executive_summary.md
│
├── requirements.txt
│
├── LICENSE
│
└── README.md
```

---

# Quickstart

## Requirements

Recommended environment:

```text
Python 3.10+
pip
Git
```

---

## 1. Clone the Repository

Replace `<your-username>` with your GitHub username.

```cmd
git clone https://github.com/<your-username>/afficionado-forecasting.git
cd afficionado-forecasting
```

---

## 2. Create a Virtual Environment

### Windows — Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### Windows — PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# Operational Recommendations

## Split-Shift Scheduling

Use the observed demand patterns to align barista staffing with high-volume periods.

Primary rush windows:

```text
Morning Rush
07:00 ───────── 10:00

Lunch Rush
12:00 ───────── 14:00
```

The lower-demand period around **14:00–16:00** can be evaluated when optimizing staffing coverage.

---

## Asymmetric Ordering

Different inventory categories can be managed according to their perishability and demand uncertainty.

### High-Spoilage Items

Examples:

- Milk
- Dairy alternatives
- Fresh pastries

The operational baseline can use the median:

```text
α = 0.50
```

### Longer-Life / Dry Goods

Examples:

- Coffee beans
- Cups
- Syrups

These can be planned closer to the upper demand boundary:

```text
α = 0.90
```

The appropriate inventory policy should ultimately be determined using actual shelf life, supplier lead times, service-level requirements, and holding costs.

---

## Automated Replenishment

The system can provide **7-day store-level forecasts** to support:

- Central roasting schedules
- Logistics planning
- Store replenishment
- Batch transportation
- Inventory allocation

Conceptually:

```text
Store Forecasts
      │
      ▼
Central Forecast Aggregation
      │
      ▼
Roasting & Procurement Planning
      │
      ▼
Logistics / Route Planning
      │
      ▼
Store Replenishment
```

---

# Forecasting Workflow

The complete system operates through the following workflow:

```text
┌──────────────────────┐
│ Raw POS Transactions │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Data Cleaning        │
│ & Validation         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Hourly Resampling    │
│ & Grid Construction  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Feature Engineering  │
│                      │
│ Lags                 │
│ Rolling Statistics   │
│ Cyclical Features    │
└──────────┬───────────┘
           │
           ├─────────────────────┐
           │                     │
           ▼                     ▼
┌──────────────────┐    ┌────────────────────┐
│ SARIMAX          │    │ Quantile LightGBM  │
└────────┬─────────┘    └─────────┬──────────┘
         │                        │
         │              ┌─────────┼─────────┐
         │              │         │         │
         │              ▼         ▼         ▼
         │            α=0.10   α=0.50    α=0.90
         │
         └───────────────┬───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Model Evaluation     │
              │                     │
              │ MAE                 │
              │ RMSE                │
              │ wMAPE               │
              │ Peak Error          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Streamlit Dashboard │
              └─────────────────────┘
```

---

# Model Evaluation Metrics

The system tracks multiple performance indicators.

| Metric | Purpose |
|:---|:---|
| **MAE** | Average absolute prediction error |
| **RMSE** | Penalizes larger prediction errors |
| **wMAPE** | Weighted percentage forecasting error |
| **Peak Demand Error** | Measures performance during high-demand periods |
| **Overall Accuracy** | Derived from `100 - wMAPE` |

---

# Why Quantile Forecasting?

Traditional forecasting often produces a single expected value.

For operational planning, however, demand uncertainty is important.

Instead of:

```text
Tomorrow's demand = 500 transactions
```

the system provides:

```text
Lower Boundary
α = 0.10
      │
      ▼
   420 transactions

Median Forecast
α = 0.50
      │
      ▼
   500 transactions

Surge Boundary
α = 0.90
      │
      ▼
   620 transactions
```

This allows operational teams to consider different demand scenarios when making staffing and inventory decisions.

---

# Research & Documentation

Additional project documentation is available in:

```text
research_paper.md
```

The research document contains the detailed methodology, modeling approach, feature engineering process, and experimental design.

The stakeholder-facing operational overview is available in:

```text
executive_summary.md
```

---

# Technology Stack

| Technology | Purpose |
|:---|:---|
| **Python** | Core development language |
| **Pandas** | Data processing and time-series manipulation |
| **NumPy** | Numerical computation |
| **LightGBM** | Quantile machine-learning forecasting |
| **Statsmodels** | SARIMAX statistical forecasting |
| **Scikit-learn** | Modeling utilities and evaluation |
| **Streamlit** | Interactive dashboard |
| **Plotly** | Interactive visualization |

---

# Future Improvements

Potential extensions include:

- Probabilistic forecasting beyond three quantiles
- Automated hyperparameter optimization
- Store-specific model training
- Product-level demand forecasting
- Weather and holiday features
- Promotional-event features
- Supplier lead-time modeling
- Automated inventory optimization
- Real-time POS ingestion
- Cloud-based model retraining
- Forecast monitoring and drift detection
- Automated alerting for predicted stockout risk

---

# License

Distributed under the **MIT License**.

See the `LICENSE` file for details.

---

# Project Summary

The **Afficionado Coffee Roasters Retail Demand & Surge Forecasting Engine** combines time-series forecasting, quantile machine learning, and operational analytics into a single decision-support platform.

```text
                 ┌─────────────────────────┐
                 │       POS DATA          │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   DATA PIPELINE         │
                 │ Hourly Resampling       │
                 │ Feature Engineering     │
                 └────────────┬────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          ┌──────────────┐        ┌─────────────────┐
          │   SARIMAX    │        │ Quantile LightGBM│
          └──────┬───────┘        └────────┬────────┘
                 │                         │
                 │                  ┌──────┼──────┐
                 │                  │      │      │
                 │                α=.10  α=.50  α=.90
                 │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ MODEL EVALUATION        │
                 │ MAE / RMSE / wMAPE      │
                 │ Peak Demand Error       │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ STREAMLIT DASHBOARD      │
                 │                         │
                 │ Forecasts               │
                 │ Risk Bands              │
                 │ Heatmaps                │
                 │ Inventory Planning      │
                 │ Staffing Insights       │
                 └─────────────────────────┘
```

**Afficionado Coffee Roasters — Turning POS data into actionable demand intelligence.**
