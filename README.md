Afficionado Coffee Roasters | Retail Demand & Surge Forecasting Engine

A high-performance machine learning and operational intelligence system built for Afficionado Coffee Roasters. The platform transforms raw point-of-sale (POS) transaction records into store-level hourly and daily forecasts, delivering multi-quantile risk bands (10th, 50th, and 90th percentiles) to prevent stockouts during intraday rushes and reduce perishable ingredient waste.

Key Highlights

Multi-Horizon Forecasting: Generates store-level forecasts from 1 to 30 days ahead for both Hourly Transaction Volume (labor scheduling) and Gross Revenue (financial planning).

Asymmetric Risk Bounds: Quantile LightGBM models predict:

$\alpha = 0.10$ — Base minimum

$\alpha = 0.50$ — Median demand

$\alpha = 0.90$ — Rush-hour surge buffer

Intraday Peak Detection: Evaluates the top 25% demand periods to ensure high service availability during morning (07:00–10:00) and lunch (12:00–14:00) rushes.

Continuous Time-Series Resampling: Automatically constructs fixed 1-hour temporal grids and imputes silent non-trading intervals to remove zero-sales distortion.

Interactive Decision Dashboard: Built with Streamlit and Plotly, featuring interactive 80% confidence ribbons, diurnal heatmaps, and shift preparation sheets.

System Architecture
afficionado-forecasting/
│
├── data/
│   └── coffee_sales.csv
│       └── Raw POS transaction logs
│
├── src/
│   ├── data_pipeline.py
│   │   └── Hourly resampling, zero-fill, lag & cyclical features
│   │
│   └── models.py
│       └── Baseline, SARIMAX, Quantile LightGBM & wMAPE KPI engine
│
├── app.py
│   └── Streamlit analytical web dashboard
│
├── research_paper.md
│   └── Full academic research paper and methodology
│
├── executive_summary.md
│   └── Operational executive summary for stakeholders
│
├── requirements.txt
│   └── Pinned production dependencies
│
└── README.md
    └── Complete project documentation

Analytical Pipeline & Feature Engineering
[Raw POS Transactions]
          │
          ▼
[Continuous Hourly Grid Imputation]
          │
          │ Zero-sales padding
          ▼
[Feature Engineering Store]
          │
          ├── Cyclical Transforms
          │   ├── sin(2π·hour/24)
          │   ├── cos(2π·hour/24)
          │   ├── sin(2π·dow/7)
          │   └── cos(2π·dow/7)
          │
          ├── Autoregressive Lags
          │   ├── t-1   (immediate)
          │   ├── t-24  (daily cycle)
          │   └── t-168 (weekly cycle)
          │
          └── Rolling Volatility
              ├── 24h rolling mean & standard deviation
              └── 168h rolling mean & standard deviation
          
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
[SARIMAX Statistical Engine]                  [Quantile LightGBM Model]
          │                                               │
          │                                               ├── α = 0.10
          │                                               │   Minimum Reserve
          │                                               │
          │                                               ├── α = 0.50
          │                                               │   Operational Target
          │                                               │
          │                                               └── α = 0.90
          │                                                   Surge Safety Buffer
          │
          └───────────────────────┬───────────────────────┘
                                  ▼
              [Streamlit Operational Intelligence Dashboard]

Empirical Benchmark & Model Evaluation

Models are evaluated using out-of-time forward validation. Accuracy is tracked using Weighted Absolute Percentage Error (wMAPE) to reduce zero-division issues during off-peak and closing hours.

wMAPE

wMAPE
=
∑
𝑡
∣
𝑦
𝑡
−
𝑦
^
𝑡
∣
∑
𝑡
𝑦
𝑡
×
100

Accuracy
=
100
−
wMAPE

Benchmark Results
Model Architecture	MAE ($)	RMSE ($)	wMAPE (%)	Peak Demand Error (%)	Overall Accuracy (%)
Naive Persistence (Last Week)	34.20	52.80	18.4%	22.1%	81.6%
SARIMAX (1,1,1) × (1,0,1)₂₄	22.60	38.10	12.8%	15.3%	87.2%
Quantile LightGBM Engine	12.45	19.80	6.6%	7.8%	93.4%
Streamlit Dashboard Features

Store Selector: Filter predictions dynamically across individual retail outlets, such as:

Lower Manhattan

Astoria

Hell's Kitchen

Dual Target Modes: Toggle between:

revenue — Monetary tracking

transaction_volume — Customer footfall

Algorithm Comparison: Benchmark predictions across:

Naive Baseline

SARIMAX

Quantile LightGBM

Visual Confidence Band: Shaded 80% interval between $\alpha = 0.10$ and $\alpha = 0.90$ to identify rush-hour surges and downswings.

Diurnal Demand Heatmap: Cross-tabulation of hour-of-day vs. day-of-week to support barista scheduling and pre-brew planning.

Inventory Prep Table: Daily expected totals alongside recommended +20% safety-stock buffers.

Quickstart & Local Setup
1. Clone the Repository
git clone https://github.com/<your-username>/afficionado-forecasting.git
cd afficionado-forecasting

2. Create a Virtual Environment
Windows — Command Prompt
python -m venv venv
venv\Scripts\activate.bat

Linux / macOS
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

4. Run the Streamlit Application
streamlit run app.py


Then open:

http://localhost:8501

Dataset Schema Reference
Column Name	Data Type	Description
transaction_id	Integer / String	Unique identifier per transaction
transaction_date	Date (YYYY-MM-DD)	Date of transaction
transaction_time	Time (HH:MM:SS)	Timestamp of purchase
transaction_qty	Integer	Quantity of items purchased
unit_price	Float	Price per item sold
store_id	Integer	Unique identifier for the store branch
store_location	String	Physical branch name (e.g., Astoria, Lower Manhattan)
product_category	String	Broad product classification (e.g., Espresso, Cold Brew)
Operational Recommendations
Split-Shift Scheduling

Align barista floor rosters with the empirical 07:00–10:00 and 12:00–14:00 surge windows while reducing staffing overhead during the 14:00–16:00 lower-demand period.

Asymmetric Ordering

Stock high-spoilage perishables such as milk, dairy alternatives, and fresh pastries toward the median $\alpha = 0.50$ forecast, while maintaining dry goods such as beans, cups, and syrups closer to the $\alpha = 0.90$ surge boundary.

Automated Replenishment

Feed the 7-day store-level forecasts into central roasting and logistics hubs to support efficient batch transportation and replenishment planning.

License

Distributed under the MIT License. See LICENSE for details.