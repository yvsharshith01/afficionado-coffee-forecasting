\# Afficionado Coffee Roasters | Predictive Retail Intelligence \& Demand Forecasting



\[!\[Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

\[!\[Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)

\[!\[LightGBM](https://img.shields.io/badge/LightGBM-4.3+-brightgreen.svg)](https://lightgbm.readthedocs.io/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



An end-to-end machine learning and time-series forecasting framework designed for \*\*Afficionado Coffee Roasters\*\*. The platform converts granular point-of-sale transaction logs into store-level predictive intelligence, quantifying uncertainty during high-frequency rush periods and optimizing staff scheduling and inventory allocation.



\---



\## Executive Overview



Coffee retail demand is volatile:

\- \*\*Intraday Rush Concentration:\*\* Over 40% of transaction activity occurs in compressed morning (07:00–10:00) and lunch (12:00–14:00) windows.

\- \*\*Perishable Inventory Risk:\*\* Overestimating demand leads to ingredient spoilage; underestimating demand creates walkouts and unfulfilled transactions.

\- \*\*Store Variance:\*\* Demand distribution shifts fundamentally between downtown flagship units, business district hubs, and suburban centers.



This system replaces static heuristics with an automated data pipeline, statistical and gradient-boosted time-series models, quantile prediction intervals (10th to 90th percentile), and an interactive Streamlit operations dashboard.



\---



\## System Architecture



afficionado-forecasting/

├── data/

│   └── coffee\_sales.csv            # Raw transaction logs

├── src/

│   ├── data\_pipeline.py            # Continuous time-grid construction \& lag engineering

│   └── models.py                   # Baseline, SARIMAX, and Quantile LightGBM models

├── app.py                          # Multi-page interactive Streamlit dashboard

├── research\_paper.md               # Detailed analytical methodology \& empirical study

├── executive\_summary.md            # Operational brief for leadership \& stakeholders

├── requirements.txt                # Pinned production dependencies

└── README.md                       # Documentation \& execution guide





\---



\## Pipeline \& Predictive Methodology



\[Raw POS Transactions]

│

▼

\[Continuous Hourly Grid Imputation] (Zero-sales padding)

│

▼

\[Feature Engineering]

├── Cyclical Transforms: Sin/Cos representations of hour and day-of-week

├── Autoregressive Lags: t-1, t-24 (diurnal), t-168 (weekly)

└── Moving Window Features: 24h \& 168h rolling means and standard deviations

│

┌────┴───────────────────────────┐

▼                                ▼

\[SARIMAX Statistical Model]   \[Multi-Quantile LightGBM Regressor]

│                                ├── Alpha = 0.10 (Base Stock Minimum)

│                                ├── Alpha = 0.50 (Median Operational Target)

│                                └── Alpha = 0.90 (Surge Safety Buffer)

└────┬───────────────────────────┘

▼

\[Streamlit Interactive Intelligence Engine]

├── Dynamic Forecast vs. Actual Timeline with 80% Confidence Band

├── Diurnal Demand Heatmap (Hour of Day vs. Day of Week)

└── Daily Shift \& Inventory Allocation Prep-Sheets





\---



\## Model Benchmark \& Evaluation



Models were evaluated across dynamic out-of-time test horizons using Weighted Absolute Percentage Error (wMAPE) and Peak Demand Capture Rate (top 25% rush hours):



| Model Architecture | MAE ($) | RMSE ($) | wMAPE (%) | Peak Demand Error (%) | Overall Accuracy (%) |

| :--- | :---: | :---: | :---: | :---: | :---: |

| \*\*Naive Weekly Baseline\*\* | 34.20 | 52.80 | 18.4% | 22.1% | 81.6% |

| \*\*SARIMAX $(1,1,1) \\times (1,0,1)\_{24}$\*\* | 22.60 | 38.10 | 12.8% | 15.3% | 87.2% |

| \*\*Quantile LightGBM Engine\*\* | \*\*12.45\*\* | \*\*19.80\*\* | \*\*6.6%\*\* | \*\*7.8%\*\* | \*\*93.4%\*\* |



\---



\## Key Performance Indicators (KPIs)



\- \*\*Forecast Accuracy:\*\* Reliability score based on out-of-sample ground truth comparisons.

\- \*\*wMAPE (Weighted MAPE):\*\* Evaluates overall scale-dependent error without dividing by zero during off-peak and closing hours.

\- \*\*Peak Demand Capture Rate:\*\* Measures accuracy during the highest 25% demand spikes to prevent stockouts during morning and lunch rushes.

\- \*\*Surge Buffer Allocation:\*\* 90th percentile predictions supply store managers with data-backed prep-sheet safety thresholds.



\---



\## Local Installation \& Execution



\### 1. Clone the Repository

```bash

git clone \[https://github.com/](https://github.com/)<your-username>/afficionado-forecasting.git

cd afficionado-forecasting

2\. Configure Virtual Environment

Windows (Command Prompt):



DOS

python -m venv venv

venv\\Scripts\\activate.bat

Linux / macOS:



Bash

python3 -m venv venv

source venv/bin/activate

3\. Install Dependencies

Bash

pip install --upgrade pip

pip install -r requirements.txt

4\. Run the Streamlit Dashboard

Bash

streamlit run app.py

The application will launch automatically at http://localhost:8501.



Dashboard Capabilities

Store Selector: Filter forecasts dynamically by location (e.g., Lower Manhattan, Astoria, Hell's Kitchen).



Forecast Horizon Slider: Generate multi-step forecasts from 1 to 30 days ahead.



Target Toggle: Switch between predicting hourly Transaction Volume (for staff scheduling) and Gross Revenue (for financial modeling).



Algorithm Comparison: Evaluate performance between Naive Baseline, SARIMAX, and Multi-Quantile GBDT models.



Confidence Intervals: Interactive visual bands representing the 80% prediction interval (10th to 90th percentiles).



Operational Heatmaps: Hour-of-day versus day-of-week intensity matrix to pinpoint store rush windows.



License

Distributed under the MIT License. See LICENSE for more information.





\---

