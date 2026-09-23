\# Data-Driven Forecasting \& Peak Demand Prediction for Afficionado Coffee Roasters



\*\*Target Domain:\*\* Retail Analytics, High-Frequency Time-Series Forecasting, Operational Logistics  



\---



\## Abstract

Coffee retail is characterized by non-stationary, periodic demand curves where intraday surges (morning and midday rushes) dictate profitability and labor utilization. Inadequate forecasting results in stockouts during demand spikes or costly over-preparation and ingredient spoilage. This paper presents an end-to-end predictive framework for Afficionado Coffee Roasters that ingests transaction logs, synthesizes consistent temporal intervals, and evaluates baseline, statistical, and gradient-boosted decision tree architectures. The proposed Quantile LightGBM engine achieves a \*\*93.4% Forecast Accuracy\*\* and reduces rush-hour Peak Error Rate to \*\*7.8%\*\*, outperforming standard SARIMAX and rolling baseline benchmarks while providing 10th-to-90th percentile prediction bands for risk-calibrated inventory scheduling.



\---



\## 1. Introduction \& Operational Challenges

Retail coffee operations run on strict shelf-life constraints and compressed consumption cycles:

\- \*\*Morning Rush Disruption:\*\* Over 40% of daily transactions take place between 07:00 and 10:00. Unpredicted peaks create bottleneck queues, resulting in abandoned orders.

\- \*\*Midday Volatility:\*\* Midday revenue is heavily determined by office proximity, footfall, and local patterns.

\- \*\*Supply Waste:\*\* Fresh dairy, brewed batches, and baked goods deteriorate rapidly, making aggregate daily forecasts insufficient for hourly prep-sheet schedules.



Traditional planning relies on informal intuition or uniform linear rules (such as historical same-day averages). This project builds a localized predictive model capable of delivering granular forecasts (hourly and daily) across all retail units.



\---



\## 2. Dataset Processing \& Time-Series Construction

The raw dataset contains transaction-level metadata:

$$\\mathcal{D} = \\{\\text{transaction\\\_id}, \\text{timestamp}, \\text{store\\\_id}, \\text{store\\\_location}, \\text{product\\\_category}, \\text{transaction\\\_qty}, \\text{unit\\\_price}\\}$$



\### 2.1 Missing Interval Imputation

A primary hurdle in transaction datasets is "silent periods"—hours where zero sales occur are omitted from logs rather than explicitly recorded. To address this:

1\. A continuous chronological grid $T \\in \[t\_{\\min}, t\_{\\max}]$ with step size $\\Delta t = 1\\text{ hour}$ was generated for every `store\_id`.

2\. Missing intervals were joined and imputed with $\\text{revenue} = 0$ and $\\text{transaction\\\_volume} = 0$, preventing upward sampling bias.



\### 2.2 Feature Engineering

Temporal, cyclical, and autoregressive lag predictors were formulated:

\- \*\*Cyclical Trigonometric Transformations:\*\*

&#x20; $$\\sin\\left(\\frac{2\\pi \\cdot \\text{hour}}{24}\\right), \\quad \\cos\\left(\\frac{2\\pi \\cdot \\text{hour}}{24}\\right), \\quad \\sin\\left(\\frac{2\\pi \\cdot \\text{dayofweek}}{7}\\right), \\quad \\cos\\left(\\frac{2\\pi \\cdot \\text{dayofweek}}{7}\\right)$$

\- \*\*Autoregressive Lags:\*\* Lagged targets at $t-1$ (previous hour autocorrelation), $t-24$ (diurnal cycle), and $t-168$ (weekly cycle).

\- \*\*Rolling Windows:\*\* Trailing 24-hour and 168-hour moving averages and standard deviations, strictly lagged by 1 step to eliminate lookahead bias.



\---



\## 3. Modeling Architectures



\[Raw Transactions]

│

▼

\[Continuous Hourly Grid Imputation]

│

▼

\[Lag \& Cyclical Feature Store]

│

┌────┴───────────────────────────┐

▼                                ▼

\[SARIMAX (Statistical)]   \[Quantile LightGBM (Machine Learning)]

│                                │

│                                ├── Alpha = 0.10 (Lower Bound / Minimum)

│                                ├── Alpha = 0.50 (Median Expectation)

│                                └── Alpha = 0.90 (Surge Buffer / Peak)

└────┬───────────────────────────┘

▼

\[Model Evaluation \& Operational Heatmaps via Streamlit]





\### 3.1 Baselines

\- \*\*Naive Persistence:\*\* Assumes $y\_{t} = y\_{t-168}$ (demand mirrors the exact hour of the prior week).

\- \*\*Moving Average:\*\* Trailing 7-day mean demand per hour slot.



\### 3.2 SARIMAX

Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors:

$$\\Phi\_P(B^s)\\phi\_p(B)(1-B)^d(1-B^s)^D y\_t = \\Theta\_Q(B^s)\\theta\_q(B)\\epsilon\_t$$

Configured with order $(1, 1, 1)$ and seasonal order $(1, 0, 1)\_{24}$ to capture diurnal repetition.



\### 3.3 Multi-Quantile Gradient Boosting (LightGBM)

Rather than optimizing solely on Mean Squared Error ($L\_2$ loss), three separate gradient boosted trees were trained using Pinball Loss:

$$\\mathcal{L}\_\\alpha(y, \\hat{y}) = \\max(\\alpha(y - \\hat{y}), (1 - \\alpha)(\\hat{y} - y))$$

\- $\\alpha = 0.50$: Median forecast (operational baseline).

\- $\\alpha = 0.10$: Lower boundary (base stock minimum).

\- $\\alpha = 0.90$: Upper boundary (surge allocation to ensure a 90% service level during peak hours).



\---



\## 4. Empirical Evaluation



Models were evaluated across a rolling out-of-time horizon:



| Model Architecture | MAE ($) | RMSE ($) | MAPE (%) | Peak Demand Error (%) | Overall Accuracy (%) |

| :--- | :---: | :---: | :---: | :---: | :---: |

| \*\*Naive (Last Week)\*\* | 34.20 | 52.80 | 18.4% | 22.1% | 81.6% |

| \*\*SARIMAX (1,1,1)x(1,0,1)\*\* | 22.60 | 38.10 | 12.8% | 15.3% | 87.2% |

| \*\*Quantile LightGBM\*\* | \*\*12.45\*\* | \*\*19.80\*\* | \*\*6.6%\*\* | \*\*7.8%\*\* | \*\*93.4%\*\* |



The Quantile LightGBM model outperforms statistical baselines by learning non-linear interactions between hour indicators, weekly drift, and recent rolling volatility.



\---



\## 5. Strategic Recommendations

1\. \*\*Dynamic Shift Rostering:\*\* Transition from fixed 8-hour staff schedules to split-shift rosters aligned with predicted 07:00-10:00 and 12:00-14:00 surges, reducing overstaffing overhead during the 14:00-16:00 slump.

2\. \*\*Buffer Inventory Policy:\*\* Order perishable supplies (e.g., dairy and pastries) to match the median $\\alpha=0.5$ forecast, while maintaining dry goods and syrups up to the 90th percentile ceiling ($\\alpha=0.9$).

3\. \*\*Automated Replenishment:\*\* Feed the 7-day predictive store estimates into warehouse

