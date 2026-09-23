\# Executive Summary: Predictive Retail Intelligence \& Demand Forecasting

\*\*Target Entity:\*\* Executive Leadership \& Store Operations, Afficionado Coffee Roasters  

\*\*Author:\*\* Yalamarthi Viswa Sri Harshith  

\*\*Status:\*\* Completed \& Validated  



\---



\### Executive Problem Overview

Afficionado Coffee Roasters operates high-volume urban and suburban retail units. Demand is heavily concentrated in morning (07:00–10:00) and midday (12:00–14:00) peaks. Operating without data-driven forecasting created two major operational failures:

\- \*\*Under-preparation during surges:\*\* Stockouts and barista understaffing caused long queues, order abandonments, and lost revenue.

\- \*\*Over-preparation during lulls:\*\* Overestimating afternoon footfall caused excessive spoilage of fresh milk, brewed coffee, and artisan pastries.



\---



\### Key Analytical Achievements \& Business Impact



| Metric / Capability | Legacy Operational Heuristics | Proposed ML Framework | Operational Benefit |

| :--- | :---: | :---: | :--- |

| \*\*Forecast Accuracy\*\* | \~81.6% | \*\*93.4%\*\* | Reliable financial and revenue planning |

| \*\*Peak Demand Error\*\* | 22.1% error | \*\*7.8% error\*\* | High rush-hour fulfillment without bottlenecks |

| \*\*Inventory Spoilage\*\* | Baseline industry rate | \*\*Estimated 14%–18% reduction\*\* | Substantial weekly cost savings on perishables |

| \*\*Uncertainty Quantification\*\* | None (Single static number) | \*\*10th to 90th Percentile Bands\*\* | Safe risk buffers for inventory stocking |



\---



\### Core Deliverables Provided



1\. \*\*Interactive Streamlit Operations Dashboard:\*\*

&#x20;  - \*\*Store-Level Selector:\*\* Instant drill-down across individual locations (Astoria, Lower Manhattan, Hell's Kitchen).

&#x20;  - \*\*Dual Target Forecasting:\*\* Toggle between \*\*Gross Revenue\*\* (financial budgeting) and \*\*Transaction Volume\*\* (barista labor scheduling).

&#x20;  - \*\*Visual 80% Confidence Ribbons:\*\* Interactive timeline showing expected demand alongside lower-bound reserves and upper surge limits.

&#x20;  - \*\*Diurnal Heatmaps:\*\* Day-of-week vs. hour-of-day intensity matrices for shift planning.



2\. \*\*Continuous Feature Engineering Pipeline:\*\*

&#x20;  - Automatically repairs raw POS logs by imputing non-trading hours, calculating trigonometric time cycles, and producing multi-period autoregressive rolling features.



3\. \*\*Multi-Horizon Engine:\*\*

&#x20;  - Supports operational short-term daily scheduling (1–7 days) and strategic supply chain ordering (14–30 days).



\---



\### Strategic Action Plan for Leadership



1\. \*\*Implement Flexible Rostering:\*\* Stagger floor shifts into targeted 4-hour blocks aligned with the dashboard's hourly demand heatmaps, reducing labor costs during off-peak windows.

2\. \*\*Adopt Asymmetric Prep Policies:\*\* Direct store managers to prep perishable dairy and pastries against the median ($\\alpha = 0.50$) forecast while holding dry goods at the 90th percentile ($\\alpha = 0.90$) surge buffer.

3\. \*\*Logistics Integration:\*\* Connect the 7-day store volume forecasts directly to central roasting facility dispatchers to optimize batch roasting schedules and delivery routes.

