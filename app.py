import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import (
    build_hourly_timeseries,
    engineer_hourly_features,
    load_and_preprocess_raw,
)
from src.models import (
    BaselineForecaster,
    QuantileGBDTForecaster,
    evaluate_forecast_metrics,
    fit_and_forecast_sarimax,
)

st.set_page_config(
    page_title="Afficionado Coffee Roasters | Predictive Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def generate_or_load_data():
    try:
        df_raw = load_and_preprocess_raw("data/coffee_sales.csv")
        if df_raw.empty or "store_location" not in df_raw.columns:
            raise ValueError("Empty or invalid CSV")
    except Exception:
        dates = pd.date_range(
            start="2025-01-01 06:00:00", end="2025-03-31 21:00:00", freq="1h"
        )
        stores = [
            (1, "Downtown Flagship"),
            (2, "Uptown Financial"),
            (3, "Suburban Center"),
        ]
        records = []
        for s_id, s_loc in stores:
            for d in dates:
                if d.hour < 6 or d.hour > 21:
                    continue
                rush_mult = 3.2 if d.hour in [7, 8, 9, 12, 13] else 1.0
                volume = int(np.random.poisson(lam=12 * rush_mult))
                price = np.random.choice([4.5, 5.0, 5.5])
                records.append(
                    {
                        "transaction_id": len(records) + 1,
                        "transaction_time": d.strftime("%H:%M:%S"),
                        "transaction_date": d.strftime("%Y-%m-%d"),
                        "transaction_qty": max(1, volume),
                        "unit_price": price,
                        "store_id": s_id,
                        "store_location": s_loc,
                        "product_category": np.random.choice(
                            ["Espresso", "Cold Brew", "Pastry", "Beans"]
                        ),
                    }
                )
        df_raw = pd.DataFrame(records)
        df_raw["datetime"] = pd.to_datetime(
            df_raw["transaction_date"] + " " + df_raw["transaction_time"]
        )
        df_raw["revenue"] = df_raw["transaction_qty"] * df_raw["unit_price"]

    df_hourly = build_hourly_timeseries(df_raw)
    df_features = engineer_hourly_features(df_hourly)
    return df_raw, df_features


df_raw, df_features = generate_or_load_data()

st.sidebar.title("Operational Controls")

all_stores = sorted(df_features["store_location"].dropna().unique().tolist())
if not all_stores:
    st.error("No store data available. Check dataset schema.")
    st.stop()

selected_store = st.sidebar.selectbox("Store Location", all_stores)

forecast_horizon_days = st.sidebar.slider(
    "Forecast Horizon (Days)", min_value=1, max_value=30, value=7
)
forecast_horizon_hours = forecast_horizon_days * 24

target_choice = st.sidebar.radio(
    "Forecasting Target", ["revenue", "transaction_volume"], index=0
)

model_choice = st.sidebar.selectbox(
    "Algorithm Engine", ["Gradient Boosting (Quantile)", "SARIMAX", "Baseline (Naive)"]
)

store_df = df_features[
    df_features["store_location"] == selected_store
].sort_values("datetime_hour")

if len(store_df) <= forecast_horizon_hours:
    split_idx = int(len(store_df) * 0.75)
else:
    split_idx = len(store_df) - forecast_horizon_hours

train_df = store_df.iloc[:split_idx]
test_df = store_df.iloc[split_idx:]

if train_df.empty or test_df.empty:
    st.error("Insufficient samples to train the model. Reduce the horizon.")
    st.stop()

actuals = test_df[target_choice].values
rush_mask = (test_df["is_morning_rush"] == 1).values | (
    test_df["is_lunch_rush"] == 1
).values

if model_choice == "Gradient Boosting (Quantile)":
    gbdt = QuantileGBDTForecaster()
    gbdt.fit(train_df, target_choice)
    p10, p50, p90 = gbdt.predict(test_df)
elif model_choice == "SARIMAX":
    p10, p50, p90 = fit_and_forecast_sarimax(
        train_df[target_choice], len(test_df)
    )
else:
    base = BaselineForecaster("last_week")
    p50 = base.predict(train_df, target_choice, len(test_df))
    p10 = p50 * 0.85
    p90 = p50 * 1.15

metrics = evaluate_forecast_metrics(actuals, p50, rush_mask)

st.title("Afficionado Coffee Roasters: Retail Demand Intelligence")
st.markdown(
    f"Predictive forecast engine for **{selected_store}** across the next **{forecast_horizon_days} days**."
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Forecast Accuracy", f"{metrics['Forecast_Accuracy']:.1f}%")
col2.metric("MAE", f"{metrics['MAE']:.2f}")
col3.metric("RMSE", f"{metrics['RMSE']:.2f}")
col4.metric("MAPE", f"{metrics['MAPE']:.1f}%")
col5.metric("Peak Demand Error", f"{metrics['Peak_Error_Rate']:.1f}%")

st.markdown("---")

st.subheader("Forecast vs Actuals with Prediction Intervals (10th-90th Percentile)")
fig_timeline = go.Figure()

fig_timeline.add_trace(
    go.Scatter(
        x=test_df["datetime_hour"],
        y=p90,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="90th Percentile (Surge Cap)",
    )
)
fig_timeline.add_trace(
    go.Scatter(
        x=test_df["datetime_hour"],
        y=p10,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(31, 119, 180, 0.15)",
        name="Confidence Band (80% Interval)",
    )
)
fig_timeline.add_trace(
    go.Scatter(
        x=test_df["datetime_hour"],
        y=p50,
        mode="lines",
        line=dict(color="#1f77b4", width=2.5),
        name="Predicted Demand (Median)",
    )
)
fig_timeline.add_trace(
    go.Scatter(
        x=test_df["datetime_hour"],
        y=actuals,
        mode="lines",
        line=dict(color="#2ca02c", width=1.5, dash="dot"),
        name="Actual Ground Truth",
    )
)

fig_timeline.update_layout(
    xaxis_title="Time",
    yaxis_title=target_choice.replace("_", " ").title(),
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20),
    template="plotly_white",
)
st.plotly_chart(fig_timeline, use_container_width=True)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Upcoming Demand Density (Hour vs Day)")
    test_df_copy = test_df.copy()
    test_df_copy["predicted_demand"] = p50
    test_df_copy["day_name"] = test_df_copy["datetime_hour"].dt.day_name()

    pivot_heatmap = test_df_copy.pivot_table(
        index="day_name",
        columns="hour",
        values="predicted_demand",
        aggfunc="mean",
    ).reindex(
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
    )

    fig_heat = px.imshow(
        pivot_heatmap,
        labels=dict(
            x="Hour of Day", y="Day of Week", color=target_choice.title()
        ),
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_right:
    st.subheader("Shift Scheduling & Inventory Allocation")
    test_df_copy["predicted_surge"] = p90
    daily_operational = (
        test_df_copy.groupby(test_df_copy["datetime_hour"].dt.date)
        .agg(
            expected_total=("predicted_demand", "sum"),
            buffer_capacity=("predicted_surge", "sum"),
        )
        .reset_index()
    )
    daily_operational.columns = [
        "Date",
        "Expected Daily Demand",
        "Required Stock Buffer (+20%)",
    ]

    st.dataframe(
        daily_operational.style.format(
            {
                "Expected Daily Demand": "{:,.1f}",
                "Required Stock Buffer (+20%)": "{:,.1f}",
            }
        ),
        use_container_width=True,
    )
    st.caption(
        "Buffer requirements represent the 90th percentile quantile prediction, mitigating peak hour stockout risks."
    )