import numpy as np
import pandas as pd


def load_and_preprocess_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "transaction_date" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["transaction_date"].astype(str)
            + " "
            + df["transaction_time"].astype(str)
        )
    elif "year" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["year"].astype(str) + "-01-01 " + df["transaction_time"].astype(str)
        )
    else:
        df["datetime"] = pd.to_datetime(df["transaction_time"])

    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    return df


def build_hourly_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime_hour"] = df["datetime"].dt.floor("h")

    hourly = (
        df.groupby(["store_id", "store_location", "datetime_hour"])
        .agg(
            transaction_volume=("transaction_id", "nunique"),
            total_quantity=("transaction_qty", "sum"),
            revenue=("revenue", "sum"),
        )
        .reset_index()
    )

    records = []
    for store_id, group in hourly.groupby("store_id"):
        location = group["store_location"].iloc[0]
        min_time = group["datetime_hour"].min()
        max_time = group["datetime_hour"].max()

        full_index = pd.date_range(
            start=min_time, end=max_time, freq="h", name="datetime_hour"
        )
        group = (
            group.set_index("datetime_hour")
            .reindex(full_index)
            .fillna(
                {
                    "store_id": store_id,
                    "store_location": location,
                    "transaction_volume": 0.0,
                    "total_quantity": 0.0,
                    "revenue": 0.0,
                }
            )
            .reset_index()
        )
        records.append(group)

    return pd.concat(records, ignore_index=True)


def engineer_hourly_features(df_hourly: pd.DataFrame) -> pd.DataFrame:
    df = df_hourly.sort_values(["store_id", "datetime_hour"]).copy()

    df["hour"] = df["datetime_hour"].dt.hour
    df["dayofweek"] = df["datetime_hour"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["month"] = df["datetime_hour"].dt.month

    df["is_morning_rush"] = df["hour"].isin([7, 8, 9, 10]).astype(int)
    df["is_lunch_rush"] = df["hour"].isin([12, 13, 14]).astype(int)

    for target in ["transaction_volume", "revenue"]:
        df[f"{target}_lag_1"] = df.groupby("store_id")[target].shift(1).bfill().fillna(0)
        df[f"{target}_lag_24"] = df.groupby("store_id")[target].shift(24).bfill().fillna(0)
        df[f"{target}_lag_168"] = df.groupby("store_id")[target].shift(168).bfill().fillna(0)

        df[f"{target}_roll_mean_24"] = (
            df.groupby("store_id")[target]
            .shift(1)
            .rolling(window=24, min_periods=1)
            .mean()
            .bfill()
            .fillna(0)
        )
        df[f"{target}_roll_std_24"] = (
            df.groupby("store_id")[target]
            .shift(1)
            .rolling(window=24, min_periods=1)
            .std()
            .fillna(0)
        )
        df[f"{target}_roll_mean_168"] = (
            df.groupby("store_id")[target]
            .shift(1)
            .rolling(window=168, min_periods=1)
            .mean()
            .bfill()
            .fillna(0)
        )

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    return df.fillna(0).reset_index(drop=True)