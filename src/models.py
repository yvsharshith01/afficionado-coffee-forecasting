from typing import Dict, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def evaluate_forecast_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, rush_mask: np.ndarray = None
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    # Weighted MAPE (avoids division-by-zero during low/zero sales hours)
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual > 0:
        wmape = (np.sum(np.abs(y_true - y_pred)) / sum_actual) * 100.0
    else:
        wmape = 0.0

    forecast_acc = max(0.0, min(100.0, 100.0 - wmape))

    # Peak Error: Evaluate the top 25% highest actual demand periods
    if len(y_true) > 0:
        threshold = np.quantile(y_true, 0.75)
        peak_idx = y_true >= threshold
        if np.any(peak_idx) and np.sum(y_true[peak_idx]) > 0:
            peak_err = (
                np.sum(np.abs(y_true[peak_idx] - y_pred[peak_idx]))
                / np.sum(y_true[peak_idx])
            ) * 100.0
        else:
            peak_err = wmape
    else:
        peak_err = 0.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(np.clip(wmape, 0.0, 100.0)),
        "Peak_Error_Rate": float(np.clip(peak_err, 0.0, 100.0)),
        "Forecast_Accuracy": float(forecast_acc),
    }


class BaselineForecaster:

    def __init__(self, strategy: str = "last_week"):
        self.strategy = strategy

    def predict(self, df_history: pd.DataFrame, target_col: str, horizon: int) -> np.ndarray:
        history = df_history[target_col].values
        if len(history) == 0:
            return np.zeros(horizon)

        if self.strategy == "naive_last":
            return np.full(horizon, history[-1])
        elif self.strategy == "last_week" and len(history) >= 24:
            # Repeat the last diurnal cycle
            cycle_len = min(168, len(history))
            vals = history[-cycle_len:]
            tiles = int(np.ceil(horizon / len(vals)))
            return np.tile(vals, tiles)[:horizon]
        else:
            # Fallback to mean of recent history
            return np.full(horizon, np.mean(history[-24:] if len(history) >= 24 else history))


class QuantileGBDTForecaster:

    def __init__(self):
        self.features = [
            "hour",
            "dayofweek",
            "is_weekend",
            "month",
            "is_morning_rush",
            "is_lunch_rush",
            "sin_hour",
            "cos_hour",
            "sin_dow",
            "cos_dow",
        ]
        self.lag_features = [
            "lag_1",
            "lag_24",
            "lag_168",
            "roll_mean_24",
            "roll_std_24",
            "roll_mean_168",
        ]
        self.models = {}

    def fit(self, train_df: pd.DataFrame, target_col: str):
        dynamic_lags = [f"{target_col}_{feat}" for feat in self.lag_features]
        feature_cols = self.features + dynamic_lags

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        for alpha in [0.1, 0.5, 0.9]:
            reg = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=180,
                learning_rate=0.04,
                num_leaves=31,
                random_state=42,
                verbosity=-1,
            )
            reg.fit(X_train, y_train)
            self.models[alpha] = reg

        self.fitted_features = feature_cols

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_test = test_df[self.fitted_features]
        p10 = self.models[0.1].predict(X_test)
        p50 = self.models[0.5].predict(X_test)
        p90 = self.models[0.9].predict(X_test)

        p10 = np.clip(p10, 0, None)
        p50 = np.clip(p50, 0, None)
        p90 = np.maximum(p90, p50)
        return p10, p50, p90


def fit_and_forecast_sarimax(
    train_series: pd.Series, horizon: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        model = SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 24),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        forecast_res = res.get_forecast(steps=horizon)
        p50 = np.clip(forecast_res.predicted_mean.values, 0, None)
        conf_int = forecast_res.conf_int(alpha=0.2).values
        p10 = np.clip(conf_int[:, 0], 0, None)
        p90 = np.clip(conf_int[:, 1], 0, None)
        return p10, p50, p90
    except Exception:
        mean_val = train_series.iloc[-168:].mean() if len(train_series) >= 168 else train_series.mean()
        std_val = train_series.iloc[-168:].std() if len(train_series) >= 168 else train_series.std()
        std_val = 0.0 if np.isnan(std_val) else std_val
        p50 = np.full(horizon, mean_val)
        p10 = np.full(horizon, max(0.0, mean_val - 1.28 * std_val))
        p90 = np.full(horizon, mean_val + 1.28 * std_val)
        return p10, p50, p90