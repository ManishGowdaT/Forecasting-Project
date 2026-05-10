import matplotlib
matplotlib.use("Agg")

import os
import logging
import warnings

import numpy as np
import pandas as pd
import joblib
from forecast_metrics import calculate_metrics

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

HORIZON  = 8
FEATURES = [
    "lag_1", "lag_4", "lag_52",
    "rolling_mean_4", "rolling_mean_12", "rolling_std_4",
    "week_of_year", "month", "quarter", "year", "is_holiday",
]

state_datasets = joblib.load("artifacts/state_datasets.pkl")
weekly_data    = joblib.load("artifacts/weekly_data.pkl")
states         = sorted(state_datasets.keys())


def calc_metrics(actual, predicted):
    return calculate_metrics(actual, predicted)


# ── prophet ───────────────────────────────────────────────
print("\n--- prophet ---")
from prophet import Prophet

prophet_results = {}
prophet_models  = {}

for i, state in enumerate(states):
    series = weekly_data[state]
    train  = series.iloc[:-HORIZON]
    val    = series.iloc[-HORIZON:]
    try:
        df_train = pd.DataFrame({"ds": train.index, "y": train.values})
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
        )
        m.add_country_holidays(country_name="US")
        m.fit(df_train)

        future = m.make_future_dataframe(periods=HORIZON, freq="W")
        fc     = m.predict(future)
        preds  = np.clip(fc["yhat"].iloc[-HORIZON:].values, 0, None)

        mae, rmse, mape = calc_metrics(val.values, preds)
        prophet_results[state] = {
            "forecast": pd.Series(preds, index=val.index),
            "actual": val, "mae": mae, "rmse": rmse, "mape": mape,
        }
        prophet_models[state] = m
        print(f"  [{i+1:02d}/{len(states)}] {state:<20} mape={mape:.1f}%")
    except Exception as e:
        print(f"  [{i+1:02d}/{len(states)}] {state:<20} failed: {e}")
        prophet_results[state] = None

joblib.dump(prophet_results, "artifacts/prophet_results.pkl")
joblib.dump(prophet_models,  "artifacts/prophet_models.pkl")
print("prophet saved")


# ── xgboost ───────────────────────────────────────────────
print("\n--- xgboost ---")
from xgboost import XGBRegressor

xgb_results = {}
xgb_models  = {}

for i, state in enumerate(states):
    data   = state_datasets[state]
    train  = data["train"]
    val    = data["val"]
    series = data["weekly_series"]
    try:
        if len(val) != HORIZON or not val.index.equals(series.iloc[-HORIZON:].index):
            raise ValueError("Validation features do not match the weekly holdout dates")
        m = XGBRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        m.fit(train[FEATURES], train["y"])

        hist  = series.copy()
        preds = []
        for step in range(HORIZON):
            v  = hist.values
            fd = val.index[step]
            row = np.array([[
                v[-1],
                v[-4]  if len(v) >= 4  else v[0],
                v[-52] if len(v) >= 52 else v[0],
                np.mean(v[-4:]),  np.mean(v[-12:]),  np.std(v[-4:]),
                fd.isocalendar().week, fd.month,
                (fd.month - 1) // 3 + 1, fd.year,
                1 if fd.month in (1, 7, 11, 12) else 0,
            ]])
            pred = max(float(m.predict(pd.DataFrame(row, columns=FEATURES))[0]), 0)
            preds.append(pred)
            hist = pd.concat([hist, pd.Series([pred], index=[fd])])

        actual = series.iloc[-HORIZON:]
        mae, rmse, mape = calc_metrics(actual.values, preds)
        xgb_results[state] = {
            "forecast": pd.Series(preds, index=val.index),
            "actual": actual, "mae": mae, "rmse": rmse, "mape": mape,
        }
        xgb_models[state] = m
        print(f"  [{i+1:02d}/{len(states)}] {state:<20} mape={mape:.1f}%")
    except Exception as e:
        print(f"  [{i+1:02d}/{len(states)}] {state:<20} failed: {e}")
        xgb_results[state] = None

joblib.dump(xgb_results, "artifacts/xgb_results.pkl")
joblib.dump(xgb_models,  "artifacts/xgb_models.pkl")
print("xgboost saved")


# ── lstm ──────────────────────────────────────────────────
print("\n--- lstm (~5-8 min) ---")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 12

def _seq(data, lb):
    X, y = [], []
    for j in range(lb, len(data)):
        X.append(data[j - lb:j, 0])
        y.append(data[j, 0])
    return np.array(X), np.array(y)

def _lstm(lb):
    net = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lb, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    net.compile(optimizer="adam", loss="mse")
    return net

lstm_results = {}
lstm_scalers = {}

for i, state in enumerate(states):
    raw = weekly_data[state].values.reshape(-1, 1)
    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(raw)
        train_sc = scaled[:-HORIZON]
        val_act  = weekly_data[state].iloc[-HORIZON:]

        X, y = _seq(train_sc, LOOKBACK)
        X    = X.reshape(X.shape[0], LOOKBACK, 1)

        model = _lstm(LOOKBACK)
        model.fit(
            X, y, epochs=100, batch_size=16,
            validation_split=0.1, verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        )

        seq   = train_sc[-LOOKBACK:].copy()
        preds_sc = []
        for _ in range(HORIZON):
            p = float(np.clip(model.predict(seq.reshape(1, LOOKBACK, 1), verbose=0)[0, 0], 0, 1))
            preds_sc.append(p)
            seq = np.append(seq[1:], [[p]], axis=0)

        preds = np.clip(scaler.inverse_transform(np.array(preds_sc).reshape(-1, 1)).flatten(), 0, None)
        fc    = pd.Series(preds, index=val_act.index)

        mae, rmse, mape = calc_metrics(val_act.values, preds)
        lstm_results[state] = {
            "forecast": fc, "actual": val_act,
            "mae": mae, "rmse": rmse, "mape": mape,
        }
        lstm_scalers[state] = scaler
        model.save(f"artifacts/lstm_{state.replace(' ', '_')}.keras")
        print(f"  [{i+1:02d}/{len(states)}] {state:<20} mape={mape:.1f}%")
    except Exception as e:
        print(f"  [{i+1:02d}/{len(states)}] {state:<20} failed: {e}")
        lstm_results[state] = None

joblib.dump(lstm_results, "artifacts/lstm_results.pkl")
joblib.dump(lstm_scalers, "artifacts/lstm_scalers.pkl")
print("lstm saved")


# ── verify ────────────────────────────────────────────────
print("\n--- checking saved files ---")
for fname in ["arima_results", "prophet_results", "xgb_results", "lstm_results"]:
    path   = f"artifacts/{fname}.pkl"
    status = "ok" if os.path.exists(path) else "MISSING"
    print(f"  {fname}: {status}")

print("\ndone. next: python model_selection.py")