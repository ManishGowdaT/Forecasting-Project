import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

HORIZON  = 8
LOOKBACK = 12


def _sequences(scaled, lookback):
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y)


def _build_model(lookback):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


# ── load ──────────────────────────────────────────────────
weekly_data = joblib.load("artifacts/weekly_data.pkl")
states      = sorted(weekly_data.keys())
print(f"loaded {len(states)} states — fitting lstm (~5-8 min)\n")

results = {}
scalers = {}
models  = {}

for i, state in enumerate(states):
    raw = weekly_data[state].values.reshape(-1, 1)

    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(raw)

        train_scaled = scaled[:-HORIZON]
        val_actual   = weekly_data[state].iloc[-HORIZON:]

        X, y = _sequences(train_scaled, LOOKBACK)
        X    = X.reshape(X.shape[0], LOOKBACK, 1)

        model = _build_model(LOOKBACK)
        model.fit(
            X, y,
            epochs=100,
            batch_size=16,
            validation_split=0.1,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        )

        seq  = train_scaled[-LOOKBACK:].copy()
        preds_scaled = []
        for _ in range(HORIZON):
            p = float(model.predict(seq.reshape(1, LOOKBACK, 1), verbose=0)[0, 0])
            p = np.clip(p, 0, 1)
            preds_scaled.append(p)
            seq = np.append(seq[1:], [[p]], axis=0)

        preds = np.clip(
            scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten(),
            0, None,
        )
        fc_series = pd.Series(preds, index=val_actual.index)

        mae  = mean_absolute_error(val_actual, fc_series)
        rmse = float(np.sqrt(mean_squared_error(val_actual, fc_series)))
        mape = float(np.mean(np.abs((val_actual.values - preds) / val_actual.values)) * 100)

        results[state] = {
            "forecast": fc_series,
            "actual":   val_actual,
            "mae":      mae,
            "rmse":     rmse,
            "mape":     mape,
        }
        scalers[state] = scaler
        models[state]  = model

        model.save(f"artifacts/lstm_{state.replace(' ', '_')}.keras")
        print(f"  [{i+1:02d}/43] {state:<20} rmse=${rmse:>14,.0f}  mape={mape:.1f}%")

    except Exception as e:
        print(f"  [{i+1:02d}/43] {state:<20} failed: {e}")
        results[state] = None

# ── summary ───────────────────────────────────────────────
valid   = {k: v for k, v in results.items() if v}
by_mape = sorted(valid.items(), key=lambda x: x[1]["mape"])

print(f"\n{'state':<22} {'mape':>8} {'rmse':>18}")
print("-" * 52)
for st, r in by_mape:
    print(f"{st:<22} {r['mape']:>7.1f}%  ${r['rmse']:>15,.0f}")

avg_mape = np.mean([r["mape"] for r in valid.values()])
avg_rmse = np.mean([r["rmse"] for r in valid.values()])
print(f"\navg mape: {avg_mape:.1f}%  avg rmse: ${avg_rmse:,.0f}")

# ── plot ──────────────────────────────────────────────────
plot_states = ["California", "Texas", "New York", "Alabama"]
fig, axes   = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("LSTM — forecast vs actual", fontsize=12)

for ax, st in zip(axes.flat, plot_states):
    if not results[st]:
        continue
    r    = results[st]
    ser  = weekly_data[st]
    hist = ser.iloc[-30:-HORIZON]

    ax.plot(hist.index,          hist.values / 1e6,          color="#4C72B0", lw=1.2, label="history")
    ax.plot(r["actual"].index,   r["actual"].values / 1e6,   color="green",   lw=1.5, label="actual")
    ax.plot(r["forecast"].index, r["forecast"].values / 1e6, color="red",     lw=1.5, ls="--", label="lstm")
    ax.set_title(f"{st}  ({r['mape']:.1f}%)")
    ax.set_ylabel("sales ($M)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=28)

plt.tight_layout()
plt.savefig("artifacts/06_lstm_forecast.png", dpi=120, bbox_inches="tight")
plt.close()
print("plot saved → artifacts/06_lstm_forecast.png")

# ── save ──────────────────────────────────────────────────
joblib.dump(results, "artifacts/lstm_results.pkl")
joblib.dump(scalers, "artifacts/lstm_scalers.pkl")
print("saved → artifacts/lstm_results.pkl")
print(f"\ndone. avg mape: {avg_mape:.1f}%  next: python model_selection.py")