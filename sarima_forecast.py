import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from forecast_metrics import calculate_metrics

warnings.filterwarnings("ignore")

HORIZON = 8
ORDER          = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 0, 52)

# ── load ──────────────────────────────────────────────────
state_datasets = joblib.load("artifacts/state_datasets.pkl")
weekly_data    = joblib.load("artifacts/weekly_data.pkl")
states         = sorted(state_datasets.keys())
print(f"loaded {len(states)} states")

# ── quick stationarity check (alabama) ───────────────────
al_series  = weekly_data["Alabama"]
adf_stat, adf_p, *_ = adfuller(al_series.dropna())
print(f"\nalabama ADF: stat={adf_stat:.3f}  p={adf_p:.4f}")
print("stationary" if adf_p < 0.05 else "not stationary → d=1 handles this")

# ── train ─────────────────────────────────────────────────
print(f"\nfitting SARIMA{ORDER}x{SEASONAL_ORDER} for all states (~3-5 min)\n")

results = {}
models  = {}

for i, state in enumerate(states):
    series = weekly_data[state]
    train  = series.iloc[:-HORIZON]
    val    = series.iloc[-HORIZON:]

    try:
        fit = SARIMAX(
            train,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        preds = fit.forecast(steps=HORIZON)
        preds.index = val.index

        mae, rmse, mape = calculate_metrics(val, preds)

        results[state] = {
            "forecast":  preds,
            "actual":    val,
            "mae":       mae,
            "rmse":      rmse,
            "mape":      mape,
            "train_end": train.index[-1],
        }
        models[state] = fit
        print(f"  [{i+1:02d}/{len(states)}] {state:<20} rmse=${rmse:>14,.0f}  mape={mape:.1f}%")

    except Exception as e:
        print(f"  [{i+1:02d}/{len(states)}] {state:<20} failed: {e}")
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
fig.suptitle("SARIMA — forecast vs actual", fontsize=12)

for ax, st in zip(axes.flat, plot_states):
    if not results.get(st):
        ax.set_visible(False)
        continue
    r   = results[st]
    ser = weekly_data[st]
    hist = ser.iloc[-30:-HORIZON]

    ax.plot(hist.index,           hist.values / 1e6,           color="#4C72B0", lw=1.2, label="history")
    ax.plot(r["actual"].index,    r["actual"].values / 1e6,    color="green",   lw=1.5, label="actual")
    ax.plot(r["forecast"].index,  r["forecast"].values / 1e6,  color="red",     lw=1.5, ls="--", label="sarima")
    ax.set_title(f"{st}  ({r['mape']:.1f}%)")
    ax.set_ylabel("sales ($M)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=28)

plt.tight_layout()
plt.savefig("artifacts/03_arima_forecast.png", dpi=120, bbox_inches="tight")
plt.close()
print("plot saved → artifacts/03_arima_forecast.png")

# ── save ──────────────────────────────────────────────────
joblib.dump(results, "artifacts/arima_results.pkl")
joblib.dump(models,  "artifacts/arima_models.pkl")
print("saved → artifacts/arima_results.pkl")
print(f"\ndone. avg mape: {avg_mape:.1f}%  next: python prophet_forecast.py")