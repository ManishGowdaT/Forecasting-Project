import logging
import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from forecast_metrics import calculate_metrics

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

HORIZON = 8

# ── load ──────────────────────────────────────────────────
state_datasets = joblib.load("artifacts/state_datasets.pkl")
weekly_data    = joblib.load("artifacts/weekly_data.pkl")
states         = sorted(state_datasets.keys())
print(f"loaded {len(states)} states — fitting prophet (~2-3 min)\n")

results = {}
models  = {}

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

        future   = m.make_future_dataframe(periods=HORIZON, freq="W")
        forecast = m.predict(future)
        preds    = np.clip(forecast["yhat"].iloc[-HORIZON:].values, 0, None)

        mae, rmse, mape = calculate_metrics(val, preds)

        results[state] = {
            "forecast":      pd.Series(preds, index=val.index),
            "actual":        val,
            "mae":           mae,
            "rmse":          rmse,
            "mape":          mape,
            "full_forecast": forecast,
        }
        models[state] = m
        print(f"  [{i+1:02d}/{len(states)}] {state:<20} rmse=${rmse:>14,.0f}  mape={mape:.1f}%")

    except Exception as e:
        print(f"  [{i+1:02d}/{len(states)}] {state:<20} failed: {e}")
        results[state] = None

# ── summary ───────────────────────────────────────────────
valid   = {k: v for k, v in results.items() if v}
if not valid:
    raise RuntimeError("No successful model evaluations; check the training errors above")
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
fig.suptitle("Prophet — forecast vs actual", fontsize=12)

for ax, st in zip(axes.flat, plot_states):
    if not results.get(st):
        ax.set_visible(False)
        continue
    r    = results[st]
    ser  = weekly_data[st]
    hist = ser.iloc[-30:-HORIZON]

    ax.plot(hist.index,          hist.values / 1e6,          color="#4C72B0", lw=1.2, label="history")
    ax.plot(r["actual"].index,   r["actual"].values / 1e6,   color="green",   lw=1.5, label="actual")
    ax.plot(r["forecast"].index, r["forecast"].values / 1e6, color="orange",  lw=1.5, ls="--", label="prophet")
    ax.set_title(f"{st}  ({r['mape']:.1f}%)")
    ax.set_ylabel("sales ($M)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=28)

plt.tight_layout()
plt.savefig("artifacts/04_prophet_forecast.png", dpi=120, bbox_inches="tight")
plt.close()
print("plot saved → artifacts/04_prophet_forecast.png")

# ── save ──────────────────────────────────────────────────
joblib.dump(results, "artifacts/prophet_results.pkl")
joblib.dump(models,  "artifacts/prophet_models.pkl")
print("saved → artifacts/prophet_results.pkl")
print(f"\ndone. avg mape: {avg_mape:.1f}%  next: python xgboost_forecast.py")