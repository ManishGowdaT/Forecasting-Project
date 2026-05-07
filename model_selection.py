import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── load ──────────────────────────────────────────────────
weekly_data = joblib.load("artifacts/weekly_data.pkl")
all_results = {
    "ARIMA":   joblib.load("artifacts/arima_results.pkl"),
    "Prophet": joblib.load("artifacts/prophet_results.pkl"),
    "XGBoost": joblib.load("artifacts/xgb_results.pkl"),
    "LSTM":    joblib.load("artifacts/lstm_results.pkl"),
}
states = sorted(weekly_data.keys())
print(f"comparing 4 models across {len(states)} states\n")

# ── pick best per state ───────────────────────────────────
selection = {}
rows      = []

for state in states:
    best_name, best_mape = None, float("inf")
    row = {"State": state}

    for name, res in all_results.items():
        r = res.get(state)
        if r is None:
            row[name] = None
            continue
        row[name] = round(r["mape"], 1)
        if r["mape"] < best_mape:
            best_mape = r["mape"]
            best_name = name

    row["winner"] = best_name
    row["mape"]   = round(best_mape, 1)
    rows.append(row)

    selection[state] = {
        "best_model": best_name,
        "best_mape":  best_mape,
        "forecast":   all_results[best_name][state]["forecast"],
        "actual":     all_results[best_name][state]["actual"],
        "rmse":       all_results[best_name][state]["rmse"],
    }

comp = pd.DataFrame(rows).set_index("State")

# ── comparison table ──────────────────────────────────────
print(f"{'state':<22} {'arima':>8} {'prophet':>9} {'xgboost':>9} {'lstm':>7}  winner")
print("-" * 68)
for state in states:
    r       = comp.loc[state]
    arima   = f"{r['ARIMA']:.1f}%"   if r["ARIMA"]   is not None else "  N/A"
    prophet = f"{r['Prophet']:.1f}%" if r["Prophet"] is not None else "  N/A"
    xgb     = f"{r['XGBoost']:.1f}%" if r["XGBoost"] is not None else "  N/A"
    lstm    = f"{r['LSTM']:.1f}%"    if r["LSTM"]    is not None else "  N/A"
    print(f"{state:<22} {arima:>8} {prophet:>9} {xgb:>9} {lstm:>7}  → {r['winner']} ({r['mape']:.1f}%)")

# ── win counts ────────────────────────────────────────────
print()
wins = {}
for info in selection.values():
    w = info["best_model"]
    wins[w] = wins.get(w, 0) + 1

for name, count in sorted(wins.items(), key=lambda x: -x[1]):
    print(f"  {name:<10} {'█' * count} ({count})")

top = max(wins, key=wins.get)
print(f"\noverall winner: {top}  ({wins[top]}/{len(states)} states)")

# ── per-state summary ─────────────────────────────────────
print(f"\n{'state':<22} {'model':<12} {'mape':>7}  {'avg wk forecast':>18}")
print("-" * 64)
for state in states:
    info = selection[state]
    avg  = info["forecast"].mean()
    print(f"{state:<22} {info['best_model']:<12} {info['best_mape']:>6.1f}%  ${avg:>16,.0f}")

# ── charts ────────────────────────────────────────────────
palette = {"Prophet": "#FF7F0E", "ARIMA": "#1F77B4", "XGBoost": "#9467BD", "LSTM": "#D62728"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("model selection results", fontsize=12)

# bar chart
names  = list(wins.keys())
counts = [wins[n] for n in names]
axes[0].bar(names, counts, color=[palette.get(n, "gray") for n in names])
axes[0].set_title("states won")
axes[0].set_ylabel("count")
for j, (n, c) in enumerate(zip(names, counts)):
    axes[0].text(j, c + 0.05, str(c), ha="center", fontweight="bold")

# boxplot
mape_lists = {n: [r["mape"] for r in res.values() if r] for n, res in all_results.items()}
axes[1].boxplot(list(mape_lists.values()), labels=list(mape_lists.keys()), patch_artist=True,
                boxprops=dict(facecolor="lightblue"))
axes[1].axhline(20, color="red", ls="--", alpha=0.5)
axes[1].set_title("mape distribution")
axes[1].set_ylabel("mape %")
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("artifacts/07_model_selection.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nchart saved → artifacts/07_model_selection.png")

# ── save ──────────────────────────────────────────────────
joblib.dump(selection, "artifacts/best_model_selection.pkl")
comp.to_csv("artifacts/model_comparison.csv")
print("saved → artifacts/best_model_selection.pkl")
print("saved → artifacts/model_comparison.csv")

avg_best = np.mean([v["best_mape"] for v in selection.values()])
print(f"\ndone. overall avg best mape: {avg_best:.1f}%  next: python forecast_api.py")