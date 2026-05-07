import joblib
import warnings

warnings.filterwarnings("ignore")

weekly_data = joblib.load("artifacts/weekly_data.pkl")
all_results = {
    "ARIMA":   joblib.load("artifacts/arima_results.pkl"),
    "Prophet": joblib.load("artifacts/prophet_results.pkl"),
    "XGBoost": joblib.load("artifacts/xgb_results.pkl"),
    "LSTM":    joblib.load("artifacts/lstm_results.pkl"),
}
states = sorted(weekly_data.keys())

selection = {}
for state in states:
    best_name, best_mape = None, float("inf")
    for name, res in all_results.items():
        r = res.get(state)
        if r and r["mape"] < best_mape:
            best_mape = r["mape"]
            best_name = name
    selection[state] = {
        "best_model": best_name,
        "best_mape":  best_mape,
        "forecast":   all_results[best_name][state]["forecast"],
        "actual":     all_results[best_name][state]["actual"],
        "rmse":       all_results[best_name][state]["rmse"],
    }

joblib.dump(selection, "artifacts/best_model_selection.pkl")
print("saved → artifacts/best_model_selection.pkl\n")
for state, info in selection.items():
    print(f"  {state:<22} {info['best_model']}  ({info['best_mape']:.1f}%)")