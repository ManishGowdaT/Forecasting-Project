import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

HORIZON = 8

weekly_data = joblib.load("artifacts/weekly_data.pkl")
states      = sorted(weekly_data.keys())
print(f"rebuilding features for {len(states)} states...")


def _flag_holiday(date):
    m, d = date.month, date.day
    if m == 1  and d <= 7:  return 1
    if m == 7  and d <= 7:  return 1
    if m == 11 and d >= 22: return 1
    if m == 12 and d >= 20: return 1
    return 0


def build(series, state):
    out = pd.DataFrame({"y": series}, index=series.index)

    out["t-1"]  = out["y"].shift(1)
    out["t-7"]  = out["y"].shift(7)
    out["t-30"] = out["y"].shift(30)

    base = out["y"].shift(1)
    out["rolling_mean_4"]  = base.rolling(4).mean()
    out["rolling_mean_12"] = base.rolling(12).mean()
    out["rolling_std_4"]   = base.rolling(4).std()

    out["day_of_week"]  = out.index.dayofweek
    out["week_of_year"] = out.index.isocalendar().week.astype(int)
    out["month"]        = out.index.month
    out["quarter"]      = out.index.quarter
    out["year"]         = out.index.year
    out["holiday_flag"] = [_flag_holiday(d) for d in out.index]
    out["state"]        = state
    return out.dropna()


state_datasets = {}
for st in states:
    feat  = build(weekly_data[st], st).sort_index()
    state_datasets[st] = {
        "weekly_series": weekly_data[st],
        "full_features": feat,
        "train": feat.iloc[:-HORIZON],
        "val":   feat.iloc[-HORIZON:],
    }

joblib.dump(state_datasets, "artifacts/state_datasets.pkl")
print("saved → artifacts/state_datasets.pkl")

# quick sanity check
sample = state_datasets["Alabama"]["full_features"]
print(f"\ncolumns: {list(sample.columns)}")
print(f"\nalabama sample:")
print(sample.head(2)[["y","t-1","t-7","t-30","rolling_mean_4","holiday_flag"]].to_string())