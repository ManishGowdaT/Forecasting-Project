import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

HORIZON = 8

df = pd.read_excel("data/sales_data.xlsx")
df["Date"] = pd.to_datetime(df["Date"])
states = sorted(df["State"].unique())
print(f"loaded {len(df):,} rows, {len(states)} states")


def to_weekly(sdf):
    s = sdf.set_index("Date")["Total"].resample("W").sum()
    s = s.replace(0, np.nan).interpolate(method="linear")
    return s.ffill().bfill()


def build_features(series, state):
    out = pd.DataFrame({"y": series}, index=series.index)
    out["lag_1"]  = out["y"].shift(1)
    out["lag_4"]  = out["y"].shift(4)
    out["lag_52"] = out["y"].shift(52)

    base = out["y"].shift(1)
    out["rolling_mean_4"]  = base.rolling(4).mean()
    out["rolling_mean_12"] = base.rolling(12).mean()
    out["rolling_std_4"]   = base.rolling(4).std()

    out["week_of_year"] = out.index.isocalendar().week.astype(int)
    out["month"]        = out.index.month
    out["quarter"]      = out.index.quarter
    out["year"]         = out.index.year
    out["is_holiday"]   = 0
    out["state"]        = state
    return out.dropna()


print("resampling to weekly...")
weekly_data = {st: to_weekly(df[df["State"] == st]) for st in states}

print("building features...")
state_datasets = {}
for st in states:
    feat  = build_features(weekly_data[st], st).sort_index()
    state_datasets[st] = {
        "weekly_series": weekly_data[st],
        "full_features": feat,
        "train": feat.iloc[:-HORIZON],
        "val":   feat.iloc[-HORIZON:],
    }

joblib.dump(state_datasets, "artifacts/state_datasets.pkl")
joblib.dump(weekly_data,    "artifacts/weekly_data.pkl")
print("saved → artifacts/state_datasets.pkl")
print("saved → artifacts/weekly_data.pkl")
print("done. next: python train_models.py")