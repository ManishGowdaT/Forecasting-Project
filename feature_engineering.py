import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

HORIZON = 8   # weeks we want to forecast

# ── load ──────────────────────────────────────────────────
df = pd.read_excel("data/sales_data.xlsx")
df["Date"] = pd.to_datetime(df["Date"])
states = sorted(df["State"].unique())
print(f"loaded {len(df):,} rows across {len(states)} states")


# ── weekly resampling ─────────────────────────────────────
# raw dates are irregular (1–91 day gaps), models need fixed steps
def to_weekly(sdf):
    s = sdf.set_index("Date")["Total"].resample("W").sum()
    s = s.replace(0, np.nan).interpolate(method="linear")
    return s.ffill().bfill()

weekly_data = {st: to_weekly(df[df["State"] == st]) for st in states}

sample = weekly_data["Alabama"]
print(f"weekly rows per state: {len(sample)}")
print(f"span: {sample.index[0].date()} → {sample.index[-1].date()}")


# ── feature builder ───────────────────────────────────────
_holiday_months = {1, 7, 11, 12}   # new year, independence, thanksgiving, christmas

def _is_holiday(date):
    if date.month not in _holiday_months:
        return 0
    return 1 if date.isocalendar().week % 4 == 0 else 0

def build_features(series, state):
    out = pd.DataFrame({"y": series}, index=series.index)

    # lags
    out["lag_1"]  = out["y"].shift(1)
    out["lag_4"]  = out["y"].shift(4)
    out["lag_52"] = out["y"].shift(52)

    # rolling stats — shift(1) so we never leak current week
    base = out["y"].shift(1)
    out["rolling_mean_4"]  = base.rolling(4).mean()
    out["rolling_mean_12"] = base.rolling(12).mean()
    out["rolling_std_4"]   = base.rolling(4).std() 

    # calendar
    out["week_of_year"] = out.index.isocalendar().week.astype(int)
    out["month"]        = out.index.month
    out["quarter"]      = out.index.quarter
    out["year"]         = out.index.year
    out["is_holiday"]   = [_is_holiday(d) for d in out.index]

    out["state"] = state
    return out.dropna()


# ── build for all states ──────────────────────────────────
feat_parts = [build_features(weekly_data[st], st) for st in states]
all_feats  = pd.concat(feat_parts)

print(f"features done — {len(all_feats):,} rows total")
print(f"cols: {list(all_feats.columns)}")

al_sample = all_feats[all_feats["state"] == "Alabama"].head(3)
print(f"\nalabama sample:\n{al_sample.to_string()}")


# ── train / val split ─────────────────────────────────────
# time-based only — no shuffling, no leakage
state_datasets = {}

for st in states:
    feat = all_feats[all_feats["state"] == st].copy().sort_index()
    state_datasets[st] = {
        "weekly_series": weekly_data[st],
        "full_features": feat,
        "train": feat.iloc[:-HORIZON],
        "val":   feat.iloc[-HORIZON:],
    }

al = state_datasets["Alabama"]
print(f"\nalabama split:")
print(f"  train: {len(al['train'])} rows → ends {al['train'].index[-1].date()}")
print(f"  val:   {len(al['val'])} rows  → {al['val'].index[0].date()} to {al['val'].index[-1].date()}")


# ── plot (alabama) ────────────────────────────────────────
series = al["weekly_series"]
train  = al["train"]
val    = al["val"]

fig, axes = plt.subplots(3, 1, figsize=(13, 10))
fig.suptitle("Alabama — feature preview", fontsize=12)

axes[0].plot(series.index, series.values / 1e6, color="#4C72B0", lw=1)
axes[0].axvspan(val.index[0], val.index[-1], alpha=0.15, color="red", label="val window")
axes[0].set_title("weekly sales ($M)")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.2)

axes[1].plot(train.index, train["lag_1"]  / 1e6, label="lag_1",  alpha=0.85)
axes[1].plot(train.index, train["lag_4"]  / 1e6, label="lag_4",  alpha=0.85)
axes[1].plot(train.index, train["lag_52"] / 1e6, label="lag_52", alpha=0.85)
axes[1].set_title("lag features")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.2)

axes[2].plot(train.index, train["rolling_mean_4"]  / 1e6, label="4wk avg",  lw=1.5)
axes[2].plot(train.index, train["rolling_mean_12"] / 1e6, label="12wk avg", lw=1.5)
axes[2].set_title("rolling means")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.2)

plt.tight_layout()
plt.savefig("artifacts/02_features.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nplot saved → artifacts/02_features.png")


# ── save ──────────────────────────────────────────────────
joblib.dump(state_datasets, "artifacts/state_datasets.pkl")
joblib.dump(weekly_data,    "artifacts/weekly_data.pkl")
print("saved → artifacts/state_datasets.pkl")
print("saved → artifacts/weekly_data.pkl")
print("\ndone. next: python arima_model.py")