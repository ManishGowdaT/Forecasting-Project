import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")

# ── load ──────────────────────────────────────────────────
df = pd.read_excel("data/sales_data.xlsx")
df["Date"] = pd.to_datetime(df["Date"])

print(f"Shape     : {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"Columns   : {list(df.columns)}")
print(f"Dates     : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"States    : {df['State'].nunique()}")
print(f"Category  : {df['Category'].unique()}")
print(f"Sales rng : ${df['Total'].min():,.0f} – ${df['Total'].max():,.0f}")
print()
print(df.head().to_string())


# ── nulls ─────────────────────────────────────────────────
print("\n--- missing values ---")
null_counts = df.isnull().sum()
print(null_counts)
if null_counts.sum() == 0:
    print("none found.")
else:
    print("will handle in feature pipeline.")


# ── date gaps ─────────────────────────────────────────────
print("\n--- date spacing (alabama sample) ---")
al = df[df["State"] == "Alabama"].sort_values("Date")
gaps = al["Date"].diff().dropna().dt.days
print(f"gap range: {gaps.min()}d – {gaps.max()}d (irregular)")
print("will resample to weekly in the feature step.")


# ── resample helper ───────────────────────────────────────
def to_weekly(sdf):
    s = sdf.set_index("Date")["Total"].resample("W").sum()
    s = s.replace(0, np.nan).interpolate(method="linear")
    return s

al_w = to_weekly(al)
print(f"\nalabama: {len(al)} raw rows → {len(al_w)} weekly")
print(f"weekly span: {al_w.index[0].date()} → {al_w.index[-1].date()}")
print(al_w.head())


# ── trend plots ───────────────────────────────────────────
plot_states = ["California", "Texas", "New York", "Florida"]
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
date_span = f"{df['Date'].min():%Y}–{df['Date'].max():%Y}"
fig.suptitle(f"Weekly Sales — Beverages ({date_span})", fontsize=13, fontweight="bold")

for ax, state in zip(axes.flat, plot_states):
    w = to_weekly(df[df["State"] == state])
    ax.plot(w.index, w.values / 1e6, color="#4C72B0", linewidth=1.2)
    ax.set_title(state, fontsize=10)
    ax.set_ylabel("Sales ($M)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", rotation=28)

plt.tight_layout()
plt.savefig("artifacts/01_sales_trends.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nchart saved → artifacts/01_sales_trends.png")


# ── state summary ─────────────────────────────────────────
print("\n--- top 10 states by avg sales ---")
summary = (
    df.groupby("State")["Total"]
    .agg(mean="mean", std="std", min="min", max="max", n="count")
    .sort_values("mean", ascending=False)
)
summary["mean"] = summary["mean"].map("${:,.0f}".format)
summary["max"]  = summary["max"].map("${:,.0f}".format)
print(summary.head(10).to_string())

print("\ndone. next: python prepare_data.py")