import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import yfinance as yf
from plottable import Table, ColumnDefinition

warnings.filterwarnings("ignore", category=FutureWarning)

BASE        = Path(__file__).resolve().parent
DATA_DIR    = BASE / "clean_data"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

mpl.rcParams["figure.dpi"] = 150

# ------------------------------------------------------------------ #
# 1. Data
# ------------------------------------------------------------------ #
prices = pd.read_parquet(DATA_DIR / "clean_stock_prices.parquet")
caps   = pd.read_parquet(DATA_DIR / "clean_mkt_cap.parquet")

prices.index = pd.to_datetime(prices.index)
caps.index   = pd.to_datetime(caps.index)

# Align indexes
caps = caps.reindex(prices.index).ffill()

# Daily returns
rets = prices.pct_change().dropna(how="all")

# S&P500 benchmark
spx = (
    yf.download("^GSPC", start=str(rets.index.min().date()),
                end=str(rets.index.max().date()),
                auto_adjust=True, progress=False)["Close"]
      .pct_change()
      .dropna()
)

# ------------------------------------------------------------------ #
# 2. Utility Functions
# ------------------------------------------------------------------ #
ANNUAL_RF = 0.0492
TRADING_DAYS = 252
RF_DAILY = ANNUAL_RF / TRADING_DAYS

def ann_ret(x):
    return float((1 + x).prod() ** (TRADING_DAYS / len(x)) - 1) if len(x) else np.nan

def ann_vol(x):
    return float(x.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(x) else np.nan

def sharpe(x):
    r, v = ann_ret(x), ann_vol(x)
    return float((r - ANNUAL_RF) / v) if v and v > 0 else np.nan

def backtest_momentum(prices, lookback, top_frac=None, cash=0.05):
    """
    Daily momentum with monthly rebalance and fixed weights during the month.
    top_frac=None -> Time-Series (TS): selects assets with return > 0
    top_frac=0.xx -> Cross-Sectional (CS): picks top frac of assets
    """
    rets = prices.pct_change().dropna()
    last_td = (
        pd.Series(1, index=rets.index)
          .groupby(rets.index.to_period("M"))
          .tail(1)
          .index
    )

    all_daily, weights_hist = [], []
    cols = prices.columns

    for i in range(len(last_td) - 1):
        start, end = last_td[i], last_td[i+1]
        month_rets = rets.loc[(rets.index > start) & (rets.index <= end)]
        if month_rets.empty: continue

        # momentum signal computed at rebalance date
        signal = (prices.loc[start] / prices.shift(lookback).loc[start] - 1)

        if top_frac is None:  # TS
            selected = signal[signal > 0].index
        else:                 # CS
            k = max(1, int(len(signal) * top_frac))
            selected = signal.nlargest(k).index

        # target weights
        if len(selected) == 0:
            target = pd.Series(0.0, index=cols)
        else:
            target = pd.Series((1 - cash)/len(selected), index=selected)
            target = target.reindex(cols).fillna(0.0)
        target["CASH"] = cash

        # apply fixed weights during the month
        for dt, r in month_rets.iterrows():
            port_ret = float(np.dot(target.drop("CASH", errors="ignore"),
                                    r.reindex(cols).fillna(0)))
            all_daily.append(pd.Series(port_ret, index=[dt]))
            weights_hist.append(target.drop("CASH").rename(dt))

    daily_series = pd.concat(all_daily).sort_index().squeeze()
    weights_df   = pd.DataFrame(weights_hist)
    return daily_series, weights_df

# ------------------------------------------------------------------ #
# 3. EW and VW (with drift + monthly rebalance)
# ------------------------------------------------------------------ #
last_td = (
    pd.Series(1, index=rets.index)
      .groupby(rets.index.to_period("M"))
      .tail(1)
      .index
)

ew_daily, vw_daily = [], []
ew_w_hist, vw_w_hist = [], []

cols = rets.columns
N = len(cols)

for i in range(len(last_td) - 1):
    start, end = last_td[i], last_td[i+1]
    month_rets = rets.loc[(rets.index > start) & (rets.index <= end)]
    if month_rets.empty: continue

    # EW target
    ew_target = pd.Series(1.0/N, index=cols)

    # VW target (from market cap)
    mcaps = caps.loc[start].reindex(cols)
    vw_target = mcaps / mcaps.sum()
    vw_target = vw_target.fillna(1.0/N)

    w_ew, w_vw = ew_target.copy(), vw_target.copy()
    for dt, r in month_rets.iterrows():
        g_ew = float(np.dot(w_ew.values, (1 + r.values)))
        g_vw = float(np.dot(w_vw.values, (1 + r.values)))
        ew_daily.append(pd.Series(g_ew - 1.0, index=[dt]))
        vw_daily.append(pd.Series(g_vw - 1.0, index=[dt]))

        w_ew = w_ew * (1 + r.values)
        if w_ew.sum() > 0: w_ew /= w_ew.sum()
        ew_w_hist.append(w_ew.rename(dt))

        w_vw = w_vw * (1 + r.values)
        if w_vw.sum() > 0: w_vw /= w_vw.sum()
        vw_w_hist.append(w_vw.rename(dt))

ew_ret = pd.concat(ew_daily).sort_index()
vw_ret = pd.concat(vw_daily).sort_index()
ew_weights = pd.DataFrame(ew_w_hist)
vw_weights = pd.DataFrame(vw_w_hist)

# ------------------------------------------------------------------ #
# 4. Momentum Portfolios
# ------------------------------------------------------------------ #
ts_ret, ts_weights = backtest_momentum(prices, lookback=12, cash=0.05)
cs_ret, cs_weights = backtest_momentum(prices, lookback=12, cash=0.05, top_frac=0.20)

# ------------------------------------------------------------------ #
# 5. Performance Table
# ------------------------------------------------------------------ #
idx = ew_ret.index.intersection(vw_ret.index).intersection(spx.index)
idx = idx.intersection(ts_ret.index).intersection(cs_ret.index)

portfolios = {
    "EW": ew_ret.reindex(idx),
    "VW": vw_ret.reindex(idx),
    "TS": ts_ret.reindex(idx),
    "CS": cs_ret.reindex(idx),
    "S&P500": spx.reindex(idx)
}

perf_tbl = pd.DataFrame({
    k: {
        "CAGR": ann_ret(v),
        "Vol": ann_vol(v),
        "Sharpe": sharpe(v)
    }
    for k, v in portfolios.items()
}).T.round(4)

# Plottable table
perf_fmt = perf_tbl.reset_index().rename(columns={"index":"Strategy"})
col_defs = [
    ColumnDefinition("Strategy", title="Strategy"),
    ColumnDefinition("CAGR", formatter="{:.2%}"),
    ColumnDefinition("Vol", formatter="{:.2%}"),
    ColumnDefinition("Sharpe", formatter="{:.2f}")
]
fig, ax = plt.subplots(figsize=(8,2+0.35*len(perf_fmt)))
Table(perf_fmt, column_definitions=col_defs, ax=ax)
fig.savefig(RESULTS_DIR/"perf_table.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ #
# 6. Capital Growth Curve
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(9,5))
for k, v in portfolios.items():
    ((1+v).cumprod()).plot(ax=ax, label=k, lw=1.2)
ax.set_title("Growth of $1 — Strategy Comparison")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR/"comparison_curve.png")
plt.close(fig)

# ------------------------------------------------------------------ #
# 7. Weight Charts
# ------------------------------------------------------------------ #
def plot_weights(weights_df, name):
    fig, ax = plt.subplots(figsize=(10,6))
    weights_df.plot.area(ax=ax, linewidth=0)
    ax.set_title(f"Weight Evolution — {name}")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", bbox_to_anchor=(1,1), fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR/f"weights_{name}.png", dpi=150)
    plt.close(fig)

plot_weights(ew_weights, "EW")
plot_weights(vw_weights, "VW")
plot_weights(ts_weights, "TS")
plot_weights(cs_weights, "CS")

# ------------------------------------------------------------------ #
# 8. Console
# ------------------------------------------------------------------ #
print("\n===== Performance =====")
print(perf_tbl.to_markdown())
print(f"\nFiles saved in {RESULTS_DIR.resolve()}")
