import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

from helper import (
    plot_results,
    animate_wealth_curves,
    monte_carlo_wealth,
    monte_carlo_paths,
    plot_monte_carlo,
    animate_monte_carlo_paths,
)

# ---- 1. CONFIG ----
TICKER = "SPY"   # S&P 500
YEARS = 10
TRAIN_RATIO = 0.8   
end = "2026-02-23"
# ---- 2. LOAD DATA ----
def load_data(ticker, years):
    end = pd.Timestamp.now().normalize()
    start = end - pd.DateOffset(years=years)
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# ---- 3. FEATURES ----
def _rsi(series, window=14):
    """Relative Strength Index (0–100)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def build_features(df):
    df = df.copy()
    close = df["Close"]
    returns = close.pct_change()

    # Moving averages & trend
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()
    df["price_vs_ma20"] = close / df["ma_20"]
    df["price_vs_ma50"] = close / df["ma_50"]

    # Returns
    df["return_1d"] = returns
    df["return_2d"] = returns.shift(1)

    # Volatility (rolling std of returns; *sqrt(252) would annualize)
    df["vol_20"] = returns.rolling(20).std()

    # Momentum / overbought–oversold
    df["rsi_14"] = _rsi(close, 14)

    df["target"] = returns.shift(-1)
    df = df.dropna()
    return df

# ---- 4. TRAIN / VAL SPLIT ----
def split_train_val(df, feature_cols, train_ratio):
    n = int(len(df) * train_ratio)
    train = df.iloc[:n]
    val = df.iloc[n:] #test
    X_train = train[feature_cols]
    y_train = train["target"]
    X_val = val[feature_cols]
    y_val = val["target"]
    return X_train, y_train, X_val, y_val

# ---- 5. TRAIN MODEL ----
def train_model(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return model, pred

# ---- 6. RUN PIPELINE ----
def run(animate=False, save_animation=None, animate_monte_carlo=False, save_monte_carlo_animation=None):
    print("Loading data...")
    raw = load_data(TICKER, YEARS)
    print(f"  {len(raw)} days from {raw.index[0].date()} to {raw.index[-1].date()}")

    print("Building features...")
    df = build_features(raw)
    feature_cols = [
        "ma_20", "ma_50", "price_vs_ma20", "price_vs_ma50",
        "return_1d", "return_2d", "vol_20", "rsi_14",
    ]

    X_train, y_train, X_val, y_val = split_train_val(df, feature_cols, TRAIN_RATIO)
    train_size = len(X_train)
    print(f"  Train: {train_size}  Val: {len(X_val)}")

    print("Training model...")
    model, pred_returns = train_model(X_train, y_train, X_val, y_val)

    # Direction accuracy: % of ALL days (up + down) where we got the sign right
    actual_direction = (y_val.values > 0)
    pred_direction = (pred_returns > 0)
    direction_accuracy = 100 * (actual_direction == pred_direction).mean()
    pct_pred_up = 100 * pred_direction.mean()
    pct_pred_down = 100 * (1 - pred_direction.mean())
    pct_actual_up = 100 * actual_direction.mean()
    pct_actual_down = 100 * (1 - actual_direction.mean())
    print(f"  Direction accuracy (val): {direction_accuracy:.1f}%  (up and down combined)")
    print(f"  Predicted  UP: {pct_pred_up:.1f}% of days  |  Predicted  DOWN: {pct_pred_down:.1f}% of days")
    print(f"  Actual     UP: {pct_actual_up:.1f}% of days  |  Actual     DOWN: {pct_actual_down:.1f}% of days")

    print("Plotting...")
    plot_results(df, pred_returns, train_size)

    # Monte Carlo: sample from return distribution for a distribution of outcomes
    print("Monte Carlo evaluation (bootstrap return paths)...")
    val_returns = df.iloc[train_size:]["target"].values  # next-day returns in val
    positions = np.sign(pred_returns)  # +1 long, -1 short, 0 flat
    pct_long = 100 * (positions > 0).mean()
    pct_short = 100 * (positions < 0).mean()
    pct_flat = 100 * (positions == 0).mean()
    print(f"  ML positions: long {pct_long:.1f}%  short {pct_short:.1f}%  flat {pct_flat:.1f}%")
    ml_final, bh_final = monte_carlo_wealth(val_returns, positions, n_sim=2000)
    print(f"  ML strategy  — median final wealth: ${np.median(ml_final):.0f}  5th–95th: ${np.percentile(ml_final, 5):.0f} – ${np.percentile(ml_final, 95):.0f}")
    print(f"  Buy & Hold   — median final wealth: ${np.median(bh_final):.0f}  5th–95th: ${np.percentile(bh_final, 5):.0f} – ${np.percentile(bh_final, 95):.0f}")
    plot_monte_carlo(ml_final, bh_final)

    if animate_monte_carlo:
        print("Animating Monte Carlo paths...")
        ml_paths, bh_paths = monte_carlo_paths(val_returns, positions, n_sim=500)
        val_df = df.iloc[train_size:]
        animate_monte_carlo_paths(
            ml_paths, bh_paths, val_df.index,
            n_paths_show=60, frames=150, interval=40,
            save_path=save_monte_carlo_animation,
        )

    if animate:
        print("Animating wealth curves...")
        animate_wealth_curves(df, pred_returns, train_size, frames=120, interval=50, save_path=save_animation)

    return model, pred_returns, df


if __name__ == "__main__":
    model, predictions, df = run(
        animate=True,
        save_animation="wealth_animation.gif",
        animate_monte_carlo=True,
        save_monte_carlo_animation="monte_carlo_paths.gif",
    )
