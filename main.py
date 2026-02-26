import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

from helper import plot_results, animate_wealth_curves

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
def build_features(df):
    df = df.copy()
    close = df["Close"]
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()
    df["price_vs_ma20"] = close / df["ma_20"]
    df["return_1d"] = close.pct_change()
    df["target"] = close.pct_change().shift(-1)  
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
def run(animate=False, save_animation=None):
    print("Loading data...")
    raw = load_data(TICKER, YEARS)
    print(f"  {len(raw)} days from {raw.index[0].date()} to {raw.index[-1].date()}")

    print("Building features...")
    df = build_features(raw)
    feature_cols = ["ma_20", "ma_50", "price_vs_ma20", "return_1d"]

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

    if animate:
        print("Animating wealth curves...")
        animate_wealth_curves(df, pred_returns, train_size, frames=120, interval=50, save_path=save_animation)

    return model, pred_returns, df


if __name__ == "__main__":
    model, predictions, df = run(
        animate=True,
        save_animation="wealth_animation.gif",
    )
