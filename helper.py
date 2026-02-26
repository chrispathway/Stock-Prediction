"""
Helper functions for plotting and wealth curves.
Kept separate so main.py stays focused on the ML pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def wealth_curves(val_df, pred_returns, start_money=100):
    """Compute Buy & Hold vs ML strategy wealth over time."""
    close = val_df["Close"]
    buy_hold = start_money * (close / close.iloc[0])
    actual_returns = close.pct_change().fillna(0)
    invest = (pred_returns > 0).astype(float)
    strategy_returns = invest * actual_returns
    ml_wealth = start_money * (1 + strategy_returns).cumprod()
    return buy_hold, ml_wealth


def plot_results(df, pred_returns, train_size):
    """Static plot: actual vs predicted price + wealth comparison."""
    val_df = df.iloc[train_size:]
    close = val_df["Close"]
    pred_price = close * (1 + pred_returns)
    buy_hold, ml_wealth = wealth_curves(val_df, pred_returns)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(val_df.index, close, label="Actual", color="steelblue")
    ax1.plot(val_df.index, pred_price, label="Predicted", color="coral", linestyle="--")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(val_df.index, buy_hold, label="Buy & Hold", color="green")
    ax2.plot(val_df.index, ml_wealth, label="ML strategy", color="purple")
    ax2.set_ylabel("Wealth ($)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def animate_wealth_curves(df, pred_returns, train_size, frames=120, interval=50, save_path=None):
    """Animate wealth curves drawing over time (for video/GIF)."""
    val_df = df.iloc[train_size:]
    buy_hold, ml_wealth = wealth_curves(val_df, pred_returns)
    dates = val_df.index
    n = len(dates)
    step = max(1, n // frames)
    indices = np.arange(0, n, step)
    if indices[-1] != n - 1:
        indices = np.append(indices, n - 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(dates[0], dates[-1])
    y_min = min(buy_hold.min(), ml_wealth.min())
    y_max = max(buy_hold.max(), ml_wealth.max())
    margin = (y_max - y_min) * 0.1 or 10
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_ylabel("Wealth ($)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.set_title("Wealth over time — $100 initial")
    line_bh, = ax.plot([], [], color="green", linewidth=2, label="Buy & Hold")
    line_ml, = ax.plot([], [], color="purple", linewidth=2, label="ML strategy")
    ax.legend(loc="upper left")

    def init():
        line_bh.set_data([], [])
        line_ml.set_data([], [])
        return line_bh, line_ml

    def update(frame_idx):
        i = indices[frame_idx]
        line_bh.set_data(dates[: i + 1], buy_hold.iloc[: i + 1])
        line_ml.set_data(dates[: i + 1], ml_wealth.iloc[: i + 1])
        return line_bh, line_ml

    anim = FuncAnimation(
        fig, update, init_func=init, frames=len(indices),
        interval=interval, blit=True, repeat=True
    )
    if save_path:
        try:
            anim.save(save_path, writer="pillow", fps=1000 // max(1, interval))
            print(f"  Saved animation to {save_path}")
        except Exception as e:
            print(f"  Could not save (install pillow for GIF): {e}")
    plt.tight_layout()
    plt.show()
    return anim
