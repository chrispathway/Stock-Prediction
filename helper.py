"""
Helper functions for plotting and wealth curves.
Kept separate so main.py stays focused on the ML pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation


def wealth_curves(val_df, pred_returns, start_money=100):
    """
    Compute Buy & Hold vs ML strategy wealth over time. ML can long (+1) or short (-1).

    No lookahead: pred_returns[i] predicts return from val day i to day i+1, so we hold
    that position during day i+1. On the first val day we have no prior prediction → flat.
    """
    close = val_df["Close"]
    buy_hold = start_money * (close / close.iloc[0])
    actual_returns = close.pct_change().fillna(0).values
    pred_returns = np.asarray(pred_returns, dtype=float)
    position_held = np.zeros(len(actual_returns))
    position_held[1:] = np.sign(pred_returns[:-1])
    strategy_returns = position_held * actual_returns
    ml_wealth = start_money * np.cumprod(1 + strategy_returns)
    return buy_hold, pd.Series(ml_wealth, index=val_df.index)


def monte_carlo_wealth(returns_val, positions, n_sim=2000, start_money=100, random_state=42):
    """
    Monte Carlo evaluation: sample many return paths from the empirical distribution.

    positions: +1 long, -1 short, 0 flat (e.g. np.sign(pred_returns)).
    Keeps the strategy's positions fixed; for each simulation we draw
    a random sequence of returns and compute final wealth.

    Returns
    -------
    ml_final : 1d array of length n_sim (ML strategy final wealth per path)
    bh_final : 1d array of length n_sim (buy-and-hold final wealth per path)
    """
    rng = np.random.default_rng(random_state)
    n = len(returns_val)
    returns_val = np.asarray(returns_val, dtype=float)
    position = np.asarray(positions, dtype=float)  # +1, -1, or 0

    ml_final = np.empty(n_sim)
    bh_final = np.empty(n_sim)

    for i in range(n_sim):
        idx = rng.integers(0, n, size=n)
        path = returns_val[idx]
        # ML: long earns +return, short earns -return, flat earns 0
        ml_wealth = start_money * np.prod(1 + position * path)
        bh_wealth = start_money * np.prod(1 + path)
        ml_final[i] = ml_wealth
        bh_final[i] = bh_wealth

    return ml_final, bh_final


def monte_carlo_paths(returns_val, positions, n_sim=500, start_money=100, random_state=42):
    """
    Like monte_carlo_wealth but returns full wealth trajectories for each path.
    positions: +1 long, -1 short, 0 flat.
    """
    rng = np.random.default_rng(random_state)
    n = len(returns_val)
    returns_val = np.asarray(returns_val, dtype=float)
    position = np.asarray(positions, dtype=float)

    ml_paths = np.empty((n_sim, n + 1))
    bh_paths = np.empty((n_sim, n + 1))
    ml_paths[:, 0] = start_money
    bh_paths[:, 0] = start_money

    for i in range(n_sim):
        idx = rng.integers(0, n, size=n)
        path = returns_val[idx]
        ml_paths[i, 1:] = start_money * np.cumprod(1 + position * path)
        bh_paths[i, 1:] = start_money * np.cumprod(1 + path)

    return ml_paths, bh_paths


def animate_monte_carlo_paths(
    ml_paths, bh_paths, dates, n_paths_show=60,
    frames=150, interval=40, save_path=None, start_money=100
):
    """
    Animate Monte Carlo paths: the 'fan' of possible wealth curves grows over time.

    Draws a subset of paths as semi-transparent lines; median path as bold line.
    Each frame reveals one more time step so the fan widens over the validation period.
    """
    n_sim, n_steps = ml_paths.shape
    n_days = n_steps - 1
    if dates is not None and len(dates) > n_days:
        dates = dates[:n_days]
    elif dates is None:
        dates = np.arange(n_days)

    # Which paths to draw (spread across sims so we see variety)
    step = max(1, n_sim // n_paths_show)
    idx = np.arange(0, n_sim, step)[:n_paths_show]
    ml_show = ml_paths[idx]   # (n_paths_show, n_steps)
    bh_show = bh_paths[idx]

    # Median across all sims (not just shown subset)
    ml_median = np.median(ml_paths, axis=0)
    bh_median = np.median(bh_paths, axis=0)

    # Frame index -> how many days to show (1 .. n_days)
    frame_indices = np.linspace(1, n_days, frames, dtype=int)
    # Separate y-limits per strategy so each plot is clean
    ml_ymin = max(0, np.percentile(ml_paths, 1))
    ml_ymax = np.percentile(ml_paths, 99)
    bh_ymin = max(0, np.percentile(bh_paths, 1))
    bh_ymax = np.percentile(bh_paths, 99)
    margin_ml = (ml_ymax - ml_ymin) * 0.08 or 10
    margin_bh = (bh_ymax - bh_ymin) * 0.08 or 10

    fig, (ax_ml, ax_bh) = plt.subplots(1, 2, figsize=(12, 5))
    ax_ml.set_xlim(dates[0], dates[-1])
    ax_ml.set_ylim(ml_ymin - margin_ml, ml_ymax + margin_ml)
    ax_ml.set_ylabel("Wealth ($)")
    ax_ml.set_xlabel("Date")
    ax_ml.grid(True, alpha=0.3)
    ax_ml.axhline(start_money, color="gray", linestyle="--", alpha=0.7)
    ax_ml.set_title("ML strategy (long/short) — Monte Carlo paths (bold = median)")
    locator = mdates.AutoDateLocator(maxticks=8)
    ax_ml.xaxis.set_major_locator(locator)
    ax_ml.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax_bh.set_xlim(dates[0], dates[-1])
    ax_bh.set_ylim(bh_ymin - margin_bh, bh_ymax + margin_bh)
    ax_bh.set_ylabel("Wealth ($)")
    ax_bh.set_xlabel("Date")
    ax_bh.grid(True, alpha=0.3)
    ax_bh.axhline(start_money, color="gray", linestyle="--", alpha=0.7)
    ax_bh.set_title("Buy & Hold — Monte Carlo paths (bold = median)")
    locator_bh = mdates.AutoDateLocator(maxticks=8)
    ax_bh.xaxis.set_major_locator(locator_bh)
    ax_bh.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator_bh))

    # Line artists: one subplot per strategy
    alpha_path = 0.15
    lines_ml = [ax_ml.plot([], [], color="purple", alpha=alpha_path)[0] for _ in range(n_paths_show)]
    line_ml_med, = ax_ml.plot([], [], color="darkviolet", linewidth=2.5, label="Median")
    ax_ml.legend(loc="upper left")

    lines_bh = [ax_bh.plot([], [], color="green", alpha=alpha_path)[0] for _ in range(n_paths_show)]
    line_bh_med, = ax_bh.plot([], [], color="darkgreen", linewidth=2.5, label="Median")
    ax_bh.legend(loc="upper left")

    def init():
        for L in lines_ml + lines_bh:
            L.set_data([], [])
        line_ml_med.set_data([], [])
        line_bh_med.set_data([], [])
        return lines_ml + lines_bh + [line_ml_med, line_bh_med]

    def update(frame_idx):
        t = frame_indices[frame_idx]
        x = dates[:t]
        for k in range(n_paths_show):
            lines_ml[k].set_data(x, ml_show[k, 1 : t + 1])
            lines_bh[k].set_data(x, bh_show[k, 1 : t + 1])
        line_ml_med.set_data(dates[:t], ml_median[1 : t + 1])
        line_bh_med.set_data(dates[:t], bh_median[1 : t + 1])
        return lines_ml + lines_bh + [line_ml_med, line_bh_med]

    anim = FuncAnimation(
        fig, update, init_func=init, frames=frames,
        interval=interval, blit=True, repeat=True
    )
    if save_path:
        try:
            anim.save(save_path, writer="pillow", fps=1000 // max(1, interval))
            print(f"  Saved Monte Carlo animation to {save_path}")
        except Exception as e:
            print(f"  Could not save (install pillow for GIF): {e}")
    plt.tight_layout()
    plt.show()
    return anim


def plot_monte_carlo(ml_final, bh_final, start_money=100):
    """Plot distribution of final wealth from Monte Carlo (e.g. histograms or box plot)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(bh_final, bins=50, alpha=0.6, label="Buy & Hold", color="green", density=True)
    ax1.hist(ml_final, bins=50, alpha=0.6, label="ML strategy", color="purple", density=True)
    ax1.axvline(start_money, color="gray", linestyle="--", label="Start")
    ax1.set_xlabel("Final wealth ($)")
    ax1.set_ylabel("Density")
    ax1.set_title("Monte Carlo: distribution of final wealth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    names = ["Buy & Hold", "ML strategy"]
    datas = [bh_final, ml_final]
    colors = ["green", "purple"]
    x = np.arange(len(names))
    for i, (name, data) in enumerate(zip(names, datas)):
        p5, p50, p95 = np.percentile(data, [5, 50, 95])
        ax2.bar(x[i], p50, color=colors[i], alpha=0.7, label=name)
        ax2.errorbar(
            x[i], p50, yerr=[[p50 - p5], [p95 - p50]],
            fmt="none", color="black", capsize=5
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.axhline(start_money, color="gray", linestyle="--")
    ax2.set_ylabel("Final wealth ($)")
    ax2.set_title("Median and 5th–95th percentile")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


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
