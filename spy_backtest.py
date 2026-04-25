import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def download_spy_data(start_date: str = "2000-01-01", end_date: str | None = None) -> pd.DataFrame:
    df = yf.download("SPY", start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})
    return df


def compute_return_zscores(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    close = df["Close"]
    periods = [1, 5, 10, 20, 60]

    for period in periods:
        df[f"{period}d_return"] = close.pct_change(period) * 100
        df[f"{period}d_return_zscore"] = (
            df[f"{period}d_return"] - df[f"{period}d_return"].rolling(window=window).mean()
        ) / df[f"{period}d_return"].rolling(window=window).std()

    weights = {
        1: 1 / 96,
        5: 5 / 96,
        10: 10 / 96,
        20: 20 / 96,
        60: 60 / 96,
    }

    df["composite_zscore_weighted"] = sum(
        weights[p] * df[f"{p}d_return_zscore"] for p in periods
    )

    return df


def get_investment_weight(zscore: float) -> float:
    w1 = 3.0
    w2 = 10.0
    w3 = 50.0

    if zscore < -3:
        return w3
    if -3 <= zscore < -2:
        return np.interp(zscore, [-3, -2], [w3, w2])
    if -2 <= zscore < -1:
        return np.interp(zscore, [-2, -1], [w2, w1])
    if -1 <= zscore <= 1:
        return 1.0
    if 1 < zscore <= 2:
        return np.interp(zscore, [1, 2], [1 / w1, 1 / w2])
    if 2 < zscore <= 3:
        return np.interp(zscore, [2, 3], [1 / w2, 1 / w3])
    return 1.0 / w1


def run_weekly_backtest(df: pd.DataFrame, weekly_amount: float = 1.0 / 52) -> pd.DataFrame:
    portfolio = {
        "date": [],
        "benchmark_invested": [],
        "benchmark_shares": [],
        "test_invested": [],
        "test_shares": [],
        "test_weight": [],
        "benchmark_mtm": [],
        "test_mtm": [],
    }

    benchmark_shares = 0.0
    test_shares = 0.0
    benchmark_invested = 0.0
    test_invested = 0.0

    for i in range(100, len(df) - 1):
        if df.index[i].weekday() == 4:  # Only invest on Fridays
            next_close = df["Close"].iloc[i]
            weight = get_investment_weight(df["composite_zscore_weighted"].iloc[i])

            benchmark_invested += weekly_amount
            benchmark_shares += weekly_amount / next_close

            test_invested += weight * weekly_amount
            test_shares += weight * weekly_amount / next_close

            portfolio["date"].append(df.index[i + 1])
            portfolio["benchmark_invested"].append(benchmark_invested)
            portfolio["benchmark_shares"].append(benchmark_shares)
            portfolio["test_invested"].append(test_invested)
            portfolio["test_shares"].append(test_shares)
            portfolio["test_weight"].append(weight)
            portfolio["benchmark_mtm"].append(benchmark_shares * next_close)
            portfolio["test_mtm"].append(test_shares * next_close)

    result = pd.DataFrame(portfolio)
    result.set_index("date", inplace=True)
    return result


def calculate_metrics(df: pd.DataFrame) -> dict:
    returns = df["test_mtm"].pct_change().dropna()
    annualized_return = (df["test_mtm"].iloc[-1] / df["test_mtm"].iloc[0]) ** (252 / len(df)) - 1
    sharpe = np.nan
    if returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    return {
        "final_test_value": df["test_mtm"].iloc[-1],
        "final_benchmark_value": df["benchmark_mtm"].iloc[-1],
        "test_invested": df["test_invested"].iloc[-1],
        "benchmark_invested": df["benchmark_invested"].iloc[-1],
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
    }


def plot_results(df: pd.DataFrame, spy_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["benchmark_mtm"], label="Benchmark Strategy", linewidth=1.5)
    plt.plot(df.index, df["test_mtm"], label="Test Strategy", linewidth=1.5)
    plt.title("Benchmark vs Test Strategy Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(spy_df.index, spy_df["Close"], color="black", label="SPY Close")
    signals = spy_df[spy_df["composite_zscore_weighted"] < -3]
    if not signals.empty:
        plt.scatter(signals.index, signals["Close"], color="red", label="Z-score < -3", s=30)
    signals = spy_df[(spy_df["composite_zscore_weighted"] > 2) & (spy_df["composite_zscore_weighted"] <= 3)]
    if not signals.empty:
        plt.scatter(signals.index, signals["Close"], color="green", label="Z-score > 2", s=30)
    plt.title("SPY Close Price with Signal Highlights")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_weight_distribution(df: pd.DataFrame) -> None:
    weights = pd.Series(df["test_weight"]).dropna()
    bins = [0, 0.1, 0.3, 1, 3, 10, 20, weights.max() if weights.max() > 20 else 20]
    labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]
    weight_bins = pd.cut(weights, bins=bins, labels=labels, include_lowest=True)
    frequency = weight_bins.value_counts().sort_index()
    frequency_pct = (frequency / frequency.sum() * 100).round(2)

    distribution = pd.DataFrame({"Count": frequency, "Percent": frequency_pct})
    print("Weight distribution:")
    print(distribution)


def main() -> None:
    spy_df = download_spy_data(start_date="2000-01-01")
    spy_df = compute_return_zscores(spy_df, window=100)

    backtest_df = run_weekly_backtest(spy_df)
    metrics = calculate_metrics(backtest_df)

    print("=== Backtest Summary ===")
    print(f"Final benchmark MTM: ${metrics['final_benchmark_value']:.2f}")
    print(f"Final test MTM:      ${metrics['final_test_value']:.2f}")
    print(f"Benchmark invested:  ${metrics['benchmark_invested']:.2f}")
    print(f"Test strategy invested: ${metrics['test_invested']:.2f}")
    print(f"Annualized return (test): {metrics['annualized_return']:.2%}")
    print(f"Sharpe ratio (test): {metrics['sharpe_ratio']:.2f}")

    print_weight_distribution(backtest_df)
    plot_results(backtest_df, spy_df)


if __name__ == "__main__":
    main()
