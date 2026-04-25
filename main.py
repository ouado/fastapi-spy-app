from pathlib import Path

import yfinance as yf
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from spy_backtest import (
    compute_return_zscores,
    download_spy_data,
    get_investment_weight,
    run_weekly_backtest,
)

app = FastAPI(title="SPY Historical Data API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/backtest")
def get_backtest():
    """Run the SPY backtest strategy and return results."""
    df = download_spy_data()
    df = compute_return_zscores(df, window=100)

    backtest_results = run_weekly_backtest(df, weekly_amount=100.0 / 52)
    backtest_results_reset = backtest_results.reset_index()
    backtest_results_reset["date"] = backtest_results_reset["date"].dt.strftime("%Y-%m-%d")

    records = backtest_results_reset.to_dict(orient="records")

    chart_data = {
        "labels": backtest_results_reset["date"].tolist(),
        "benchmark_mtm": backtest_results_reset["benchmark_mtm"].tolist(),
        "test_mtm": backtest_results_reset["test_mtm"].tolist(),
    }

    price_chart = df.reset_index()
    price_chart["date"] = price_chart["Date"].dt.strftime("%Y-%m-%d")
    below_minus_3 = price_chart[price_chart["composite_zscore_weighted"] < -3]
    between_minus_3_minus_2 = price_chart[
        (price_chart["composite_zscore_weighted"] >= -3)
        & (price_chart["composite_zscore_weighted"] < -2)
    ]
    above_2 = price_chart[price_chart["composite_zscore_weighted"] > 2]

    signal_chart_data = {
        "labels": price_chart["date"].tolist(),
        "close": price_chart["Close"].tolist(),
        "below_minus_3": [
            {"x": row.date, "y": float(row.Close)}
            for row in below_minus_3.itertuples(index=False)
        ],
        "between_minus_3_minus_2": [
            {"x": row.date, "y": float(row.Close)}
            for row in between_minus_3_minus_2.itertuples(index=False)
        ],
        "above_2": [
            {"x": row.date, "y": float(row.Close)}
            for row in above_2.itertuples(index=False)
        ],
    }

    final_benchmark = backtest_results_reset["benchmark_mtm"].iloc[-1]
    final_test = backtest_results_reset["test_mtm"].iloc[-1]
    total_benchmark_invested = backtest_results_reset["benchmark_invested"].iloc[-1]
    total_test_invested = backtest_results_reset["test_invested"].iloc[-1]
    latest_composite_zscore = float(df["composite_zscore_weighted"].dropna().iloc[-1])
    latest_weight = float(get_investment_weight(latest_composite_zscore))

    metrics = {
        "final_benchmark_value": float(final_benchmark),
        "final_test_value": float(final_test),
        "total_benchmark_invested": float(total_benchmark_invested),
        "total_test_invested": float(total_test_invested),
        "latest_composite_zscore": latest_composite_zscore,
        "latest_recommended_weight": latest_weight,
        "benchmark_return_pct": float(
            (final_benchmark - total_benchmark_invested) / total_benchmark_invested * 100
        )
        if total_benchmark_invested > 0
        else 0,
        "test_return_pct": float(
            (final_test - total_test_invested) / total_test_invested * 100
        )
        if total_test_invested > 0
        else 0,
    }

    recent_zscores_df = df[["composite_zscore_weighted"]].dropna().tail(10).reset_index()
    recent_zscores_df["Date"] = recent_zscores_df["Date"].dt.strftime("%Y-%m-%d")
    recent_zscores_df["weight"] = recent_zscores_df["composite_zscore_weighted"].apply(get_investment_weight)
    recent_zscores = [
        {"date": row["Date"], "zscore": float(row["composite_zscore_weighted"]), "weight": float(row["weight"])}
        for _, row in recent_zscores_df.iloc[::-1].iterrows()
    ]

    return {
        "records": records,
        "chart_data": chart_data,
        "signal_chart_data": signal_chart_data,
        "metrics": metrics,
        "recent_zscores": recent_zscores,
    }


@app.get("/live")
def get_live():
    """Fetch live SPY price and compute intraday-adjusted investment weight."""
    # Need enough history for a full 100-day rolling window across all periods (max period=60)
    df = download_spy_data(start_date=(pd.Timestamp.today() - pd.DateOffset(days=300)).strftime("%Y-%m-%d"))
    df = compute_return_zscores(df, window=100)

    ticker = yf.Ticker("SPY")
    try:
        live_price = float(ticker.fast_info["last_price"])
    except Exception:
        live_price = float(df["Close"].iloc[-1])

    # For each period p, the p-day return at the last bar = close[-1] / close[-p-1] - 1
    # The live version replaces close[-1] with live_price, but the reference close is close[-p] back from today
    # Since live_price represents *today* (one day after close[-1]), the reference close is close[-p] from today's perspective
    periods = [1, 5, 10, 20, 60]
    weights = {1: 1 / 96, 5: 5 / 96, 10: 10 / 96, 20: 20 / 96, 60: 60 / 96}

    close = df["Close"]
    live_composite = 0.0
    for p in periods:
        # Match pct_change(p) at the latest bar: close[-1] / close[-(p+1)] - 1.
        # Replacing close[-1] with live_price keeps the live and historical definitions aligned.
        ref_close = float(close.iloc[-(p + 1)])
        live_p_return = (live_price / ref_close - 1) * 100

        mean_p = float(df[f"{p}d_return"].rolling(100).mean().iloc[-1])
        std_p = float(df[f"{p}d_return"].rolling(100).std().iloc[-1])
        live_p_zscore = (live_p_return - mean_p) / std_p if std_p != 0 else 0.0

        live_composite += weights[p] * live_p_zscore

    live_weight = float(get_investment_weight(live_composite))

    return {
        "live_price": live_price,
        "live_composite_zscore": live_composite,
        "live_weight": live_weight,
    }


@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = Path(__file__).parent / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/manifest.json")
def get_manifest():
    manifest_path = Path(__file__).parent / "manifest.json"
    return FileResponse(manifest_path, media_type="application/manifest+json")


@app.get("/service-worker.js")
def get_service_worker():
    sw_path = Path(__file__).parent / "service-worker.js"
    return FileResponse(sw_path, media_type="application/javascript")
