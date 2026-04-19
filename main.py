from pathlib import Path

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

    return {
        "records": records,
        "chart_data": chart_data,
        "signal_chart_data": signal_chart_data,
        "metrics": metrics,
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
