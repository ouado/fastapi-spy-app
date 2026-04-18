from pathlib import Path

import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

app = FastAPI(title="SPY Historical Data API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_spy_history() -> pd.DataFrame:
    df = yf.download("SPY", start="2000-01-01", progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()
    df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df[["date", "Open", "High", "Low", "Close", "Volume"]]


@app.get("/data")
def get_data():
    df = fetch_spy_history()
    records = df.to_dict(orient="records")
    chart_data = {
        "labels": df["date"].tolist(),
        "close": df["Close"].tolist(),
        "volume": df["Volume"].tolist(),
    }
    return {
        "columns": df.columns.tolist(),
        "records": records,
        "chart_data": chart_data,
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
