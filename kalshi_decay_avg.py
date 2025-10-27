#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# r-string docstring prevents backslash warnings on Windows paths.
r"""
Kalshi Event Contract Decay — Averaged Curve (Windows-friendly)

- Reads KEY ID from .\api_key.pem (first UUID found)
- Reads RSA private key from .\kalshi_private_key.pem
- Reads market titles (col A) from .\kalshi_study_candidates.xlsx
- Builds per-market daily losing-side prices, aligns by DTE (0..60),
  and then computes ONE averaged point per DTE across all markets.
- Saves:
    {output_prefix}_avg_points.csv
    {output_prefix}_avg_scatter.png
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import requests
from requests import exceptions as req_exc
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
GUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

def p(msg: str) -> None:
    print(msg, flush=True)

# -------- Key ID loader (accepts UUID literal OR file containing it) --------

def load_key_id(key_id: Optional[str], key_id_file: Optional[str]) -> str:
    if key_id and GUID_RE.fullmatch(key_id.strip()):
        return key_id.strip()
    cand = (key_id if key_id and os.path.isfile(key_id) else key_id_file) or ""
    if not cand or not os.path.isfile(cand):
        raise FileNotFoundError(
            "KEY ID not provided. Pass --key-id <uuid> OR --key-id-file <path> (e.g., .\\api_key.pem)."
        )
    with open(cand, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    m = GUID_RE.search(text)
    if not m:
        raise ValueError(
            f"No KEY ID (UUID) found in {cand}. Copy your Key ID into that file, or pass --key-id YOUR-UUID."
        )
    return m.group(0)

# ----------------------- Minimal signed GET client --------------------------

class KalshiClient:
    def __init__(self, key_id: str, private_key_path: str) -> None:
        self.key_id = key_id
        with open(private_key_path, "rb") as f:
            self.private_key: rsa.RSAPrivateKey = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )

    def _sign(self, message: str) -> str:
        sig = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        ts_ms = int(dt.datetime.utcnow().timestamp() * 1000)
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(f"{ts_ms}{method}{path}"),
            "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
        }

    def get(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> dict:
        path = endpoint if endpoint.startswith("/") else "/" + endpoint
        url = f"{API_BASE}{path}"
        r = requests.get(url, headers=self._headers("GET", path), params=params, timeout=60)
        r.raise_for_status()
        return r.json()

# ------------------------------- Core logic --------------------------------

@dataclass
class MarketMeta:
    title: str
    ticker: str
    series_ticker: str
    result: Optional[str]

def load_market_titles(path: str) -> List[str]:
    df = pd.read_excel(path, header=None)
    return [str(t).strip() for t in df.iloc[:, 0].dropna().tolist()]

def match_markets(client: KalshiClient, titles: Iterable[str]) -> List[MarketMeta]:
    title_set = {t.lower(): t for t in titles}
    matched: Dict[str, MarketMeta] = {}
    cursor: Optional[str] = None
    while True:
        params = {"status": "settled", "limit": "1000"}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/markets", params=params)
        for m in data.get("markets", []):
            tl = (m.get("title") or "").lower()
            if tl in title_set:
                matched[tl] = MarketMeta(
                    title=m["title"],
                    ticker=m["ticker"],
                    series_ticker=m.get("series_ticker", m.get("event_ticker", "")),
                    result=m.get("result"),
                )
        cursor = data.get("cursor")
        if not cursor:
            break
    missing = [t for t in titles if t.lower() not in matched]
    if missing:
        raise ValueError(f"Could not find markets ({len(missing)} missing): {missing}")
    return [matched[t.lower()] for t in titles]

def fetch_trades(client: KalshiClient, ticker: str) -> List[dict]:
    trades: List[dict] = []
    cursor: Optional[str] = None
    while True:
        params = {"ticker": ticker, "limit": "1000"}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/markets/trades", params=params)
        trades.extend(data.get("trades", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return trades

def aggregate_daily_losing_prices(trades: Iterable[dict], result: Optional[str]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["date", "losing_price"])

    df = pd.DataFrame(trades)

    # Robust timestamp parsing: handles with/without fractional seconds, Z or offsets
    # Prefer ISO8601; fall back to 'mixed' for older/newer pandas differences.
    try:
        df["created_dt"] = pd.to_datetime(
            df["created_time"].astype(str),
            utc=True,
            format="ISO8601",
            errors="coerce",
        )
    except TypeError:
        # Some pandas versions only support format="mixed"
        df["created_dt"] = pd.to_datetime(
            df["created_time"].astype(str),
            utc=True,
            format="mixed",
            errors="coerce",
        )

    # Drop any rows that failed to parse
    df = df.dropna(subset=["created_dt"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "losing_price"])

    df["date"] = df["created_dt"].dt.date

    # Determine losing side price column
    if result == "yes":
        price_col = "no_price_dollars"
    elif result == "no":
        price_col = "yes_price_dollars"
    else:
        price_col = None

    if price_col and price_col in df.columns:
        df["losing_price"] = pd.to_numeric(df[price_col], errors="coerce")
    else:
        # Fallback: take the min across sides when result missing
        df["losing_price"] = pd.concat(
            [
                pd.to_numeric(df.get("yes_price_dollars"), errors="coerce"),
                pd.to_numeric(df.get("no_price_dollars"), errors="coerce"),
            ],
            axis=1,
        ).min(axis=1)

    # Keep one price per day: last trade of that day
    df = (
        df.sort_values("created_dt")
          .dropna(subset=["losing_price"])
          .drop_duplicates(subset="date", keep="last")
          .loc[:, ["date", "losing_price"]]
          .reset_index(drop=True)
    )
    return df


def compute_dte(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        daily["days_till_event"] = []
        return daily
    idxs = daily.index[daily["losing_price"] <= 0.01].tolist()
    event_idx = idxs[0] if idxs else (len(daily) - 1)
    event_date = daily.loc[event_idx, "date"]
    daily["days_till_event"] = daily["date"].apply(lambda d: (event_date - d).days)
    return daily

def compute_average_curve(all_points: pd.DataFrame, dte_max: int = 60) -> pd.DataFrame:
    """
    From all per-market points, keep only 0..dte_max DTE and compute
    one average price per DTE. Returns columns: days_till_event, mean_price, n_markets.
    """
    df = all_points[(all_points["days_till_event"] >= 0) & (all_points["days_till_event"] <= dte_max)]
    grp = (
        df.groupby("days_till_event", as_index=False)
          .agg(mean_price=("losing_price", "mean"), n_markets=("losing_price", "size"))
          .sort_values("days_till_event")
          .reset_index(drop=True)
    )
    return grp

def plot_average_curve(avg_df: pd.DataFrame, out_png: str) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.scatter(avg_df["days_till_event"], avg_df["mean_price"], s=28, alpha=0.9)
    ax.plot(avg_df["days_till_event"], avg_df["mean_price"], linewidth=1.6, alpha=0.8)
    ax.set_xlabel("Days Till Event (DTE)")
    ax.set_ylabel("Average Losing Side Price (dollars)")
    ax.set_title("Average Time Decay of Losing Side Prices (Kalshi Event Contracts)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    # Optional subtle annotation: how many markets contribute at DTE=60, 30, 0
    for d in [60, 30, 0]:
        row = avg_df[avg_df["days_till_event"] == d]
        if not row.empty:
            n = int(row.iloc[0]["n_markets"])
            y = float(row.iloc[0]["mean_price"])
            ax.annotate(f"n={n}", (d, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)

# ---------------------------------- CLI ------------------------------------

def main() -> None:
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description="Average decay curve across markets (Windows-friendly).")
    parser.add_argument("--markets-file", default=os.path.join(cwd, "kalshi_study_candidates.xlsx"))
    parser.add_argument("--output-prefix", default=os.path.join(cwd, "analysis_results", "kalshi_avg"))
    parser.add_argument("--key-id", default=None, help="Kalshi KEY ID (UUID) or path to a file containing it.")
    parser.add_argument("--key-id-file", default=os.path.join(cwd, "api_key.pem"))
    parser.add_argument("--private-key", default=os.path.join(cwd, "kalshi_private_key.pem"))
    parser.add_argument("--dte-max", type=int, default=60, help="Maximum DTE to include (default 60).")
    args = parser.parse_args()

    p("Resolving credentials and inputs…")
    key_id = load_key_id(args.key_id, args.key_id_file)
    if not os.path.isfile(args.private_key):
        raise FileNotFoundError(f"Private key not found: {args.private_key}")
    if not os.path.isfile(args.markets_file):
        raise FileNotFoundError(f"Markets Excel not found: {args.markets_file}")
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    p(f"KEY ID: {key_id}")
    p(f"Private key: {args.private_key}")
    p(f"Markets file: {args.markets_file}")

    titles = load_market_titles(args.markets_file)
    p(f"Loaded {len(titles)} market titles")

    client = KalshiClient(key_id, args.private_key)

    p("Matching markets…")
    metas = match_markets(client, titles)
    p(f"Matched {len(metas)} markets")

    # Collect per-market points
    all_records: List[pd.DataFrame] = []
    for m in metas:
        p(f"Fetching trades: {m.title} (ticker={m.ticker})")
        try:
            trades = fetch_trades(client, m.ticker)
        except req_exc.RequestException as e:
            p(f"  ! API error on {m.ticker}: {e}")
            continue
        daily = aggregate_daily_losing_prices(trades, m.result)
        daily = compute_dte(daily)
        # keep only 0..dte_max (exclude negatives/post-event)
        daily = daily[(daily["days_till_event"] >= 0) & (daily["days_till_event"] <= args.dte_max)]
        if daily.empty:
            p("  ! Skipping: no data in requested DTE window")
            continue
        daily["market"] = m.title
        p(f"  + {len(daily)} daily points")
        all_records.append(daily)

    if not all_records:
        raise SystemExit("No data points collected. Exiting.")

    all_points = pd.concat(all_records, ignore_index=True)

    # Build averaged curve
    avg_df = compute_average_curve(all_points, dte_max=args.dte_max)

    # Outputs
    csv_path = f"{args.output_prefix}_avg_points.csv"
    png_path = f"{args.output_prefix}_avg_scatter.png"
    avg_df.to_csv(csv_path, index=False)
    plot_average_curve(avg_df, png_path)

    p(f"Saved CSV: {csv_path}")
    p(f"Saved PNG: {png_path}")

if __name__ == "__main__":
    main()
