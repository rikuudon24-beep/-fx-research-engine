#!/usr/bin/env python3
"""Extract forward 100+ pip movement events from H4/D1 OHLC data.

The event timestamp is the CLOSE of the source candle. All features used later
must be computed from candles at or before this timestamp; this script only
labels future movement.

Outputs:
  reports/100pip_events.csv
  reports/100pip_event_summary.csv

For each source candle, both bullish and bearish future excursions are measured.
Overlapping events are clustered per pair/timeframe/direction so one sustained
move
does not become hundreds of duplicate labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

PAIRS = [
    "usdjpy", "eurjpy", "gbpjpy", "audjpy", "eurusd", "gbpusd",
    "audusd", "nzdusd", "usdcad", "usdchf", "audnzd", "eurgbp",
]
TIMEFRAMES = ["h4", "d1"]
PIP = {p: 0.01 if "jpy" in p else 0.0001 for p in PAIRS}
HORIZONS = {"h4": [1, 3, 6, 12, 24], "d1": [1, 3, 5, 10, 20]}
THRESHOLDS = [100, 150, 200, 300]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def extract_events(df: pd.DataFrame, pair: str, tf: str) -> list[dict]:
    pip = PIP[pair]
    horizons = HORIZONS[tf]
    max_h = max(horizons)
    rows = []

    for i in range(len(df) - max_h):
        entry = float(df.at[i, "close"])
        future = df.iloc[i + 1 : i + max_h + 1]
        up = (future["high"].cummax() - entry) / pip
        down = (entry - future["low"].cummin()) / pip

        for direction, excursion in (("bull", up), ("bear", down)):
            for threshold in THRESHOLDS:
                hit = excursion[excursion >= threshold]
                if hit.empty:
                    continue
                first_pos = int(hit.index[0] - i)
                first_ts = df.at[hit.index[0], "timestamp"]
                row = {
                    "pair": pair,
                    "timeframe": tf,
                    "direction": direction,
                    "threshold_pips": threshold,
                    "event_timestamp": df.at[i, "timestamp"].isoformat(),
                    "hit_timestamp": first_ts.isoformat(),
                    "bars_to_hit": first_pos,
                    "entry_close": entry,
                    "max_excursion_pips": round(float(excursion.max()), 2),
                }
                rows.append(row)

    return rows


def cluster_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    events = events.sort_values(
        ["pair", "timeframe", "direction", "threshold_pips", "event_timestamp"]
    ).copy()
    keep = []
    last_hit = {}
    for row in events.itertuples(index=False):
        key = (row.pair, row.timeframe, row.direction, row.threshold_pips)
        ts = pd.Timestamp(row.event_timestamp)
        if key not in last_hit or ts > last_hit[key]:
            keep.append(row._asdict())
            last_hit[key] = pd.Timestamp(row.hit_timestamp)
    return pd.DataFrame(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/market")
    ap.add_argument("--out", default="reports/100pip_events.csv")
    ap.add_argument("--summary", default="reports/100pip_event_summary.csv")
    args = ap.parse_args()

    all_events = []
    quality = []
    for tf in TIMEFRAMES:
        for pair in PAIRS:
            path = Path(args.data_root) / tf / f"{pair}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            df = load_csv(path)
            gaps = df["timestamp"].diff().dropna().dt.total_seconds() / 3600
            expected = 4 if tf == "h4" else 24
            bad_gaps = int((gaps > expected * 2.5).sum())
            quality.append({
                "pair": pair, "timeframe": tf, "rows": len(df),
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat(),
                "bad_gaps": bad_gaps,
            })
            all_events.extend(extract_events(df, pair, tf))

    raw = pd.DataFrame(all_events)
    events = cluster_events(raw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out, index=False)

    if events.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            events.groupby(["pair", "timeframe", "direction", "threshold_pips"])
            .agg(events=("event_timestamp", "count"),
                 median_bars_to_hit=("bars_to_hit", "median"),
                 median_max_excursion=("max_excursion_pips", "median"))
            .reset_index()
        )
    summary.to_csv(args.summary, index=False)
    pd.DataFrame(quality).to_csv("reports/100pip_data_quality.csv", index=False)

    print(f"raw_labels={len(raw)} clustered_events={len(events)}")
    print(summary.to_string(index=False) if not summary.empty else "no events")


if __name__ == "__main__":
    main()
