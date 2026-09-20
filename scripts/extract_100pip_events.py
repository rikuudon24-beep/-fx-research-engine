#!/usr/bin/env python3
"""Extract forward 50+ pip movement events from H4/D1 OHLC data.

The event timestamp is the CLOSE of the source candle. Labels use only future
candles, preventing look-ahead leakage in later feature research.

Thresholds: 50/100/150/200/300 pips.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

PAIRS = [
    "usdjpy","eurjpy","gbpjpy","audjpy","eurusd","gbpusd",
    "audusd","nzdusd","usdcad","usdchf","audnzd","eurgbp",
]
TIMEFRAMES = ["h4","d1"]
PIP = {p: 0.01 if "jpy" in p else 0.0001 for p in PAIRS}
HORIZONS = {"h4":[1,3,6,12,24], "d1":[1,3,5,10,20]}
THRESHOLDS = [50,100,150,200,300]

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp","open","high","low","close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def extract_events(df, pair, tf):
    pip = PIP[pair]
    max_h = max(HORIZONS[tf])
    rows = []
    for i in range(len(df) - max_h):
        entry = float(df.at[i,"close"])
        future = df.iloc[i+1:i+max_h+1]
        up = (future["high"].cummax()-entry)/pip
        down = (entry-future["low"].cummin())/pip
        for direction, excursion in (("bull",up),("bear",down)):
            for threshold in THRESHOLDS:
                hit = excursion[excursion >= threshold]
                if hit.empty:
                    continue
                idx = hit.index[0]
                rows.append({
                    "pair":pair, "timeframe":tf, "direction":direction,
                    "threshold_pips":threshold,
                    "event_timestamp":df.at[i,"timestamp"].isoformat(),
                    "hit_timestamp":df.at[idx,"timestamp"].isoformat(),
                    "bars_to_hit":int(idx-i),
                    "entry_close":entry,
                    "max_excursion_pips":round(float(excursion.max()),2),
                })
    return rows

def cluster_events(events):
    if events.empty:
        return events
    events = events.sort_values(
        ["pair","timeframe","direction","threshold_pips","event_timestamp"]
    )
    keep, last_hit = [], {}
    for row in events.itertuples(index=False):
        key=(row.pair,row.timeframe,row.direction,row.threshold_pips)
        ts=pd.Timestamp(row.event_timestamp)
        if key not in last_hit or ts > last_hit[key]:
            keep.append(row._asdict())
            last_hit[key]=pd.Timestamp(row.hit_timestamp)
    return pd.DataFrame(keep)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",default="data/market")
    ap.add_argument("--out",default="reports/50pip_events.csv")
    ap.add_argument("--summary",default="reports/50pip_event_summary.csv")
    args=ap.parse_args()
    all_events=[]; quality=[]
    for tf in TIMEFRAMES:
        for pair in PAIRS:
            path=Path(args.data_root)/tf/f"{pair}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            df=load_csv(path)
            gaps=df["timestamp"].diff().dropna().dt.total_seconds()/3600
            expected=4 if tf=="h4" else 24
            quality.append({
                "pair":pair,"timeframe":tf,"rows":len(df),
                "start":df["timestamp"].min().isoformat(),
                "end":df["timestamp"].max().isoformat(),
                "bad_gaps":int((gaps>expected*2.5).sum()),
            })
            all_events.extend(extract_events(df,pair,tf))
    raw=pd.DataFrame(all_events)
    events=cluster_events(raw)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    events.to_csv(args.out,index=False)
    summary=(events.groupby(["pair","timeframe","direction","threshold_pips"])
             .agg(events=("event_timestamp","count"),
                  median_bars_to_hit=("bars_to_hit","median"),
                  median_max_excursion=("max_excursion_pips","median"))
             .reset_index()) if not events.empty else pd.DataFrame()
    summary.to_csv(args.summary,index=False)
    pd.DataFrame(quality).to_csv("reports/50pip_data_quality.csv",index=False)
    print(f"raw_labels={len(raw)} clustered_events={len(events)}")
    print(summary.to_string(index=False) if not summary.empty else "no events")

if __name__=="__main__":
    main()
