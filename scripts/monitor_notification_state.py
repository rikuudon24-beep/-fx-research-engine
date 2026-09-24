#!/usr/bin/env python3
"""Evaluate frozen H4 notification state using completed candles only.

The monitor deliberately excludes the currently forming H4 candle based on
UTC wall-clock time. A TRIGGERED state means the just-completed candle satisfied
the frozen entry rule; the next H4 open is the entry candidate.
"""
from pathlib import Path
import importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location("d", "scripts/research_50pip_direct_entry.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

PAIRS = d.PAIRS
H4 = pd.Timedelta(hours=4)

def completed_market(pair):
    g = d.load_market("h4", pair).sort_values("timestamp").reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now.floor("4h") - H4
    g = g[pd.to_datetime(g.timestamp, utc=True) <= cutoff].copy().reset_index(drop=True)
    if g.empty:
        raise RuntimeError(f"no completed H4 candles for {pair}")
    return g

def evaluate(pair):
    g = completed_market(pair)
    f = d.build_features(g)
    e20 = g.close.ewm(span=20, adjust=False).mean()
    e200 = g.close.ewm(span=200, adjust=False).mean()
    gc = (e20.shift(1) <= e200.shift(1)) & (e20 > e200)
    dc = (e20.shift(1) >= e200.shift(1)) & (e20 < e200)

    state = "WAIT"
    direction = None
    touch = None
    ref = None
    signal = None
    breach_streak = 0

    for i in range(1, len(g)):
        if state == "WAIT":
            if bool(gc.iloc[i]):
                direction = "long"; state = "SEARCH_TOUCH"
            elif bool(dc.iloc[i]):
                direction = "short"; state = "SEARCH_TOUCH"
        elif state == "SEARCH_TOUCH":
            if (direction == "long" and bool(dc.iloc[i])) or (direction == "short" and bool(gc.iloc[i])):
                state = "WAIT"; direction = None; touch = None; ref = None; breach_streak = 0
                continue
            if float(g.low.iloc[i]) <= float(e20.iloc[i]) <= float(g.high.iloc[i]):
                touch = i
                ref = float(g.high.iloc[i] if direction == "long" else g.low.iloc[i])
                breach_streak = 0
                state = "ARMED"
        elif state == "ARMED":
            if (direction == "long" and bool(dc.iloc[i])) or (direction == "short" and bool(gc.iloc[i])):
                state = "WAIT"; direction = None; touch = None; ref = None; breach_streak = 0
                continue
            filt = (
                bool(f.sma_stack_bull.iloc[i]) and bool(f.di_strong_bull.iloc[i])
                if direction == "long" else bool(f.strong_close_bear.iloc[i])
            )
            trig = float(g.close.iloc[i]) > ref if direction == "long" else float(g.close.iloc[i]) < ref
            if filt and trig:
                signal = i
                state = "TRIGGERED"
            else:
                breached = (
                    (direction == "long" and float(g.close.iloc[i]) <= float(g.low.iloc[touch]))
                    or (direction == "short" and float(g.close.iloc[i]) >= float(g.high.iloc[touch]))
                )
                breach_streak = breach_streak + 1 if breached else 0
                if breach_streak >= 2:
                    state = "WAIT"; direction = None; touch = None; ref = None; breach_streak = 0
        elif state == "TRIGGERED":
            # A completed candle after the signal means the entry window has
            # already passed; replaying further history starts a new search.
            if i == len(g) - 1:
                break
            state = "WAIT"; direction = None; touch = None; ref = None; breach_streak = 0

    i = len(g) - 1
    ts = pd.Timestamp(g.timestamp.iloc[i])
    signal_ts = pd.Timestamp(g.timestamp.iloc[signal]) if signal is not None else None
    entry_ts = signal_ts + H4 if signal_ts is not None else None
    row = {
        "pair": pair,
        "latest_completed_h4": ts.isoformat(),
        "state": state,
        "direction": direction or "",
        "signal_candle": signal_ts.isoformat() if signal_ts is not None else "",
        "entry_candidate_h4": entry_ts.isoformat() if entry_ts is not None else "",
        "close": float(g.close.iloc[i]),
        "ema20": float(e20.iloc[i]),
        "ema200": float(e200.iloc[i]),
        "gc": bool(gc.iloc[i]) if pd.notna(gc.iloc[i]) else False,
        "dc": bool(dc.iloc[i]) if pd.notna(dc.iloc[i]) else False,
        "sma_stack_bull": bool(f.sma_stack_bull.iloc[i]),
        "di_strong_bull": bool(f.di_strong_bull.iloc[i]),
        "strong_close_bear": bool(f.strong_close_bear.iloc[i]),
        "source": "dukascopy_h4_csv",
    }
    return row

def main():
    rows = [evaluate(p) for p in PAIRS]
    out = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/current_notification_state.csv", index=False)

    active = out[out.state.isin(["ARMED", "TRIGGERED"])].copy()
    active.to_csv("reports/current_notification_candidates.csv", index=False)

    alerts = out[out.state == "TRIGGERED"].copy()
    alerts["notification_id"] = alerts.apply(
        lambda r: f"{r['pair']}|{r['direction']}|{r['signal_candle']}", axis=1
    )
    alerts.to_csv("reports/current_notification_alerts.csv", index=False)

    print("=== CURRENT NOTIFICATION STATE ===")
    print(out.to_string(index=False))
    print("=== ACTIVE CANDIDATES ===")
    print(active.to_string(index=False) if len(active) else "none")
    print("=== NEW ALERT CANDIDATES ===")
    print(alerts[["notification_id","pair","direction","signal_candle","entry_candidate_h4"]].to_string(index=False)
          if len(alerts) else "none")

if __name__ == "__main__":
    main()
