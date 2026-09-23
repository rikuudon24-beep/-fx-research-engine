#!/usr/bin/env python3
"""Evaluate the frozen H4 notification state on the latest completed candle.

This is a monitoring layer, not a trade executor. It never uses the current
incomplete candle: the latest row in each repository CSV is treated as the
latest completed H4 candle available to the workflow.
"""
from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np

spec = importlib.util.spec_from_file_location("d", "scripts/research_50pip_direct_entry.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

PAIRS = d.PAIRS

def evaluate(pair):
    g = d.load_market("h4", pair).reset_index(drop=True)
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

    # Replay the complete available history, then inspect the final state.
    for i in range(1, len(g)):
        if state == "WAIT":
            if bool(gc.iloc[i]):
                direction = "long"; state = "SEARCH_TOUCH"
            elif bool(dc.iloc[i]):
                direction = "short"; state = "SEARCH_TOUCH"
        elif state == "SEARCH_TOUCH":
            if (direction == "long" and bool(dc.iloc[i])) or (direction == "short" and bool(gc.iloc[i])):
                state = "WAIT"; direction = None; touch = None; ref = None
                continue
            if float(g.low.iloc[i]) <= float(e20.iloc[i]) <= float(g.high.iloc[i]):
                touch = i
                ref = float(g.high.iloc[i] if direction == "long" else g.low.iloc[i])
                state = "ARMED"
        elif state == "ARMED":
            if (direction == "long" and bool(dc.iloc[i])) or (direction == "short" and bool(gc.iloc[i])):
                state = "WAIT"; direction = None; touch = None; ref = None
                continue
            filt = (
                bool(f.sma_stack_bull.iloc[i]) and bool(f.di_strong_bull.iloc[i])
                if direction == "long" else bool(f.strong_close_bear.iloc[i])
            )
            trig = float(g.close.iloc[i]) > ref if direction == "long" else float(g.close.iloc[i]) < ref
            if filt and trig:
                signal = i
                state = "TRIGGERED"
            elif (
                (direction == "long" and float(g.close.iloc[i]) <= float(g.low.iloc[touch]))
                or (direction == "short" and float(g.close.iloc[i]) >= float(g.high.iloc[touch]))
            ):
                state = "WAIT"; direction = None; touch = None; ref = None
        elif state == "TRIGGERED":
            # The next H4 open would be the entry candidate. Keep the state
            # visible until the next completed candle is observed.
            if i == len(g) - 1:
                break
            state = "WAIT"; direction = None; touch = None; ref = None

    i = len(g) - 1
    ts = pd.Timestamp(g.timestamp.iloc[i])
    row = {
        "pair": pair,
        "latest_completed_h4": ts.isoformat(),
        "state": state,
        "direction": direction or "",
        "signal_candle": pd.Timestamp(g.timestamp.iloc[signal]).isoformat() if signal is not None else "",
        "close": float(g.close.iloc[i]),
        "ema20": float(e20.iloc[i]),
        "ema200": float(e200.iloc[i]),
        "gc": bool(gc.iloc[i]) if pd.notna(gc.iloc[i]) else False,
        "dc": bool(dc.iloc[i]) if pd.notna(dc.iloc[i]) else False,
        "sma_stack_bull": bool(f.sma_stack_bull.iloc[i]),
        "di_strong_bull": bool(f.di_strong_bull.iloc[i]),
        "strong_close_bear": bool(f.strong_close_bear.iloc[i]),
        "source": "repository_h4_csv",
    }
    return row

def main():
    rows = [evaluate(p) for p in PAIRS]
    out = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/current_notification_state.csv", index=False)
    active = out[out.state.isin(["ARMED", "TRIGGERED"])]
    active.to_csv("reports/current_notification_candidates.csv", index=False)
    print("=== CURRENT NOTIFICATION STATE ===")
    print(out.to_string(index=False))
    print("=== ACTIVE CANDIDATES ===")
    print(active.to_string(index=False) if len(active) else "none")

if __name__ == "__main__":
    main()
