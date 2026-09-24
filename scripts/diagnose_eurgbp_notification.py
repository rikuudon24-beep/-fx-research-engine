#!/usr/bin/env python3
"""Diagnostic replay of the frozen EURGBP H4 notification state machine.

Uses the same data/features/rules as monitor_notification_state.py, but records
every state transition around a requested historical window. Completed candles
only; no lookahead.
"""
from pathlib import Path
import importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location("d", "scripts/research_50pip_direct_entry.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

PAIR = "eurgbp"
START = pd.Timestamp("2026-09-17T00:00:00Z")
END = pd.Timestamp("2026-09-24T08:00:00Z")

g = d.load_market("h4", PAIR).sort_values("timestamp").reset_index(drop=True)
f = d.build_features(g)
e20 = g.close.ewm(span=20, adjust=False).mean()
e200 = g.close.ewm(span=200, adjust=False).mean()
gc = (e20.shift(1) <= e200.shift(1)) & (e20 > e200)
dc = (e20.shift(1) >= e200.shift(1)) & (e20 < e200)

state = "WAIT"
direction = None
touch = None
ref = None
events = []

for i in range(1, len(g)):
    ts = pd.Timestamp(g.timestamp.iloc[i])
    if ts < START or ts > END:
        # We still need the historical state to be correct, so process all bars.
        pass

    old = state
    action = ""
    filt = False
    trig = False
    invalid = False

    if state == "WAIT":
        if bool(gc.iloc[i]):
            direction = "long"; state = "SEARCH_TOUCH"; action = "GC -> SEARCH_TOUCH"
        elif bool(dc.iloc[i]):
            direction = "short"; state = "SEARCH_TOUCH"; action = "DC -> SEARCH_TOUCH"
    elif state == "SEARCH_TOUCH":
        if (direction == "long" and bool(dc.iloc[i])) or (direction == "short" and bool(gc.iloc[i])):
            state = "WAIT"; direction = None; touch = None; ref = None
            action = "reverse cross -> WAIT"
        elif float(g.low.iloc[i]) <= float(e20.iloc[i]) <= float(g.high.iloc[i]):
            touch = i
            ref = float(g.high.iloc[i] if direction == "long" else g.low.iloc[i])
            state = "ARMED"
            action = f"FIRST TOUCH -> ARMED ref={ref:.5f}"
    elif state == "ARMED":
        if (direction == "long" and bool(dc.iloc[i])) or (direction == "short" and bool(gc.iloc[i])):
            state = "WAIT"; direction = None; touch = None; ref = None
            action = "reverse cross -> WAIT"
        else:
            filt = (
                bool(f.sma_stack_bull.iloc[i]) and bool(f.di_strong_bull.iloc[i])
                if direction == "long" else bool(f.strong_close_bear.iloc[i])
            )
            trig = float(g.close.iloc[i]) > ref if direction == "long" else float(g.close.iloc[i]) < ref
            if filt and trig:
                state = "TRIGGERED"
                action = "FILTER+BREAKOUT -> TRIGGERED"
            elif (
                (direction == "long" and float(g.close.iloc[i]) <= float(g.low.iloc[touch]))
                or (direction == "short" and float(g.close.iloc[i]) >= float(g.high.iloc[touch]))
            ):
                state = "WAIT"; direction = None; touch = None; ref = None
                action = "touch-candle invalidation -> WAIT"

    if START <= ts <= END:
        events.append({
            "timestamp": ts.isoformat(),
            "state_before": old,
            "state_after": state,
            "direction": direction or "",
            "open": float(g.open.iloc[i]),
            "high": float(g.high.iloc[i]),
            "low": float(g.low.iloc[i]),
            "close": float(g.close.iloc[i]),
            "ema20": float(e20.iloc[i]),
            "ema200": float(e200.iloc[i]),
            "gc": bool(gc.iloc[i]),
            "dc": bool(dc.iloc[i]),
            "sma_stack_bull": bool(f.sma_stack_bull.iloc[i]),
            "di_strong_bull": bool(f.di_strong_bull.iloc[i]),
            "filter": filt,
            "breakout_vs_ref": trig,
            "ref": ref if ref is not None else "",
            "action": action,
        })

    if state == "TRIGGERED":
        if i == len(g) - 1:
            break
        # same rule as monitor: signal window is next H4 open only
        if not action:
            action = "TRIGGERED -> WAIT after entry window"
        if START <= ts <= END:
            events[-1]["state_after"] = "TRIGGERED"
            events[-1]["action"] = action or "TRIGGERED"
        state = "WAIT"; direction = None; touch = None; ref = None

out = pd.DataFrame(events)
Path("reports").mkdir(exist_ok=True)
out.to_csv("reports/eurgbp_notification_diagnostic.csv", index=False)
print(out.to_string(index=False))
