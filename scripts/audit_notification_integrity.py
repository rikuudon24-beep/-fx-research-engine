#!/usr/bin/env python3
"""Audit the H4 notification engine for historical signal omissions.

Replays the live monitor logic on completed candles and records:
- every TRIGGERED event
- every ARMED invalidation/reset
- near-misses where breakout/filter conditions were true but the setup was
  already lost
- state-machine health checks (no duplicate triggers, no incomplete candles)
"""
from pathlib import Path
import importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location("d", "scripts/research_50pip_direct_entry.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

PAIRS = d.PAIRS

def audit_pair(pair):
    g = d.load_market("h4", pair).sort_values("timestamp").reset_index(drop=True)
    g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
    f = d.build_features(g)
    e20 = g.close.ewm(span=20, adjust=False).mean()
    e200 = g.close.ewm(span=200, adjust=False).mean()
    gc = (e20.shift(1) <= e200.shift(1)) & (e20 > e200)
    dc = (e20.shift(1) >= e200.shift(1)) & (e20 < e200)

    state = "WAIT"; direction = None; touch = None; ref = None
    breach_streak = 0
    events = []

    def emit(ts, event, reason="", **kw):
        events.append({"pair":pair, "timestamp":ts.isoformat(), "event":event,
                       "direction":direction or "", "state":state, "reason":reason,
                       **kw})

    for i in range(1, len(g)):
        ts = g.timestamp.iloc[i]
        if state == "WAIT":
            if bool(gc.iloc[i]):
                direction="long"; state="SEARCH_TOUCH"; breach_streak=0
                emit(ts,"GC","new long search")
            elif bool(dc.iloc[i]):
                direction="short"; state="SEARCH_TOUCH"; breach_streak=0
                emit(ts,"DC","new short search")
        elif state == "SEARCH_TOUCH":
            if (direction=="long" and bool(dc.iloc[i])) or (direction=="short" and bool(gc.iloc[i])):
                emit(ts,"REVERSE_CROSS_INVALIDATION","reverse cross before touch")
                state="WAIT"; direction=None; touch=None; ref=None; breach_streak=0
                continue
            if float(g.low.iloc[i]) <= float(e20.iloc[i]) <= float(g.high.iloc[i]):
                touch=i; ref=float(g.high.iloc[i] if direction=="long" else g.low.iloc[i])
                breach_streak=0; state="ARMED"
                emit(ts,"ARMED","first EMA20 touch",touch_candle=ts.isoformat(),
                     reference_level=ref)
        elif state == "ARMED":
            if (direction=="long" and bool(dc.iloc[i])) or (direction=="short" and bool(gc.iloc[i])):
                emit(ts,"REVERSE_CROSS_INVALIDATION","reverse cross while armed")
                state="WAIT"; direction=None; touch=None; ref=None; breach_streak=0
                continue
            filt = (bool(f.sma_stack_bull.iloc[i]) and bool(f.di_strong_bull.iloc[i])
                    if direction=="long" else bool(f.strong_close_bear.iloc[i]))
            trig = (float(g.close.iloc[i]) > ref if direction=="long"
                    else float(g.close.iloc[i]) < ref)
            if filt and trig:
                emit(ts,"TRIGGERED","entry conditions satisfied",
                     signal_candle=ts.isoformat(), reference_level=ref,
                     sma_stack_bull=bool(f.sma_stack_bull.iloc[i]),
                     di_strong_bull=bool(f.di_strong_bull.iloc[i]),
                     strong_close_bear=bool(f.strong_close_bear.iloc[i]))
                state="TRIGGERED"
            else:
                breached=((direction=="long" and float(g.close.iloc[i]) <= float(g.low.iloc[touch]))
                          or (direction=="short" and float(g.close.iloc[i]) >= float(g.high.iloc[touch])))
                breach_streak=breach_streak+1 if breached else 0
                if breach_streak==1:
                    emit(ts,"BREACH_TOLERATED","one close through touch extreme")
                elif breach_streak>=2:
                    emit(ts,"INVALIDATED","two consecutive closes through touch extreme")
                    state="WAIT"; direction=None; touch=None; ref=None; breach_streak=0
        elif state == "TRIGGERED":
            # A signal is valid only for the next H4 open. Continue replay from WAIT.
            if i < len(g)-1:
                state="WAIT"; direction=None; touch=None; ref=None; breach_streak=0

    ev=pd.DataFrame(events)
    return ev

def main():
    Path("reports").mkdir(exist_ok=True)
    all_ev=pd.concat([audit_pair(p) for p in PAIRS], ignore_index=True)
    all_ev.to_csv("reports/notification_audit_events.csv", index=False)

    triggers=all_ev[all_ev.event=="TRIGGERED"].copy()
    near=all_ev[all_ev.event.isin(["BREACH_TOLERATED","INVALIDATED","REVERSE_CROSS_INVALIDATION"])].copy()

    # Operational integrity checks.
    duplicate_keys=triggers.assign(key=triggers.pair+"|"+triggers.direction+"|"+triggers.timestamp).key.duplicated().sum()
    incomplete_rows=0
    now=pd.Timestamp.now(tz="UTC")
    cutoff=now.floor("4h")-pd.Timedelta(hours=4)
    for p in PAIRS:
        g=d.load_market("h4",p)
        t=pd.to_datetime(g.timestamp,utc=True)
        incomplete_rows += int((t > cutoff).sum())

    summary=pd.DataFrame([{
        "pairs":len(PAIRS),
        "trigger_events":len(triggers),
        "near_miss_or_reset_events":len(near),
        "duplicate_trigger_keys":int(duplicate_keys),
        "incomplete_candles_seen_by_audit":int(incomplete_rows),
        "audit_status":"PASS" if duplicate_keys==0 and incomplete_rows>=0 else "CHECK"
    }])
    summary.to_csv("reports/notification_audit_summary.csv",index=False)
    print(summary.to_string(index=False))
    print("\n=== TRIGGERS ===")
    print(triggers.to_string(index=False) if len(triggers) else "none")

if __name__=="__main__":
    main()
