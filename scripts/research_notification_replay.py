#!/usr/bin/env python3
"""Historical candle-by-candle replay of the notification state machine.
Only completed H4 candles are used; signal is emitted after candle close."""
from pathlib import Path
import importlib.util, pandas as pd, numpy as np
spec=importlib.util.spec_from_file_location("d","scripts/research_50pip_direct_entry.py")
d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
PIP=d.PIP; PAIRS=d.PAIRS

def replay_pair(pair):
    g=d.load_market("h4",pair).reset_index(drop=True); f=d.build_features(g)
    e20=g.close.ewm(span=20,adjust=False).mean(); e200=g.close.ewm(span=200,adjust=False).mean()
    gc=(e20.shift(1)<=e200.shift(1))&(e20>e200)
    dc=(e20.shift(1)>=e200.shift(1))&(e20<e200)
    out=[]; state="WAIT"; direction=None; touch=None; ref=None; armed_at=None
    for i in range(1,len(g)):
        # All conditions below use candle i close and earlier data only.
        if state=="WAIT":
            if bool(gc.iloc[i]): direction="long"; state="SEARCH_TOUCH"
            elif bool(dc.iloc[i]): direction="short"; state="SEARCH_TOUCH"
        elif state=="SEARCH_TOUCH":
            if (direction=="long" and bool(dc.iloc[i])) or (direction=="short" and bool(gc.iloc[i])):
                state="WAIT"; direction=None; continue
            if float(g.low.iloc[i])<=float(e20.iloc[i])<=float(g.high.iloc[i]):
                touch=i; ref=float(g.high.iloc[i] if direction=="long" else g.low.iloc[i]); armed_at=i; state="ARMED"
        elif state=="ARMED":
            if (direction=="long" and bool(dc.iloc[i])) or (direction=="short" and bool(gc.iloc[i])):
                state="WAIT"; direction=None; touch=None; continue
            filt=(bool(f.sma_stack_bull.iloc[i]) and bool(f.di_strong_bull.iloc[i])) if direction=="long" else bool(f.strong_close_bear.iloc[i])
            trig=(float(g.close.iloc[i])>ref) if direction=="long" else (float(g.close.iloc[i])<ref)
            if filt and trig:
                out.append({"pair":pair,"direction":direction,"signal_timestamp":g.timestamp.iloc[i],
                            "touch_timestamp":g.timestamp.iloc[touch],"signal_close":float(g.close.iloc[i]),
                            "state":"TRIGGERED","bars_after_touch":i-touch})
                state="TRIGGERED"
            elif (direction=="long" and float(g.close.iloc[i])<=float(g.low.iloc[touch])) or (direction=="short" and float(g.close.iloc[i])>=float(g.high.iloc[touch])):
                state="WAIT"; direction=None
        elif state=="TRIGGERED":
            state="WAIT"; direction=None; touch=None; ref=None
    return pd.DataFrame(out)

def main():
    frames=[replay_pair(p) for p in PAIRS]
    sig=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    Path("reports").mkdir(exist_ok=True)
    sig.to_csv("reports/notification_replay_signals.csv",index=False)
    summary=(sig.groupby(["direction","pair"]).size().reset_index(name="signals") if len(sig) else pd.DataFrame(columns=["direction","pair","signals"]))
    summary.to_csv("reports/notification_replay_summary.csv",index=False)
    print("signals",len(sig))
if __name__=="__main__": main()
