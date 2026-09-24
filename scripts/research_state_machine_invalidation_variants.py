#!/usr/bin/env python3
"""Compare pre-entry invalidation rules for the frozen H4 20/200 EMA state machine.

Variants:
- current: close through touch-candle stop invalidates.
- reverse_cross_only: keep the setup alive until reverse EMA cross.
- touch_low_plus_200ema: invalidate only when close breaks touch low AND closes on the wrong side of EMA200.
Discovery <=2024, validation=2025, OOS >=2026.
"""
from pathlib import Path
import importlib.util, numpy as np, pandas as pd

spec=importlib.util.spec_from_file_location("d","scripts/research_50pip_direct_entry.py")
d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
PAIRS=d.PAIRS; PIP=d.PIP
VARIANTS=["current","reverse_cross_only","touch_low_plus_200ema"]

def run_variant(pair,variant):
    g=d.load_market("h4",pair).sort_values("timestamp").reset_index(drop=True)
    f=d.build_features(g)
    e20=g.close.ewm(span=20,adjust=False).mean(); e200=g.close.ewm(span=200,adjust=False).mean()
    gc=(e20.shift(1)<=e200.shift(1))&(e20>e200); dc=(e20.shift(1)>=e200.shift(1))&(e20<e200)
    rows=[]; i=1
    while i<len(g)-25:
        direction="long" if bool(gc.iloc[i]) else ("short" if bool(dc.iloc[i]) else None)
        if direction is None: i+=1; continue
        touch=None
        for j in range(i+1,min(len(g),i+25)):
            if (direction=="long" and bool(dc.iloc[j])) or (direction=="short" and bool(gc.iloc[j])): break
            if float(g.low.iloc[j])<=float(e20.iloc[j])<=float(g.high.iloc[j]):
                touch=j; break
        if touch is None: i+=1; continue
        ref_hi=float(g.high.iloc[touch]); ref_lo=float(g.low.iloc[touch]); entry=None
        for j in range(touch+1,min(len(g),touch+25)):
            if (direction=="long" and bool(dc.iloc[j])) or (direction=="short" and bool(gc.iloc[j])): break
            # pre-entry invalidation
            invalid=False
            if variant=="current":
                invalid=(direction=="long" and float(g.close.iloc[j])<=ref_lo) or (direction=="short" and float(g.close.iloc[j])>=ref_hi)
            elif variant=="touch_low_plus_200ema":
                invalid=((direction=="long" and float(g.close.iloc[j])<=ref_lo and float(g.close.iloc[j])<float(e200.iloc[j])) or
                         (direction=="short" and float(g.close.iloc[j])>=ref_hi and float(g.close.iloc[j])>float(e200.iloc[j])))
            if invalid: break
            if direction=="long" and float(g.close.iloc[j])>ref_hi: entry=j; break
            if direction=="short" and float(g.close.iloc[j])<ref_lo: entry=j; break
        if entry is None: i=touch+1; continue
        pip=PIP[pair]; ep=float(g.close.iloc[entry]); stop=ref_lo if direction=="long" else ref_hi; sign=1 if direction=="long" else -1
        x=g.iloc[entry+1:min(len(g),entry+25)]
        if len(x)==0: break
        outcome=0.0
        for k in x.index:
            hi=float(g.high.iloc[k]); lo=float(g.low.iloc[k])
            if (direction=="long" and lo<=stop) or (direction=="short" and hi>=stop): outcome=sign*(stop-ep)/pip; break
            if (direction=="long" and hi>=ep+100*pip) or (direction=="short" and lo<=ep-100*pip): outcome=100.0; break
            outcome=sign*(float(g.close.iloc[k])-ep)/pip
        rows.append({"pair":pair,"direction":direction,"entry_timestamp":g.timestamp.iloc[entry],"pnl100":outcome,"variant":variant})
        i=entry+1
    return rows

def main():
    rows=[]
    for v in VARIANTS:
        for p in PAIRS: rows.extend(run_variant(p,v))
    out=pd.DataFrame(rows)
    out["period"]=np.where(out.entry_timestamp.dt.year<=2024,"discovery",np.where(out.entry_timestamp.dt.year==2025,"validation","oos"))
    out.to_csv("reports/state_machine_invalidation_variant_events.csv",index=False)
    stats=[]
    for v in VARIANTS:
      for direction in ["long","short","all"]:
       for period in ["discovery","validation","oos"]:
        z=out[(out.variant==v)&(out.period==period)]
        if direction!="all": z=z[z.direction==direction]
        stats.append([v,direction,period,len(z),z.pnl100.mean() if len(z) else np.nan,(z.pnl100>0).mean() if len(z) else np.nan,z.pnl100.sum() if len(z) else 0])
    pd.DataFrame(stats,columns=["variant","direction","period","trades","mean_pips","positive_rate","total_pips"]).to_csv("reports/state_machine_invalidation_variant_results.csv",index=False,float_format="%.6f")
    print(pd.DataFrame(stats,columns=["variant","direction","period","trades","mean_pips","positive_rate","total_pips"]).to_string(index=False))
if __name__=="__main__": main()
