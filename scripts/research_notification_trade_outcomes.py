#!/usr/bin/env python3
"""Convert replay notifications into actual next-open trade outcomes."""
from pathlib import Path
import importlib.util,pandas as pd,numpy as np
spec=importlib.util.spec_from_file_location("d","scripts/research_50pip_direct_entry.py");d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
PIP=d.PIP;PAIRS=d.PAIRS
def main():
 sig=pd.read_csv("reports/notification_replay_signals.csv",parse_dates=["signal_timestamp"])
 rows=[]
 for pair in PAIRS:
  z=sig[sig.pair==pair]
  if z.empty: continue
  g=d.load_market("h4",pair).reset_index(drop=True); idx={pd.Timestamp(t):i for i,t in enumerate(g.timestamp)}
  for _,s in z.iterrows():
   si=idx.get(pd.Timestamp(s.signal_timestamp))
   if si is None or si+1>=len(g): continue
   direction=s.direction; e=si+1; px=float(g.open.iloc[e])
   # Recreate stop from the signal's touch candle.
   ti=idx.get(pd.Timestamp(s.touch_timestamp))
   if ti is None: continue
   stop=float(g.low.iloc[ti] if direction=="long" else g.high.iloc[ti]); sign=1 if direction=="long" else -1; pip=PIP[pair]
   vals={}
   for t in [50,100,150,200,300]:
    val=None;bars=None
    for k in range(e,min(len(g),e+25)):
     hi,lo=float(g.high.iloc[k]),float(g.low.iloc[k])
     if (direction=="long" and lo<=stop) or (direction=="short" and hi>=stop):
      val=sign*(stop-px)/pip;bars=k-e;break
     if (direction=="long" and hi>=px+t*pip) or (direction=="short" and lo<=px-t*pip):
      val=float(t);bars=k-e;break
    if val is None:
     k=min(len(g)-1,e+24);val=sign*(float(g.close.iloc[k])-px)/pip;bars=k-e
    vals[t]=(val,bars)
   row={"pair":pair,"direction":direction,"signal_timestamp":s.signal_timestamp,"entry_timestamp":g.timestamp.iloc[e],"entry_price":px,"stop_pips":sign*(stop-px)/pip,"period":("discovery" if g.timestamp.iloc[e].year<=2024 else "validation" if g.timestamp.iloc[e].year==2025 else "oos")}
   for t,(v,b) in vals.items():row[f"pnl{t}"]=v;row[f"bars{t}"]=b
   rows.append(row)
 out=pd.DataFrame(rows);Path("reports").mkdir(exist_ok=True);out.to_csv("reports/notification_trade_outcomes.csv",index=False,float_format="%.6f")
 agg=[]
 for (direction,period),q in out.groupby(["direction","period"]):
  for t in [50,100,150,200,300]:
   v=q[f"pnl{t}"];agg.append([direction,period,len(q),t,v.mean(),(v>0).mean(),v.sum()])
 pd.DataFrame(agg,columns=["direction","period","trades","target","mean_pips","positive_rate","total_pips"]).to_csv("reports/notification_trade_outcomes_summary.csv",index=False,float_format="%.6f")
 print("DONE",len(out))
if __name__=="__main__":main()
