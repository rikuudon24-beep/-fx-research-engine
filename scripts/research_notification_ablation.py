#!/usr/bin/env python3
"""Ablation study: remove individual components from the current notification rules.
Uses the same H4 state machine and completed-candle logic; OOS held out."""
from pathlib import Path
import importlib.util,pandas as pd,numpy as np
spec=importlib.util.spec_from_file_location("d","scripts/research_50pip_direct_entry.py");d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
PAIRS=d.PAIRS;PIP=d.PIP
BASE={"long":["sma_stack_bull","di_strong_bull"],"short":["strong_close_bear"]}
def run():
 rows=[]
 for pair in PAIRS:
  g=d.load_market("h4",pair).reset_index(drop=True);f=d.build_features(g)
  e20=g.close.ewm(span=20,adjust=False).mean();e200=g.close.ewm(span=200,adjust=False).mean()
  up=(e20.shift(1)<=e200.shift(1))&(e20>e200);dn=(e20.shift(1)>=e200.shift(1))&(e20<e200)
  i=1
  while i<len(g)-25:
   direction="long" if up.iloc[i] else ("short" if dn.iloc[i] else None)
   if not direction:i+=1;continue
   touch=None
   for j in range(i+1,min(len(g),i+25)):
    if (direction=="long" and dn.iloc[j]) or (direction=="short" and up.iloc[j]):break
    if g.low.iloc[j]<=e20.iloc[j]<=g.high.iloc[j]:touch=j;break
   if touch is None:i+=1;continue
   rh,rl=float(g.high.iloc[touch]),float(g.low.iloc[touch]);entry=None
   for j in range(touch+1,min(len(g),touch+13)):
    if (direction=="long" and dn.iloc[j]) or (direction=="short" and up.iloc[j]):break
    if (direction=="long" and g.close.iloc[j]>rh) or (direction=="short" and g.close.iloc[j]<rl):entry=j;break
   if entry is None:i=touch+1;continue
   configs=[("full",BASE[direction])]
   if direction=="long":
    configs += [("no_sma",["di_strong_bull"]),("no_di",["sma_stack_bull"]),("none",[])]
   else:
    configs += [("none",[])]
   for name,conds in configs:
    ok=all(bool(f[c].iloc[entry]) for c in conds)
    if not ok: continue
    px=float(g.close.iloc[entry]);stop=rl if direction=="long" else rh;sign=1 if direction=="long" else -1
    x=g.iloc[entry+1:min(len(g),entry+25)]
    for t in [100,150,200,300]:
     val=None
     for k in x.index:
      hi,lo=float(g.high.iloc[k]),float(g.low.iloc[k])
      if (direction=="long" and lo<=stop) or (direction=="short" and hi>=stop):val=sign*(stop-px)/PIP[pair];break
      if (direction=="long" and hi>=px+t*PIP[pair]) or (direction=="short" and lo<=px-t*PIP[pair]):val=float(t);break
     if val is None:
      k=x.index[-1];val=sign*(float(g.close.iloc[k])-px)/PIP[pair]
     period="discovery" if g.timestamp.iloc[entry].year<=2024 else "validation" if g.timestamp.iloc[entry].year==2025 else "oos"
     rows.append([pair,direction,name,period,val])
   i=entry+1
 return pd.DataFrame(rows,columns=["pair","direction","config","period","pnl"])
def main():
 z=run();Path("reports").mkdir(exist_ok=True);z.to_csv("reports/notification_ablation_events.csv",index=False)
 a=z.groupby(["direction","config","period"]).pnl.agg(["count","mean","sum"]).reset_index();a["positive_rate"]=z.groupby(["direction","config","period"]).pnl.apply(lambda x:(x>0).mean()).values
 a.to_csv("reports/notification_ablation_results.csv",index=False,float_format="%.6f");print("DONE",len(z))
if __name__=="__main__":main()
