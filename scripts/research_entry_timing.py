#!/usr/bin/env python3
"""Entry timing study: compare breakout entry vs first pullback/reclaim timing, using only completed candles."""
from pathlib import Path
import importlib.util,numpy as np,pandas as pd
spec=importlib.util.spec_from_file_location("d","scripts/research_50pip_direct_entry.py");d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
PAIRS=d.PAIRS;PIP=d.PIP
CANDS={"long":["sma_stack_bull+di_strong_bull"],"short":["strong_close_bear"]}
def main():
 rows=[]
 for pair in PAIRS:
  g=d.load_market("h4",pair).reset_index(drop=True);f=d.build_features(g)
  e20=g.close.ewm(span=20,adjust=False).mean();e200=g.close.ewm(span=200,adjust=False).mean()
  up=(e20.shift(1)<=e200.shift(1))&(e20>e200);dn=(e20.shift(1)>=e200.shift(1))&(e20<e200)
  pip=PIP[pair];i=1
  while i<len(g)-30:
   direction="long" if up.iloc[i] else ("short" if dn.iloc[i] else None)
   if not direction:i+=1;continue
   touch=None
   for j in range(i+1,min(len(g),i+25)):
    if (direction=="long" and dn.iloc[j]) or (direction=="short" and up.iloc[j]):break
    if g.low.iloc[j]<=e20.iloc[j]<=g.high.iloc[j]:touch=j;break
   if touch is None:i+=1;continue
   rh,rl=float(g.high.iloc[touch]),float(g.low.iloc[touch]);breakidx=None
   for j in range(touch+1,min(len(g),touch+13)):
    if (direction=="long" and dn.iloc[j]) or (direction=="short" and up.iloc[j]):break
    if direction=="long" and g.close.iloc[j]>rh:breakidx=j;break
    if direction=="short" and g.close.iloc[j]<rl:breakidx=j;break
   if breakidx is None:i=touch+1;continue
   filt="sma_stack_bull+di_strong_bull" if direction=="long" else "strong_close_bear"
   if not all(bool(f[x].iloc[breakidx]) for x in filt.split("+")):i=breakidx+1;continue
   bp=float(g.close.iloc[breakidx])
   # Entry modes: breakout close; next candle open; pullback to EMA20 then close back through prior close.
   modes={"break_close":breakidx,"next_open":breakidx+1}
   pull=None
   for j in range(breakidx+1,min(len(g),breakidx+9)):
    if (direction=="long" and dn.iloc[j]) or (direction=="short" and up.iloc[j]):break
    touched=(g.low.iloc[j]<=e20.iloc[j]) if direction=="long" else (g.high.iloc[j]>=e20.iloc[j])
    reclaim=(g.close.iloc[j]>bp) if direction=="long" else (g.close.iloc[j]<bp)
    if touched and reclaim:pull=j;break
   if pull is not None:modes["pullback_reclaim"]=pull
   for mode,ei in modes.items():
    if ei>=len(g):continue
    px=float(g.close.iloc[ei]) if mode!="next_open" else float(g.open.iloc[ei])
    stop=rl if direction=="long" else rh;sign=1 if direction=="long" else -1
    vals={}
    x=g.iloc[ei+1:min(len(g),ei+25)]
    for t in [50,100,150,200,300]:
     val=None
     for k in x.index:
      hi,lo=float(g.high.iloc[k]),float(g.low.iloc[k])
      if (direction=="long" and lo<=stop) or (direction=="short" and hi>=stop):val=sign*(stop-px)/pip;break
      if (direction=="long" and hi>=px+t*pip) or (direction=="short" and lo<=px-t*pip):val=float(t);break
     if val is None:
      k=x.index[-1];val=sign*(float(g.close.iloc[k])-px)/pip
     vals[t]=val
    row={"pair":pair,"direction":direction,"mode":mode,"entry_timestamp":g.timestamp.iloc[ei],"filter":filt}
    row.update({f"pnl{t}":vals[t] for t in vals});rows.append(row)
   i=breakidx+1
 out=pd.DataFrame(rows);out["period"]=np.where(out.entry_timestamp.dt.year<=2024,"discovery",np.where(out.entry_timestamp.dt.year==2025,"validation","oos"))
 Path("reports").mkdir(exist_ok=True);out.to_csv("reports/entry_timing_events.csv",index=False)
 agg=[]
 for keys,z in out.groupby(["direction","mode","period"]):
  for t in [50,100,150,200,300]:
   v=z[f"pnl{t}"];agg.append([*keys,len(z),t,v.mean(),(v>0).mean(),v.sum()])
 pd.DataFrame(agg,columns=["direction","mode","period","trades","target","mean_pips","positive_rate","total_pips"]).to_csv("reports/entry_timing_results.csv",index=False,float_format="%.6f")
 print("DONE",len(out))
if __name__=="__main__":main()
