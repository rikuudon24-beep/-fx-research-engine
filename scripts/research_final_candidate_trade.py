#!/usr/bin/env python3
"""Final candidate trade-level ladder using strict walk-forward-selected filters."""
from pathlib import Path
import importlib.util,numpy as np,pandas as pd
spec=importlib.util.spec_from_file_location("d","scripts/research_50pip_direct_entry.py"); d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
PAIRS=d.PAIRS; PIP=d.PIP
CANDS={"long":["sma_stack_bull+di_strong_bull"],"short":["strong_close_bear","strong_close_bear+di_strong_bear","breakout20_down"]}
TARGETS=[100,150,200,300]

def events():
 rows=[]
 for pair in PAIRS:
  g=d.load_market("h4",pair).reset_index(drop=True); f=d.build_features(g)
  e20=g.close.ewm(span=20,adjust=False).mean(); e200=g.close.ewm(span=200,adjust=False).mean()
  up=(e20.shift(1)<=e200.shift(1))&(e20>e200); dn=(e20.shift(1)>=e200.shift(1))&(e20<e200)
  xl=f.price20_cross_down&f.di_spread_down3; xs=f.price20_cross_up&f.di_spread_up3
  pip=PIP[pair]; i=1
  while i<len(g)-25:
   direction="long" if up.iloc[i] else ("short" if dn.iloc[i] else None)
   if not direction:i+=1;continue
   touch=None
   for j in range(i+1,min(len(g),i+25)):
    if (direction=="long" and dn.iloc[j]) or (direction=="short" and up.iloc[j]):break
    if g.low.iloc[j]<=e20.iloc[j]<=g.high.iloc[j]:touch=j;break
   if touch is None:i+=1;continue
   rh,rl=float(g.high.iloc[touch]),float(g.low.iloc[touch]); entry=None
   for j in range(touch+1,min(len(g),touch+13)):
    if (direction=="long" and dn.iloc[j]) or (direction=="short" and up.iloc[j]):break
    if direction=="long" and g.close.iloc[j]>rh:entry=j;break
    if direction=="short" and g.close.iloc[j]<rl:entry=j;break
   if entry is None:i=touch+1;continue
   px=float(g.close.iloc[entry]); stop=rl if direction=="long" else rh; sign=1 if direction=="long" else -1
   x=g.iloc[entry+1:min(len(g),entry+25)]
   row={"pair":pair,"direction":direction,"entry_timestamp":g.timestamp.iloc[entry]}
   for c in CANDS[direction]: row[c]=all(bool(f[k].iloc[entry]) for k in c.split("+"))
   # Store target outcomes with conservative stop-first ordering.
   for t in TARGETS:
    val=None; bars=None
    for k in x.index:
     hi,lo=float(g.high.iloc[k]),float(g.low.iloc[k])
     if (direction=="long" and lo<=stop) or (direction=="short" and hi>=stop):
      val=sign*(stop-px)/pip;bars=k-entry;break
     if (direction=="long" and hi>=px+t*pip) or (direction=="short" and lo<=px-t*pip):
      val=float(t);bars=k-entry;break
    if val is None:
     k=x.index[-1];val=sign*(float(g.close.iloc[k])-px)/pip;bars=k-entry
    row[f"pnl{t}"]=val;row[f"bars{t}"]=bars
   # Managed exit: after +50, first deterioration signal; before +50 stop.
   mval=None;mb=None;plus50=False
   for k in x.index:
    hi,lo=float(g.high.iloc[k]),float(g.low.iloc[k])
    if (direction=="long" and lo<=stop) or (direction=="short" and hi>=stop):
     mval=sign*(stop-px)/pip;mb=k-entry;break
    if (direction=="long" and hi>=px+50*pip) or (direction=="short" and lo<=px-50*pip):plus50=True
    if plus50 and bool((xl if direction=="long" else xs).iloc[k]):
     mval=sign*(float(g.close.iloc[k])-px)/pip;mb=k-entry;break
    if (direction=="long" and hi>=px+200*pip) or (direction=="short" and lo<=px-200*pip):
     mval=200.;mb=k-entry;break
   if mval is None:
    k=x.index[-1];mval=sign*(float(g.close.iloc[k])-px)/pip;mb=k-entry
   row["managed_pips"]=mval;row["managed_bars"]=mb;rows.append(row);i=entry+1
 out=pd.DataFrame(rows);out["period"]=np.where(out.entry_timestamp.dt.year<=2024,"discovery",np.where(out.entry_timestamp.dt.year==2025,"validation","oos"));return out

def main():
 out=events();Path("reports").mkdir(exist_ok=True);out.to_csv("reports/final_candidate_events.csv",index=False)
 rows=[]
 for direction,names in CANDS.items():
  for name in names:
   for p in ["discovery","validation","oos"]:
    z=out[(out.direction==direction)&(out.period==p)&out[name]]
    for t in TARGETS:
     v=z[f"pnl{t}"];rows.append([direction,name,p,len(z),t,v.mean() if len(z) else np.nan,(v>0).mean() if len(z) else np.nan,v.sum() if len(z) else np.nan,z[f"bars{t}"].mean() if len(z) else np.nan])
    m=z.managed_pips;rows.append([direction,name,p,len(z),"managed",m.mean() if len(z) else np.nan,(m>0).mean() if len(z) else np.nan,m.sum() if len(z) else np.nan,z.managed_bars.mean() if len(z) else np.nan])
 pd.DataFrame(rows,columns=["direction","filter","period","trades","target","mean_pips","positive_rate","total_pips","mean_bars"]).to_csv("reports/final_candidate_results.csv",index=False,float_format="%.6f")
 print("DONE",len(out))
if __name__=="__main__":main()
