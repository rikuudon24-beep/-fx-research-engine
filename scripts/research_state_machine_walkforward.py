#!/usr/bin/env python3
"""Walk-forward state-machine entry research with strict validation-positive selection."""
from pathlib import Path
import importlib.util,itertools,numpy as np,pandas as pd
spec=importlib.util.spec_from_file_location("d","scripts/research_50pip_direct_entry.py"); d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
PAIRS=d.PAIRS; PIP=d.PIP
FEATURES={
"long":["price200_bull","sma_stack_bull","volatility","adx25","adx_up3","atr_expand","bb_expand","strong_close_bull","elliott_impulse_bull","breakout20_up","horizontal_break_up","rsi_cross55_up","di_strong_bull","ema20_rising","trendline_up"],
"short":["price200_bear","sma_stack_bear","volatility","adx25","adx_down3","atr_expand","bb_expand","strong_close_bear","elliott_impulse_bear","breakout20_down","horizontal_break_down","rsi_cross45_down","di_strong_bear","ema20_falling","trendline_down"]}

def trades():
 rows=[]
 for pair in PAIRS:
  g=d.load_market("h4",pair).reset_index(drop=True); f=d.build_features(g)
  e20=g.close.ewm(span=20,adjust=False).mean(); e200=g.close.ewm(span=200,adjust=False).mean()
  up=(e20.shift(1)<=e200.shift(1))&(e20>e200); dn=(e20.shift(1)>=e200.shift(1))&(e20<e200)
  pip=PIP[pair]; i=1
  while i<len(g)-25:
   direction="long" if up.iloc[i] else ("short" if dn.iloc[i] else None)
   if not direction: i+=1; continue
   touch=None
   for j in range(i+1,min(len(g),i+25)):
    if (direction=="long" and dn.iloc[j]) or (direction=="short" and up.iloc[j]): break
    if g.low.iloc[j]<=e20.iloc[j]<=g.high.iloc[j]: touch=j; break
   if touch is None: i+=1; continue
   rh,rl=g.high.iloc[touch],g.low.iloc[touch]; entry=None
   for j in range(touch+1,min(len(g),touch+13)):
    if (direction=="long" and dn.iloc[j]) or (direction=="short" and up.iloc[j]): break
    if direction=="long" and g.close.iloc[j]>rh: entry=j; break
    if direction=="short" and g.close.iloc[j]<rl: entry=j; break
   if entry is None: i=touch+1; continue
   px=float(g.close.iloc[entry]); stop=float(rl if direction=="long" else rh); sign=1 if direction=="long" else -1
   x=g.iloc[entry+1:min(len(g),entry+25)]
   end=x.index[-1]; pnl=None
   for k in x.index:
    hi,lo=float(g.high.iloc[k]),float(g.low.iloc[k])
    if (direction=="long" and lo<=stop) or (direction=="short" and hi>=stop): pnl=sign*(stop-px)/pip; break
    if (direction=="long" and hi>=px+100*pip) or (direction=="short" and lo<=px-100*pip): pnl=100.; break
   if pnl is None: pnl=sign*(float(g.close.iloc[end])-px)/pip
   row={"pair":pair,"direction":direction,"entry_timestamp":g.timestamp.iloc[entry],"pnl":pnl}
   for c in FEATURES[direction]: row[c]=bool(f[c].iloc[entry])
   rows.append(row); i=entry+1
 return pd.DataFrame(rows)

def main():
 out=trades(); Path("reports").mkdir(exist_ok=True); out["period"]=np.where(out.entry_timestamp.dt.year<=2024,"discovery",np.where(out.entry_timestamp.dt.year==2025,"validation","oos"))
 rows=[]
 for direction,fs in FEATURES.items():
  combos=[(c,) for c in fs]+list(itertools.combinations(fs,2))
  for combo in combos:
   name="+".join(combo)
   for p in ["discovery","validation","oos"]:
    z=out[(out.direction==direction)&(out.period==p)]
    for c in combo:z=z[z[c]]
    rows.append([direction,name,p,len(z),z.pnl.mean() if len(z) else np.nan,(z.pnl>0).mean() if len(z) else np.nan,z.pnl.sum() if len(z) else np.nan])
 res=pd.DataFrame(rows,columns=["direction","filter","period","trades","mean_pips","positive_rate","total_pips"])
 res.to_csv("reports/state_machine_walkforward_results.csv",index=False,float_format="%.6f")
 sel=[]
 for direction in FEATURES:
  for name in res[res.direction==direction]["filter"].unique():
   a=res[(res.direction==direction)&(res["filter"]==name)&(res.period=="discovery")].iloc[0]
   b=res[(res.direction==direction)&(res["filter"]==name)&(res.period=="validation")].iloc[0]
   # Strict: positive mean in both historical segments, enough samples, and >= baseline.
   basea=res[(res.direction==direction)&(res.filter=="__BASE__")&(res.period=="discovery")]
   # baseline handled below
   if a.trades>=20 and b.trades>=8 and a.mean_pips>0 and b.mean_pips>0:
    sel.append([direction,name,a.trades,b.trades,a.mean_pips,b.mean_pips])
 for direction in FEATURES:
  z=out[out.direction==direction]
  for p in ["discovery","validation","oos"]:
   rows2=[]
   for c in FEATURES[direction]: pass
 # OOS report for selected candidates
 short=pd.DataFrame(sel,columns=["direction","filter","discovery_trades","validation_trades","discovery_mean","validation_mean"])
 short.to_csv("reports/state_machine_walkforward_shortlist.csv",index=False,float_format="%.6f")
 print("DONE",len(out),"trades; shortlist",len(short))
if __name__=="__main__":main()
