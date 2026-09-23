#!/usr/bin/env python3
"""Period stability of the frozen notification rules across Discovery/Validation/OOS."""
from pathlib import Path
import importlib.util,pandas as pd,numpy as np
spec=importlib.util.spec_from_file_location("d","scripts/research_50pip_direct_entry.py");d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
PAIRS=d.PAIRS; PIP=d.PIP
RULES={"long":["sma_stack_bull","di_strong_bull"],"short":["strong_close_bear"]}
TARGETS=[100,150,200,300]
PERIODS={"discovery":(2021,2023),"validation":(2024,2025),"oos":(2026,2026)}
rows=[]
for pair in PAIRS:
    g=d.load_market("h4",pair).reset_index(drop=True); f=d.build_features(g)
    e20=g.close.ewm(span=20,adjust=False).mean(); e200=g.close.ewm(span=200,adjust=False).mean()
    up=(e20.shift(1)<=e200.shift(1))&(e20>e200); dn=(e20.shift(1)>=e200.shift(1))&(e20<e200)
    i=1
    while i<len(g)-25:
        direction="long" if up.iloc[i] else ("short" if dn.iloc[i] else None)
        if not direction: i+=1; continue
        touch=None
        for j in range(i+1,min(len(g),i+25)):
            if (direction=="long" and dn.iloc[j]) or (direction=="short" and up.iloc[j]): break
            if g.low.iloc[j]<=e20.iloc[j]<=g.high.iloc[j]: touch=j; break
        if touch is None: i+=1; continue
        rh,rl=float(g.high.iloc[touch]),float(g.low.iloc[touch]); entry=None
        for j in range(touch+1,min(len(g),touch+13)):
            if (direction=="long" and dn.iloc[j]) or (direction=="short" and up.iloc[j]): break
            if (direction=="long" and g.close.iloc[j]>rh) or (direction=="short" and g.close.iloc[j]<rl): entry=j; break
        if entry is None: i=touch+1; continue
        if not all(bool(f[c].iloc[entry]) for c in RULES[direction]): i=entry+1; continue
        if entry+1>=len(g): break
        year=int(pd.Timestamp(g.timestamp.iloc[entry]).year)
        period=next((k for k,(a,b) in PERIODS.items() if a<=year<=b),None)
        if period is None: i=entry+1; continue
        px=float(g.open.iloc[entry+1]); stop=rl if direction=="long" else rh; sign=1 if direction=="long" else -1; pip=PIP[pair]
        x=g.iloc[entry+1:min(len(g),entry+25)]
        for t in TARGETS:
            val=None
            for k in x.index:
                hi,lo=float(g.high.iloc[k]),float(g.low.iloc[k])
                if (direction=="long" and lo<=stop) or (direction=="short" and hi>=stop):
                    val=sign*(stop-px)/pip; break
                if (direction=="long" and hi>=px+t*pip) or (direction=="short" and lo<=px-t*pip):
                    val=float(t); break
            if val is None:
                k=x.index[-1]; val=sign*(float(g.close.iloc[k])-px)/pip
            rows.append([period,pair,direction,t,val])
        i=entry+1
z=pd.DataFrame(rows,columns=["period","pair","direction","target","pnl"])
Path("reports").mkdir(exist_ok=True)
z.to_csv("reports/monitoring_stability_events.csv",index=False,float_format="%.6f")
a=z.groupby(["period","direction","pair","target"]).pnl.agg(["count","mean","sum"]).reset_index()
a["positive_rate"]=z.groupby(["period","direction","pair","target"]).pnl.apply(lambda x:(x>0).mean()).values
a.to_csv("reports/monitoring_stability_detail.csv",index=False,float_format="%.6f")
r=[]
for (period,direction,pair),q in a.groupby(["period","direction","pair"]):
    r.append([period,direction,pair,len(q),(q["mean"]>0).sum(),(q["positive_rate"]>=.5).sum(),q["mean"].mean(),q["positive_rate"].mean()])
s=pd.DataFrame(r,columns=["period","direction","pair","targets","positive_mean_targets","positive_rate_ge50_targets","mean_of_target_means","mean_positive_rate"])
s.to_csv("reports/monitoring_stability_screen.csv",index=False,float_format="%.6f")
pivot=s.pivot_table(index=["direction","pair"],columns="period",values=["positive_mean_targets","positive_rate_ge50_targets","mean_of_target_means"],aggfunc="first").reset_index()
stable=[]
for _,q in pivot.iterrows():
    ok=True
    for p in PERIODS:
        pm=q.get(("positive_mean_targets",p),np.nan); pr=q.get(("positive_rate_ge50_targets",p),np.nan)
        ok &= pd.notna(pm) and pd.notna(pr) and pm>=3 and pr>=2
    stable.append(ok)
pivot["stable_all_periods"]=stable
pivot.to_csv("reports/monitoring_stability_summary.csv",index=False,float_format="%.6f")
print("DONE",len(z))
