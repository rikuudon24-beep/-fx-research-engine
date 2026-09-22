#!/usr/bin/env python3
"""Connect validated state-machine entry filters to a multi-target trade ladder.

Candidate filters were selected without OOS data from the prior entry-filter
study. This script replays the same causal entry, reference stop and 24-bar
window, then evaluates +100/+150/+200/+300 target policies. OOS is only scored
after discovery/validation selection.
"""
from pathlib import Path
import importlib.util, itertools, numpy as np, pandas as pd

spec=importlib.util.spec_from_file_location("d","scripts/research_50pip_direct_entry.py")
d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
PAIRS=d.PAIRS; PIP=d.PIP
CANDS={
"long":["rsi_cross55_up","strong_close_bull+rsi_cross55_up","rsi_cross55_up+trendline_up"],
"short":["breakout20_down","strong_close_bear","strong_close_bear+di_strong_bear","volatility+strong_close_bear"]
}
TARGETS=[100,150,200,300]

def make_events():
    rows=[]
    for pair in PAIRS:
        g=d.load_market("h4",pair).reset_index(drop=True); f=d.build_features(g)
        e20=g.close.ewm(span=20,adjust=False).mean(); e200=g.close.ewm(span=200,adjust=False).mean()
        cross_up=(e20.shift(1)<=e200.shift(1))&(e20>e200); cross_down=(e20.shift(1)>=e200.shift(1))&(e20<e200)
        exit_long=f.price20_cross_down&f.di_spread_down3; exit_short=f.price20_cross_up&f.di_spread_up3
        pip=PIP[pair]; i=1
        while i<len(g)-25:
            direction="long" if bool(cross_up.iloc[i]) else ("short" if bool(cross_down.iloc[i]) else None)
            if direction is None: i+=1; continue
            touch=None
            for j in range(i+1,min(len(g),i+25)):
                if (direction=="long" and bool(cross_down.iloc[j])) or (direction=="short" and bool(cross_up.iloc[j])): break
                if float(g.low.iloc[j])<=float(e20.iloc[j])<=float(g.high.iloc[j]): touch=j; break
            if touch is None: i+=1; continue
            rh,rl=float(g.high.iloc[touch]),float(g.low.iloc[touch]); entry=None
            for j in range(touch+1,min(len(g),touch+13)):
                if (direction=="long" and bool(cross_down.iloc[j])) or (direction=="short" and bool(cross_up.iloc[j])): break
                if direction=="long" and g.close.iloc[j]>rh: entry=j; break
                if direction=="short" and g.close.iloc[j]<rl: entry=j; break
            if entry is None: i=touch+1; continue
            px=float(g.close.iloc[entry]); stop=rl if direction=="long" else rh; sign=1 if direction=="long" else -1
            vals={c:bool(f[c].iloc[entry]) for c in f.columns}
            x=g.iloc[entry+1:min(len(g),entry+25)]
            row={"pair":pair,"direction":direction,"entry_timestamp":g.timestamp.iloc[entry],"entry_price":px,"stop_price":stop}
            for c,v in vals.items(): row[c]=v
            for t in TARGETS:
                hit=None
                for k in x.index:
                    hi,lo=float(g.high.iloc[k]),float(g.low.iloc[k])
                    if (direction=="long" and lo<=stop) or (direction=="short" and hi>=stop): break
                    if (direction=="long" and hi>=px+t*pip) or (direction=="short" and lo<=px-t*pip):
                        hit=k; break
                if hit is not None:
                    row[f"target{t}_pips"]=float(t); row[f"target{t}_bars"]=int(hit-entry)
                else:
                    # If stop occurs, use stop P/L; otherwise use final close.
                    stop_hit=False; end=x.index[-1]
                    for k in x.index:
                        hi,lo=float(g.high.iloc[k]),float(g.low.iloc[k])
                        if (direction=="long" and lo<=stop) or (direction=="short" and hi>=stop):
                            row[f"target{t}_pips"]=sign*(stop-px)/pip; row[f"target{t}_bars"]=int(k-entry); stop_hit=True; break
                    if not stop_hit:
                        row[f"target{t}_pips"]=sign*(float(g.close.iloc[end])-px)/pip; row[f"target{t}_bars"]=int(end-entry)
            rows.append(row); i=entry+1
    out=pd.DataFrame(rows)
    out["period"]=np.where(out.entry_timestamp.dt.year<=2024,"discovery",np.where(out.entry_timestamp.dt.year==2025,"validation","oos"))
    return out

def apply_filter(z,name):
    for c in name.split("+"): z=z[z[c]]
    return z

def main():
    out=make_events(); Path("reports").mkdir(exist_ok=True)
    rows=[]
    for direction,names in CANDS.items():
        for name in names:
            for period in ["discovery","validation","oos"]:
                z=apply_filter(out[(out.direction==direction)&(out.period==period)],name)
                for t in TARGETS:
                    v=z[f"target{t}_pips"]
                    rows.append([direction,name,period,len(z),t,v.mean(),(v>0).mean(),v.sum(),z[f"target{t}_bars"].mean()])
    res=pd.DataFrame(rows,columns=["direction","filter","period","trades","target","mean_pips","positive_rate","total_pips","mean_bars"])
    res.to_csv("reports/state_machine_target_ladder_results.csv",index=False,float_format="%.6f")
    # Pair robustness for OOS candidates.
    pr=[]
    for direction,names in CANDS.items():
        for name in names:
            z=apply_filter(out[(out.direction==direction)&(out.period=="oos")],name)
            for pair,g in z.groupby("pair"):
                pr.append([direction,name,pair,len(g),g.target100_pips.mean(),g.target100_pips.gt(0).mean()])
    pd.DataFrame(pr,columns=["direction","filter","pair","trades","target100_mean","positive_rate"]).to_csv(
        "reports/state_machine_target_ladder_oos_pair.csv",index=False,float_format="%.6f")
    print("DONE",len(out),"base trades")
if __name__=="__main__": main()
