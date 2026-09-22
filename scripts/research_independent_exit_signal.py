#!/usr/bin/env python3
"""Independent exit-signal research after +A target.

Unlike the prior study, every candidate condition is evaluated independently:
for each +A target event, the first candle after +A on which THAT condition is
true is the exit decision. We then ask whether +B was reached afterwards.
Discovery <=2024, validation 2025, OOS >=2026. No pin-bar features.
"""
from pathlib import Path
import importlib.util
from itertools import combinations, product
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("mtf","scripts/research_50pip_mtf_structure.py")
mtf=importlib.util.module_from_spec(spec); spec.loader.exec_module(mtf)
TFS=mtf.TFS
TRANSITIONS=[(50,100),(100,150),(150,200),(200,300)]
WINDOWS={"h4":6,"d1":5}
FAMILIES={
"momentum":["adx_down3","di_spread_down3","macd_accel_down","rsi_down3"],
"trend":["ema20_slope_down3","price20_cross_down","price200_bull"],
"structure":["structure_lh","false_break_up","range_compression","nr4"],
"price_action":["strong_close_bear","close_near_low","pullback_bull"],
"mtf":["mtf_adx_rising","mtf_di_strong_bull","mtf_macd_bull","mtf_trend_bull"],
"volatility":["atr_expand","volatility","bb_expand"],
}
EXPLICIT=[
("adx_down3","di_spread_down3"),("adx_down3","macd_accel_down"),
("di_spread_down3","macd_accel_down"),("adx_down3","structure_lh"),
("price20_cross_down","adx_down3"),("price20_cross_down","di_spread_down3"),
("false_break_up","adx_down3"),("range_compression","adx_down3"),
("strong_close_bear","adx_down3"),("mtf_trend_bull","adx_down3","di_spread_down3"),
]
def as_bool(s):
    if pd.api.types.is_bool_dtype(s): return s.fillna(False).astype(bool)
    return pd.to_numeric(s,errors="coerce").fillna(0).ne(0)
def mask(g,c):
    m=pd.Series(True,index=g.index)
    for n in c:
        if n not in g.columns: return pd.Series(False,index=g.index)
        m &= as_bool(g[n])
    return m
def wilson(k,n):
    if n<=0:return np.nan
    z=1.959963984540054;p=k/n;den=1+z*z/n
    return ((p+z*z/(2*n))-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def events(g,pair):
    g=g.sort_values("timestamp").reset_index(drop=True)
    pip=.01 if pair.endswith("jpy") else .0001
    tf=str(g.timeframe.iloc[0]); horizon=12 if "h4" in tf else 10
    win=WINDOWS["h4" if "h4" in tf else "d1"]
    rows=[]
    for i in range(len(g)):
        entry=float(g.close.iloc[i])
        for a,b in TRANSITIONS:
            alevel=entry+a*pip; blevel=entry+b*pip
            target=None
            for j in range(i+1,min(len(g),i+1+horizon)):
                if float(g.high.iloc[j])>=alevel: target=j; break
            if target is None: continue
            end=min(len(g),target+1+win)
            # Only events where +B has not already been reached at the +A candle.
            if float(g.high.iloc[target])>=blevel: continue
            future=g.iloc[target+1:end]
            if len(future)==0: continue
            for fam,features in FAMILIES.items():
                for feat in features:
                    if feat not in g.columns: continue
                    hits=[j for j in range(target+1,end) if bool(as_bool(g[feat]).iloc[j])]
                    if not hits: continue
                    j=hits[0]; after=g.iloc[j+1:end]
                    if len(after)==0: continue
                    hit=bool((after.high.astype(float)>=blevel).any())
                    px=float(g.close.iloc[j])
                    rows.append({"timeframe":tf,"pair":pair,"from_target":a,"to_target":b,
                        "signal_timestamp":g.timestamp.iloc[i],"target_timestamp":g.timestamp.iloc[target],
                        "exit_timestamp":g.timestamp.iloc[j],"condition":feat,
                        "bars_after_target":j-target,"exit_price_from_target_pips":(px-alevel)/pip,
                        "next_target_after_exit":hit,
                        "future_mfe_from_exit_pips":float((after.high.astype(float).max()-px)/pip),
                        "future_mae_from_exit_pips":float((after.low.astype(float).min()-px)/pip)})
                # one independent condition per row above
            # Explicit combinations: first candle after target satisfying all members.
            for combo in EXPLICIT:
                if not all(x in g.columns for x in combo): continue
                hits=[j for j in range(target+1,end) if bool(mask(g.iloc[[j]],combo).iloc[0])]
                if not hits: continue
                j=hits[0]; after=g.iloc[j+1:end]
                if len(after)==0: continue
                hit=bool((after.high.astype(float)>=blevel).any()); px=float(g.close.iloc[j])
                rows.append({"timeframe":tf,"pair":pair,"from_target":a,"to_target":b,
                    "signal_timestamp":g.timestamp.iloc[i],"target_timestamp":g.timestamp.iloc[target],
                    "exit_timestamp":g.timestamp.iloc[j],"condition":"+".join(combo),
                    "bars_after_target":j-target,"exit_price_from_target_pips":(px-alevel)/pip,
                    "next_target_after_exit":hit,
                    "future_mfe_from_exit_pips":float((after.high.astype(float).max()-px)/pip),
                    "future_mae_from_exit_pips":float((after.low.astype(float).min()-px)/pip)})
    return pd.DataFrame(rows)

def pool(ev,a):
    q=ev[(ev.from_target==a)&(ev.exit_timestamp.dt.year<=2024)]
    ranked=[]
    for c,g in q.groupby("condition"):
        n=len(g)
        if n<30: continue
        fail=float((~g.next_target_after_exit).mean())
        ranked.append((c,n,fail))
    return [x[0] for x in sorted(ranked,key=lambda x:(x[2],x[1]),reverse=True)[:40]]

def main():
    Path("reports").mkdir(exist_ok=True)
    all_ev=[]
    for tf in TFS:
        ds=mtf.build_dataset(tf)
        for pair,g in ds.groupby("pair",sort=False):
            x=events(g,pair)
            if len(x): all_ev.append(x)
    ev=pd.concat(all_ev,ignore_index=True)
    rows=[]; rob=[]
    for tf in TFS:
        e=ev[ev.timeframe==tf]
        for a,b in TRANSITIONS:
            cand=pool(e,a)
            for c in cand:
                out=[]
                for period,name in [(e[e.exit_timestamp.dt.year<=2024],"discovery"),
                                    (e[e.exit_timestamp.dt.year==2025],"validation"),
                                    (e[e.exit_timestamp.dt.year>=2026],"oos")]:
                    x=period[(period.from_target==a)&(period.to_target==b)&(period.condition==c)]
                    n=len(x); fail=float((~x.next_target_after_exit).mean()) if n else np.nan
                    base=float((~period[(period.from_target==a)&(period.to_target==b)].next_target_after_exit).mean())
                    k=int((~x.next_target_after_exit).sum()) if n else 0
                    rows.append([tf,name,a,b,c,n,fail,base,fail/base if n and base else np.nan,
                        wilson(k,n),float(x.exit_price_from_target_pips.mean()) if n else np.nan,
                        float(x.future_mfe_from_exit_pips.mean()) if n else np.nan,
                        float(x.future_mae_from_exit_pips.mean()) if n else np.nan,
                        float(x.bars_after_target.mean()) if n else np.nan])
                x=e[(e.exit_timestamp.dt.year>=2026)&(e.from_target==a)&(e.to_target==b)&(e.condition==c)]
                prs=[float((~z.next_target_after_exit).mean()) for _,z in x.groupby("pair") if len(z)>=5]
                if len(x) and prs:
                    rob.append([tf,a,b,c,len(x),float((~x.next_target_after_exit).mean()),len(prs),
                                float(np.mean(prs)),float(min(prs))])
    pd.DataFrame(rows,columns=["timeframe","period","from_target","to_target","condition","samples",
        "exit_precision","baseline_exit_precision","lift","wilson95_lower","exit_price_from_target_pips_mean",
        "future_mfe_from_exit_pips_mean","future_mae_from_exit_pips_mean","bars_after_target_mean"]).to_csv(
        "reports/independent_exit_signal.csv",index=False,float_format="%.8f")
    pd.DataFrame(rob,columns=["timeframe","from_target","to_target","condition","oos_n",
        "oos_exit_precision","oos_pairs_n_ge5","oos_pair_precision_mean","oos_pair_precision_min"]).to_csv(
        "reports/independent_exit_signal_robustness.csv",index=False,float_format="%.8f")
    print("DONE",len(rows),len(rob),flush=True)
if __name__=="__main__": main()
