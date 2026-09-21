#!/usr/bin/env python3
"""Momentum-decay / exit research after a completed +A target.

A target event is the first completed candle whose high reaches +A from the
original signal close. From that target candle onward, we scan for the first
causal momentum-decay condition. The decay candle itself is the decision point;
all exit-quality metrics use only information available by its close.

Discovery <=2024, validation 2025, OOS >=2026. No pin-bar features.
"""
from pathlib import Path
import importlib.util
from itertools import combinations
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("mtf","scripts/research_50pip_mtf_structure.py")
mtf=importlib.util.module_from_spec(spec); spec.loader.exec_module(mtf)

TFS=mtf.TFS
TRANSITIONS=[(50,100),(100,150),(150,200),(200,300)]
WINDOWS={"h4":6,"d1":5}

DECAY_FAMILIES={
"momentum":["adx_down3","di_spread_down3","macd_accel_down","rsi_down3"],
"trend":["ema20_slope_down3","price20_cross_down","price200_bull"],
"structure":["structure_lh","false_break_up","range_compression","nr4"],
"price_action":["strong_close_bear","close_near_low","pullback_bull"],
"mtf":["mtf_adx_rising","mtf_di_strong_bull","mtf_macd_bull","mtf_trend_bull"],
"volatility":["atr_expand","volatility","bb_expand"],
}
EXPLICIT=[
("adx_down3","di_spread_down3"),
("adx_down3","macd_accel_down"),
("di_spread_down3","macd_accel_down"),
("adx_down3","structure_lh"),
("di_spread_down3","structure_lh"),
("macd_accel_down","structure_lh"),
("price20_cross_down","adx_down3"),
("price20_cross_down","di_spread_down3"),
("false_break_up","adx_down3"),
("range_compression","adx_down3"),
("strong_close_bear","adx_down3"),
("mtf_trend_bull","adx_down3","di_spread_down3"),
]

def as_bool(s):
    return s.fillna(False).astype(bool) if pd.api.types.is_bool_dtype(s) else pd.to_numeric(s,errors="coerce").fillna(0).ne(0)

def mask(g,c):
    m=pd.Series(True,index=g.index)
    for n in c:
        if n not in g.columns:return pd.Series(False,index=g.index)
        m &= as_bool(g[n])
    return m

def wilson(k,n):
    if n<=0:return np.nan
    z=1.959963984540054;p=k/n;den=1+z*z/n
    return ((p+z*z/(2*n))-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def build_decay_events(g,pair):
    g=g.sort_values("timestamp").reset_index(drop=True)
    pip=.01 if pair.endswith("jpy") else .0001
    feature_cols=[c for c in g.columns if c not in {"timestamp","open","high","low","close","pair","timeframe"}]
    rows=[]
    for i in range(len(g)):
        entry=float(g.close.iloc[i])
        for a,b in TRANSITIONS:
            level=entry+a*pip
            target=None
            for j in range(i+1,min(len(g),i+13)):
                if float(g.high.iloc[j])>=level:
                    target=j;break
            if target is None: continue
            horizon=WINDOWS["h4" if "h4" in str(g.timeframe.iloc[0]) else "d1"]
            end=min(len(g),target+1+horizon)
            # Search first decay candle strictly after target.
            decay=None
            for j in range(target+1,end):
                if any(bool(as_bool(pd.Series([g[n].iloc[j]]))[j*0+0]) for n in []):
                    pass
                cond=(
                    bool(g.get("adx_down3",pd.Series(False,index=g.index)).iloc[j]) or
                    bool(g.get("di_spread_down3",pd.Series(False,index=g.index)).iloc[j]) or
                    bool(g.get("macd_accel_down",pd.Series(False,index=g.index)).iloc[j]) or
                    bool(g.get("structure_lh",pd.Series(False,index=g.index)).iloc[j]) or
                    bool(g.get("price20_cross_down",pd.Series(False,index=g.index)).iloc[j]) or
                    bool(g.get("false_break_up",pd.Series(False,index=g.index)).iloc[j])
                )
                if cond: decay=j;break
            if decay is None: continue
            future=g.iloc[decay+1:end]
            if len(future)==0: continue
            next_hit=bool((future.high.astype(float)>=entry+b*pip).any())
            stall_close=float(g.close.iloc[decay])
            mfe=float((future.high.astype(float).max()-stall_close)/pip)
            mae=float((future.low.astype(float).min()-stall_close)/pip)
            retrace=float((stall_close-level)/pip)
            rec={"timeframe":g.timeframe.iloc[0],"pair":pair,"from_target":a,"to_target":b,
                 "signal_timestamp":g.timestamp.iloc[i],"target_timestamp":g.timestamp.iloc[target],
                 "decay_timestamp":g.timestamp.iloc[decay],"bars_after_target":decay-target,
                 "target_capture_pips":retrace,"next_target_after_decay":next_hit,
                 "future_mfe_pips":mfe,"future_mae_pips":mae}
            for c in feature_cols: rec[c]=g[c].iloc[decay]
            rows.append(rec)
    return pd.DataFrame(rows)

def candidate_pool(df,a):
    q=df[(df.from_target==a)&(df.decay_timestamp.dt.year<=2024)]
    base=1-float(q.next_target_after_decay.mean()) if len(q) else np.nan
    selected={}
    for fam,features in DECAY_FAMILIES.items():
        vals=[]
        for n in features:
            if n not in q.columns:continue
            m=as_bool(q[n]); nobs=int(m.sum())
            if nobs<30:continue
            fail=float((~q.loc[m,"next_target_after_decay"]).mean())
            vals.append((n,fail,nobs))
        selected[fam]=[x[0] for x in sorted(vals,key=lambda z:(z[1],z[2]),reverse=True)[:3]]
    out=[c for c in EXPLICIT if all(x in q.columns for x in c)]
    seen=set(out)
    fams=list(selected)
    for k in (2,3):
        for fs in combinations(fams,k):
            pools=[selected[f] for f in fs]
            for c in __import__("itertools").product(*pools):
                if len(set(c))<k or c in seen:continue
                seen.add(c);out.append(c)
    return out

def main():
    Path("reports").mkdir(exist_ok=True)
    scores=[];robust=[]
    for tf in TFS:
        data=mtf.build_dataset(tf)
        ev=[]
        for pair,g in data.groupby("pair",sort=False):
            x=build_decay_events(g,pair)
            if len(x):ev.append(x)
        ev=pd.concat(ev,ignore_index=True)
        for a,b in TRANSITIONS:
            disc=ev[(ev.from_target==a)&(ev.decay_timestamp.dt.year<=2024)]
            combos=candidate_pool(ev,a)
            ranked=[]
            for c in combos:
                m=mask(disc,c);n=int(m.sum())
                if n<30:continue
                fail=float((~disc.loc[m,"next_target_after_decay"]).mean())
                ranked.append((c,n,fail))
            ranked=sorted(ranked,key=lambda x:(x[2],x[1]),reverse=True)[:80]
            for c,dn,dfail in ranked:
                for period,name in [(ev[ev.decay_timestamp.dt.year<=2024],"discovery"),
                                    (ev[ev.decay_timestamp.dt.year==2025],"validation"),
                                    (ev[ev.decay_timestamp.dt.year>=2026],"oos")]:
                    p=period[period.from_target==a];m=mask(p,c);x=p[m];n=len(x)
                    fail=float((~x.next_target_after_decay).mean()) if n else np.nan
                    base=float((~p.next_target_after_decay).mean()) if len(p) else np.nan
                    k=int((~x.next_target_after_decay).sum()) if n else 0
                    scores.append([tf,name,a,b,"+".join(c),n,fail,base,fail/base if n and base else np.nan,
                                   wilson(k,n),float(x.target_capture_pips.mean()) if n else np.nan,
                                   float(x.future_mfe_pips.mean()) if n else np.nan,
                                   float(x.future_mae_pips.mean()) if n else np.nan,
                                   float(x.bars_after_target.mean()) if n else np.nan])
                o=ev[(ev.decay_timestamp.dt.year>=2026)&(ev.from_target==a)]
                x=o[mask(o,c)]
                prs=[float((~z.next_target_after_decay).mean()) for _,z in x.groupby("pair") if len(z)>=5]
                if len(x) and prs:robust.append([tf,a,b,"+".join(c),dn,len(x),float((~x.next_target_after_decay).mean()),len(prs),float(np.mean(prs)),float(min(prs))])
    pd.DataFrame(scores,columns=["timeframe","period","from_target","to_target","conditions","samples","exit_precision","unconditional_exit_precision","lift","wilson95_lower","target_capture_pips_mean","future_mfe_pips_mean","future_mae_pips_mean","bars_after_target_mean"]).to_csv("reports/momentum_decay_exit.csv",index=False,float_format="%.8f")
    pd.DataFrame(robust,columns=["timeframe","from_target","to_target","conditions","discovery_n","oos_n","oos_exit_precision","oos_pairs_n_ge5","oos_pair_precision_mean","oos_pair_precision_min"]).to_csv("reports/momentum_decay_exit_robustness.csv",index=False,float_format="%.8f")
    print("DONE",len(scores),len(robust),flush=True)

if __name__=="__main__":main()
