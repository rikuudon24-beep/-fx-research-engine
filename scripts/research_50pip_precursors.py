#!/usr/bin/env python3
"""Precursor/sequence research for directional +50pip moves.

The question is not merely which indicator is high at entry, but what tends
to change in the 1-6 completed candles BEFORE a +50pip move begins. It also
measures how often the same precursor appears without a subsequent +50pip
move, preventing "winner-only" hindsight.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("direct","scripts/research_50pip_direct_entry.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

PAIRS=mod.PAIRS; TFS=mod.TFS; HORIZONS=mod.HORIZONS
BASE=(mod.BULL_BASE,mod.BEAR_BASE)
TOOL=(mod.TOOLKIT_BULL,mod.TOOLKIT_BEAR)
LAGS=[0,1,2,3,4,5,6]

def state_rate(g, feature, lag, target):
    s=g[feature].fillna(False).shift(lag)
    ok=s.notna()
    if int(ok.sum())==0: return np.nan,np.nan
    hit=float(g.loc[ok,target].mean())
    present=s[ok]
    return float(g.loc[present.index,target].mean()), float(present.mean())

def transition_rate(g, feature, lag, target):
    # feature becomes true at this lagged candle, a causal "change happened".
    s=g[feature].fillna(False).astype(int).shift(lag)
    prev=g[feature].fillna(False).astype(int).shift(lag+1)
    tr=(s==1)&(prev==0)
    n=int(tr.sum())
    return (n,float(g.loc[tr,target].mean())) if n else (0,np.nan)

def main():
    Path("reports").mkdir(exist_ok=True)
    chunks=[]
    for tf in TFS:
        for pair in PAIRS:
            df=mod.load_market(tf,pair)
            f=mod.build_features(df); lab=mod.add_labels(df,pair,tf)
            z=pd.concat([df[["timestamp","open","high","low","close"]],f,lab],axis=1)
            z["pair"]=pair; z["timeframe"]=tf
            chunks.append(z)
    all_df=pd.concat(chunks,ignore_index=True)

    rows=[]
    for tf in TFS:
        gtf=all_df[all_df.timeframe==tf]
        for direction,features in (("bull",list(dict.fromkeys(BASE[0]+TOOL[0]))),
                                    ("bear",list(dict.fromkeys(BASE[1]+TOOL[1])))):
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}" if direction=="bull" else f"hit_short50_h{h}"
                for feature in features:
                    for lag in LAGS:
                        vals=[tf,direction,h,feature,lag]
                        for period,g in (("disc",gtf[gtf.timestamp.dt.year<=2024]),
                                         ("val",gtf[gtf.timestamp.dt.year==2025]),
                                         ("oos",gtf[gtf.timestamp.dt.year>=2026])):
                            s=g[feature].fillna(False).shift(lag)
                            present=s==True
                            n=int(present.sum())
                            rate=float(g.loc[present,target].mean()) if n else np.nan
                            base=float(g[target].mean()) if len(g) else np.nan
                            vals += [n,rate,rate/base if base else np.nan]
                        rows.append(vals)
    out=pd.DataFrame(rows,columns=[
        "timeframe","direction","horizon_bars","feature","lead_lag",
        "disc_n","disc_hit","disc_lift","val_n","val_hit","val_lift",
        "oos_n","oos_hit","oos_lift"])
    out.to_csv("reports/50pip_precursor_lags.csv",index=False,float_format="%.8f")

    # Transition-only study: when the condition first turns on, how often is
    # the next move successful? This directly targets early entry.
    tr=[]
    for tf in TFS:
        gtf=all_df[all_df.timeframe==tf]
        for direction,features in (("bull",list(dict.fromkeys(BASE[0]+TOOL[0]))),
                                    ("bear",list(dict.fromkeys(BASE[1]+TOOL[1])))):
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}" if direction=="bull" else f"hit_short50_h{h}"
                for feature in features:
                    for lag in [0,1,2,3]:
                        vals=[tf,direction,h,feature,lag]
                        for g in (gtf[gtf.timestamp.dt.year<=2024],
                                  gtf[gtf.timestamp.dt.year==2025],
                                  gtf[gtf.timestamp.dt.year>=2026]):
                            n,rate=transition_rate(g,feature,lag,target)
                            vals += [n,rate]
                        tr.append(vals)
    trdf=pd.DataFrame(tr,columns=[
        "timeframe","direction","horizon_bars","feature","lead_lag",
        "disc_n","disc_hit","val_n","val_hit","oos_n","oos_hit"])
    trdf.to_csv("reports/50pip_precursor_transitions.csv",index=False,float_format="%.8f")

    # Sequential confirmation: discover common two-step sequences where A turns
    # on before B, then the target move follows. The event is evaluated only
    # from completed candles, so no future information leaks into the signal.
    seq=[]
    for tf in TFS:
        gtf=all_df[all_df.timeframe==tf]
        for direction,features in (("bull",list(dict.fromkeys(BASE[0]+TOOL[0]))),
                                    ("bear",list(dict.fromkeys(BASE[1]+TOOL[1])))):
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}" if direction=="bull" else f"hit_short50_h{h}"
                disc=gtf[gtf.timestamp.dt.year<=2024]
                # Pre-screen individual transitions by discovery count/rate.
                cand=[]
                base=float(disc[target].mean())
                for a in features:
                    s=disc[a].fillna(False)
                    trn=s & ~s.shift(1).fillna(False)
                    n=int(trn.sum())
                    if n>=100:
                        r=float(disc.loc[trn,target].mean())
                        if r/base>=1.10: cand.append((a,r/base))
                cand=sorted(cand,key=lambda x:x[1],reverse=True)[:20]
                for a,_ in cand:
                    for b in features:
                        if b==a: continue
                        for gap in [1,2,3]:
                            vals=[tf,direction,h,a,b,gap]
                            for g in (disc,gtf[gtf.timestamp.dt.year==2025],gtf[gtf.timestamp.dt.year>=2026]):
                                aa=g[a].fillna(False)&~g[a].fillna(False).shift(1).fillna(False)
                                bb=g[b].fillna(False).shift(-gap)
                                m=aa & bb
                                n=int(m.sum())
                                rate=float(g.loc[m,target].mean()) if n else np.nan
                                vals += [n,rate]
                            seq.append(vals)
    seqdf=pd.DataFrame(seq,columns=[
        "timeframe","direction","horizon_bars","first_transition","second_state","gap_bars",
        "disc_n","disc_hit","val_n","val_hit","oos_n","oos_hit"])
    seqdf.to_csv("reports/50pip_precursor_sequences.csv",index=False,float_format="%.8f")

    print("precursor_lag_rows",len(out),"transition_rows",len(trdf),"sequence_rows",len(seqdf))
    print(out.sort_values(["oos_lift","oos_n"],ascending=False).head(40).to_string(index=False))

if __name__=="__main__": main()
