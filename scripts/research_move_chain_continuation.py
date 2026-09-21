#!/usr/bin/env python3
"""Conditional move-chain research.

Measures, for causal origin conditions, the sequence:
signal -> +50 -> +100 -> +150 -> +200 -> +300.
Unlike the ordinary ladder report, this measures continuation AFTER a prior
target has actually been reached. Discovery selects candidates; validation/OOS
are untouched until scoring.

No lookahead: the shared dataset supplies completed-candle features and
forward-only hit labels. First-hit timing is recomputed directly from future
OHLC after the signal candle.
"""
from pathlib import Path
import importlib.util
from itertools import combinations, product
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("mtf","scripts/research_50pip_mtf_structure.py")
mtf=importlib.util.module_from_spec(spec); spec.loader.exec_module(mtf)

TFS=mtf.TFS
HORIZONS={"h4":12,"d1":10}
LADDER=[50,100,150,200,300]

FAMILIES={
"trend":["trend_bull","price200_bull","ema200_rising","sma_stack_bull"],
"momentum":["adx25","adx_up3","dmi_adx_strong_bull","di_strong_bull","macd_accel_up"],
"volatility":["volatility","atr_expand","bb_expand","range_expansion"],
"structure":["breakout20_up","horizontal_break_up","hh20","structure_bull_transition","range_compression","nr4"],
"price_action":["retest_up","strong_close_bull","impulse_bull","close_near_high","pullback_bull"],
"mtf":["mtf_trend_bull","mtf_price200_bull","mtf_macd_bull","mtf_adx25","mtf_adx_rising","mtf_di_strong_bull","mtf_volatility"],
"cross":["cross_group_bull","cross_group_return_up","cross_target_relative_up"],
}

EXPLICIT=[
("sma_stack_bull","adx_up3","volatility"),
("adx_up3","volatility"),
("volatility","mtf_volatility","adx25"),
("atr_expand","mtf_volatility","adx25"),
("trend_bull","adx25","breakout20_up","mtf_volatility"),
("trend_bull","volatility","breakout20_up","mtf_volatility"),
("sma_stack_bull","volatility","mtf_volatility"),
("mtf_trend_bull","breakout20_up","atr_expand"),
("mtf_trend_bull","horizontal_break_up","atr_expand"),
("mtf_macd_bull","adx_up3","atr_expand"),
("mtf_adx_rising","breakout20_up","range_expansion"),
("mtf_trend_bull","retest_up","macd_accel_up","atr_expand"),
("price200_bull","volatility","elliott_wave5_bull","cross_group_return_up"),
]

def as_bool(s):
    if pd.api.types.is_bool_dtype(s): return s.fillna(False)
    return pd.to_numeric(s,errors="coerce").fillna(0).ne(0)

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

def first_hit_after(highs, start, target, pip):
    level=highs.iloc[start]*1.0 + target*pip
    # levels are based on signal close; target is measured from entry close.
    for j in range(start+1,len(highs)):
        if highs.iloc[j] >= level:return j
    return None

def enrich_chain(g, horizon, pair):
    # Event-level first-hit bars from each signal candle.
    g=g.sort_values("timestamp").reset_index(drop=True)
    pip=0.01 if pair.endswith("jpy") else 0.0001
    out=[]
    for i,row in g.iterrows():
        e=float(row.close)
        hits={}
        for target in LADDER:
            level=e+target*pip
            hit=None
            for j in range(i+1,min(len(g),i+1+horizon)):
                if float(g.high.iloc[j])>=level:
                    hit=j;break
            hits[target]=hit
        rec={"idx":i,"timestamp":row.timestamp,"entry_close":e}
        for t in LADDER:
            rec[f"hit{t}"]=hits[t]
            rec[f"bars{t}"]=(hits[t]-i) if hits[t] is not None else np.nan
        # conditional continuation: after target T is first reached, does T+50
        # get reached within the remaining horizon from the original signal?
        for a,b in zip(LADDER[:-1],LADDER[1:]):
            ha=hits[a]
            hb=hits[b]
            rec[f"cont_{a}_{b}"]=bool(ha is not None and hb is not None and hb>ha)
        out.append(rec)
    return pd.DataFrame(out)

def candidate_pool(data,horizon):
    disc=data[data.timestamp.dt.year<=2024]
    scores=[]
    for fam,features in FAMILIES.items():
        fs=[]
        for n in features:
            if n not in disc.columns:continue
            m=as_bool(disc[n]); nobs=int(m.sum())
            if nobs<120:continue
            # Rank on +50/+100/+150 reachability together.
            lifts=[]
            for t in [50,100,150]:
                base=float(disc[f"hit{t}_h{horizon}"].mean())
                hit=float(disc.loc[m,f"hit{t}_h{horizon}"].mean())
                if base>0:lifts.append(hit/base)
            if lifts:fs.append((n,float(np.mean(lifts)),nobs))
        for x in sorted(fs,key=lambda z:(z[1],z[2]),reverse=True)[:3]:scores.append((fam,*x))
    selected={}
    for fam,n,score,nobs in scores:selected.setdefault(fam,[]).append(n)
    combos=[]
    for k in (2,3,4):
        for fams in combinations(selected.keys(),k):
            for picked in product(*(selected[f] for f in fams)):
                if len(set(picked))==k:combos.append(tuple(picked))
    seen=set(EXPLICIT);out=list(EXPLICIT)
    for c in combos:
        if "pin_bull" in c:continue
        if c not in seen:seen.add(c);out.append(c)
    return out

def main():
    Path("reports").mkdir(exist_ok=True)
    chain_rows=[]; score_rows=[]
    for tf in TFS:
        h=HORIZONS[tf]
        print("BUILD",tf,flush=True)
        data=mtf.build_dataset(tf)
        # Build event-level first-hit data pair by pair to preserve pip size.
        events=[]
        for pair,g in data.groupby("pair",sort=False):
            # Only need feature columns + OHLC. Recompute hits from raw dataset.
            ev=enrich_chain(g[["timestamp","open","high","low","close"]].copy(),h,pair)
            ev["pair"]=pair
            # Restore feature rows by index alignment.
            gg=g.reset_index(drop=True)
            for c in [x for x in g.columns if x not in {"timestamp","open","high","low","close","pair","timeframe"}]:
                ev[c]=gg[c].values
            events.append(ev)
        ev=pd.concat(events,ignore_index=True)
        disc=ev[ev.timestamp.dt.year<=2024]
        val=ev[ev.timestamp.dt.year==2025]
        oos=ev[ev.timestamp.dt.year>=2026]
        combos=candidate_pool(data,h)
        cache={c:mask(ev,c) for c in combos}
        # Candidate selection uses discovery continuation only.
        ranked=[]
        for c,m in cache.items():
            md=m & (ev.timestamp.dt.year<=2024)
            n=int(md.sum())
            if n<120:continue
            rates=[]
            for a,b in zip(LADDER[:-1],LADDER[1:]):
                q=md & ev[f"hit{a}"].notna()
                den=int(q.sum())
                if den>=40:
                    rates.append(float(ev.loc[q,f"cont_{a}_{b}"].mean()))
            if rates:ranked.append((c,n,float(np.mean(rates))))
        ranked=sorted(ranked,key=lambda x:(x[2],x[1]),reverse=True)[:100]
        for c,dn,dc in ranked:
            m=cache[c]
            for period,name,pm in [(disc,"discovery",m & (ev.timestamp.dt.year<=2024)),
                                   (val,"validation",m & (ev.timestamp.dt.year==2025)),
                                   (oos,"oos",m & (ev.timestamp.dt.year>=2026))]:
                for a,b in zip(LADDER[:-1],LADDER[1:]):
                    q=pm & ev[f"hit{a}"].notna()
                    n=int(q.sum())
                    rate=float(ev.loc[q,f"cont_{a}_{b}"].mean()) if n else np.nan
                    base=float(period[f"hit{b}"].mean()) if len(period) else np.nan
                    score_rows.append([tf,name,a,b,"+".join(c),n,rate,base,rate/base if n and base else np.nan,
                                       wilson(int(ev.loc[q,f"cont_{a}_{b}"].sum()),n)])
            # Pair robustness on OOS for each transition.
            for a,b in zip(LADDER[:-1],LADDER[1:]):
                q=m & (ev.timestamp.dt.year>=2026) & ev[f"hit{a}"].notna()
                pairs=[]
                for pair,x in ev.loc[q].groupby("pair"):
                    if len(x)>=5:pairs.append(float(x[f"cont_{a}_{b}"].mean()))
                if pairs:
                    chain_rows.append([tf,a,b,"+".join(c),dn,int(q.sum()),float(ev.loc[q,f"cont_{a}_{b}"].mean()),
                                       len(pairs),float(np.mean(pairs)),float(min(pairs))])
    cols=["timeframe","period","from_target","to_target","conditions","samples_after_from",
          "continuation_rate","unconditional_to_rate","continuation_lift","wilson95_lower"]
    pd.DataFrame(score_rows,columns=cols).to_csv("reports/move_chain_continuation.csv",index=False,float_format="%.8f")
    pd.DataFrame(chain_rows,columns=["timeframe","from_target","to_target","conditions","discovery_n","oos_n",
                                     "oos_continuation_rate","oos_pairs_n_ge5","oos_pair_continuation_mean","oos_pair_continuation_min"]
                ).to_csv("reports/move_chain_continuation_robustness.csv",index=False,float_format="%.8f")
    print("DONE chain rows",len(score_rows),"robust",len(chain_rows),flush=True)

if __name__=="__main__":main()
