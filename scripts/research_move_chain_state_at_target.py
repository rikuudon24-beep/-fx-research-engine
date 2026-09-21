#!/usr/bin/env python3
"""True state-at-target continuation research.

When +A is first reached, the candle that first reaches +A becomes the
decision/state candle. Features are read from that completed candle only.
Continuation to +B is measured from the +A price level over a fixed clean
post-target window (H4=6 bars, D1=5 bars).

Discovery/Validation/OOS are separated. Candidate conditions are selected
from Discovery state-at-target observations independently for each transition.
No pin-bar feature is used in candidate generation.
"""
from pathlib import Path
import importlib.util
from itertools import combinations, product
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("mtf","scripts/research_50pip_mtf_structure.py")
mtf=importlib.util.module_from_spec(spec); spec.loader.exec_module(mtf)

TFS=mtf.TFS
LADDER=[50,100,150,200]
TARGET_PAIRS=[(50,100),(100,150),(150,200),(200,300)]
WINDOWS={"h4":6,"d1":5}

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

def build_states(g, pair, horizon):
    g=g.sort_values("timestamp").reset_index(drop=True)
    pip=0.01 if pair.endswith("jpy") else 0.0001
    feature_cols=[c for c in g.columns if c not in {"timestamp","open","high","low","close","pair","timeframe"}]
    rows=[]
    for i in range(len(g)):
        e=float(g.close.iloc[i])
        for a,b in TARGET_PAIRS:
            a_level=e+a*pip; b_level=e+b*pip
            ha=None; hb=None
            for j in range(i+1,min(len(g),i+1+horizon)):
                if ha is None and float(g.high.iloc[j])>=a_level: ha=j
                if hb is None and float(g.high.iloc[j])>=b_level: hb=j
                if ha is not None and hb is not None: break
            if ha is None: continue
            end=min(len(g),ha+1+WINDOWS["h4" if horizon==12 else "d1"])
            rem_hit=None
            for j in range(ha+1,end):
                if float(g.high.iloc[j])>=b_level:
                    rem_hit=j; break
            highs=g.high.iloc[ha+1:end].astype(float)
            lows=g.low.iloc[ha+1:end].astype(float)
            rec={"signal_timestamp":g.timestamp.iloc[i],"target_timestamp":g.timestamp.iloc[ha],
                 "entry_close":e,"target_price":a_level,"from_target":a,"to_target":b,
                 "reached_next_in_original_horizon":bool(hb is not None and hb>ha),
                 "reached_next_in_remaining_window":bool(rem_hit is not None),
                 "bars_to_next_after_target":(rem_hit-ha) if rem_hit is not None else np.nan,
                 "post_target_mfe_pips":float((highs.max()-a_level)/pip) if len(highs) else np.nan,
                 "post_target_mae_pips":float((lows.min()-a_level)/pip) if len(lows) else np.nan,
                 "pair":pair,"target_index":ha}
            for c in feature_cols: rec[c]=g[c].iloc[ha]
            rows.append(rec)
    return pd.DataFrame(rows)

def candidate_pool(states, transition):
    a,b=transition
    disc=states[states.target_timestamp.dt.year<=2024]
    q=disc[(disc.from_target==a)]
    selected={}
    for fam,features in FAMILIES.items():
        fs=[]
        for n in features:
            if n not in q.columns: continue
            m=as_bool(q[n]); nobs=int(m.sum())
            if nobs<30: continue
            hit=float(q.loc[m,"reached_next_in_remaining_window"].mean())
            base=float(q["reached_next_in_remaining_window"].mean())
            if base>0: fs.append((n,hit/base,nobs))
        selected[fam]=[x[0] for x in sorted(fs,key=lambda z:(z[1],z[2]),reverse=True)[:3]]
    combos=[]
    for k in (2,3,4):
        for fams in combinations([f for f in selected if selected[f]],k):
            for picked in product(*(selected[f] for f in fams)):
                if len(set(picked))==k: combos.append(tuple(picked))
    seen=set(EXPLICIT); out=[c for c in EXPLICIT if all(x in q.columns for x in c)]
    for c in combos:
        if "pin_bull" in c: continue
        if c not in seen: seen.add(c); out.append(c)
    return out

def main():
    Path("reports").mkdir(exist_ok=True)
    all_scores=[]; robust=[]
    for tf in TFS:
        horizon=12 if tf=="h4" else 10
        print("BUILD",tf,flush=True)
        data=mtf.build_dataset(tf)
        states=[]
        for pair,g in data.groupby("pair",sort=False):
            s=build_states(g,pair,horizon)
            if len(s): states.append(s)
        states=pd.concat(states,ignore_index=True)
        for transition in TARGET_PAIRS:
            a,b=transition
            base_disc=states[(states.from_target==a)&(states.target_timestamp.dt.year<=2024)]
            combos=candidate_pool(states,transition)
            ranked=[]
            for c in combos:
                m=mask(base_disc,c); n=int(m.sum())
                if n<30: continue
                rate=float(base_disc.loc[m,"reached_next_in_remaining_window"].mean())
                ranked.append((c,n,rate))
            ranked=sorted(ranked,key=lambda x:(x[2],x[1]),reverse=True)[:80]
            for c,dn,dr in ranked:
                for period,name in [
                    (states[states.target_timestamp.dt.year<=2024],"discovery"),
                    (states[states.target_timestamp.dt.year==2025],"validation"),
                    (states[states.target_timestamp.dt.year>=2026],"oos")]:
                    p=period[period.from_target==a]
                    m=mask(p,c); n=int(m.sum())
                    rate=float(p.loc[m,"reached_next_in_remaining_window"].mean()) if n else np.nan
                    base=float(p["reached_next_in_remaining_window"].mean()) if len(p) else np.nan
                    k=int(p.loc[m,"reached_next_in_remaining_window"].sum()) if n else 0
                    all_scores.append([tf,name,a,b,"+".join(c),n,rate,base,rate/base if n and base else np.nan,
                                       wilson(k,n),float(p.loc[m,"post_target_mfe_pips"].mean()) if n else np.nan,
                                       float(p.loc[m,"post_target_mae_pips"].mean()) if n else np.nan])
                o=states[(states.target_timestamp.dt.year>=2026)&(states.from_target==a)]
                m=mask(o,c)
                x=o[m]
                pair_rates=[float(z["reached_next_in_remaining_window"].mean()) for _,z in x.groupby("pair") if len(z)>=5]
                if len(x) and pair_rates:
                    robust.append([tf,a,b,"+".join(c),dn,len(x),float(x["reached_next_in_remaining_window"].mean()),
                                   len(pair_rates),float(np.mean(pair_rates)),float(min(pair_rates))])
    pd.DataFrame(all_scores,columns=["timeframe","period","from_target","to_target","conditions","samples",
      "continuation_rate","unconditional_rate","lift","wilson95_lower","post_target_mfe_pips_mean","post_target_mae_pips_mean"]
    ).to_csv("reports/move_chain_state_at_target.csv",index=False,float_format="%.8f")
    pd.DataFrame(robust,columns=["timeframe","from_target","to_target","conditions","discovery_n","oos_n",
      "oos_continuation_rate","oos_pairs_n_ge5","oos_pair_rate_mean","oos_pair_rate_min"]
    ).to_csv("reports/move_chain_state_at_target_robustness.csv",index=False,float_format="%.8f")
    print("DONE",len(all_scores),len(robust),flush=True)

if __name__=="__main__": main()
