#!/usr/bin/env python3
"""Pin-independent move-ladder origin research.

Purpose:
- Test whether the move-ladder effect survives WITHOUT pin_bull.
- Use one common candidate set per timeframe/horizon, selected only in Discovery.
- Evaluate the same candidates across +50/+100/+150/+200/+300 targets.
- Discovery <=2024, validation 2025, OOS >=2026.
- Forward-only labels and completed MTF candles come from the shared dataset builder.
"""
from pathlib import Path
import importlib.util
from itertools import combinations, product
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("mtf","scripts/research_50pip_mtf_structure.py")
mtf=importlib.util.module_from_spec(spec); spec.loader.exec_module(mtf)

TFS=mtf.TFS
LADDER=[50,100,150,200,300]
HORIZONS={"h4":[1,3,6,12],"d1":[1,3,5,10]}

FAMILIES={
"trend":["trend_bull","price20_bull","price200_bull","ema20_rising","ema200_rising","sma_stack_bull","ichimoku_cloud_bull"],
"momentum":["rsi60","rsi_up1","rsi_up3","rsi_cross55_up","macd_bull","macd_rising","macd_cross_up","macd_accel_up","dmi_adx_bull","dmi_adx_strong_bull","adx25","adx_rising","adx_up3","di_bull","di_strong_bull","di_cross_up","di_spread_up3","rci_bull","rci_strong_bull"],
"volatility":["volatility","atr_expand","bb_expand","bb_bull","bb_upper","superbb_bull","superbb_expand","squeeze_release_up","range_expansion"],
"structure":["breakout20_up","breakout_up20","horizontal_break_up","channel_breakout_up","hh20","structure_bull_transition","triangle_breakout_up","range_compression","range_tight","inside_bar","inside_2","nr7","nr4"],
"price_action":["retest_up","engulf_bull","strong_close_bull","impulse_bull","close_near_high","pullback_bull"],
"wave":["elliott_wave3_bull","elliott_wave5_bull","elliott_impulse_bull","elliott_pullback_bull"],
"mtf":["mtf_trend_bull","mtf_price20_bull","mtf_price200_bull","mtf_ema20_rising","mtf_macd_bull","mtf_macd_rising","mtf_rsi60","mtf_adx25","mtf_adx_rising","mtf_di_bull","mtf_di_strong_bull","mtf_volatility","mtf_bb_bull","mtf_breakout_up"],
"cross":["cross_group_bull","cross_group_return_up","cross_target_relative_up"],
}

def as_bool(s):
    if pd.api.types.is_bool_dtype(s): return s.fillna(False)
    return pd.to_numeric(s,errors="coerce").fillna(0).ne(0)

def mask(g,combo):
    m=pd.Series(True,index=g.index)
    for n in combo:
        if n not in g.columns:return pd.Series(False,index=g.index)
        m &= as_bool(g[n])
    return m

def wilson(k,n):
    if n<=0:return np.nan
    z=1.959963984540054;p=k/n;den=1+z*z/n
    return ((p+z*z/(2*n))-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def common_candidates(data,horizon,top_k=3,min_n=120):
    disc=data[data.timestamp.dt.year<=2024].copy()
    # Common score uses +100/+150/+200, so candidate selection is NOT target-specific.
    scores=[]
    for fam,features in FAMILIES.items():
        fam_scores=[]
        for name in features:
            if name not in disc.columns: continue
            m=as_bool(disc[name]); n=int(m.sum())
            if n<min_n: continue
            lifts=[]
            for th in [100,150,200]:
                base=float(disc[f"hit{th}_h{horizon}"].mean())
                hit=float(disc.loc[m,f"hit{th}_h{horizon}"].mean())
                if base>0:lifts.append(hit/base)
            if lifts:fam_scores.append((name,float(np.mean(lifts)),n))
        scores.extend([(fam,*x) for x in sorted(fam_scores,key=lambda z:(z[1],z[2]),reverse=True)[:top_k]])
    selected={}
    for fam,name,score,n in scores:selected.setdefault(fam,[]).append(name)

    combos=[]
    for k in (2,3,4):
        for fams in combinations(selected.keys(),k):
            for picked in product(*(selected[f] for f in fams)):
                if len(set(picked))==k: combos.append(tuple(picked))
    explicit=[
      ("mtf_trend_bull","breakout20_up","atr_expand"),
      ("mtf_trend_bull","horizontal_break_up","atr_expand"),
      ("mtf_trend_bull","retest_up","atr_expand"),
      ("mtf_macd_bull","adx_up3","atr_expand"),
      ("mtf_adx_rising","breakout20_up","range_expansion"),
      ("mtf_price200_bull","macd_cross_up","atr_expand"),
      ("mtf_trend_bull","hh20","macd_accel_up","atr_expand"),
      ("mtf_trend_bull","pullback_bull","macd_accel_up","atr_expand"),
      ("mtf_volatility","breakout20_up","adx_up3"),
      ("mtf_trend_bull","retest_up","macd_accel_up","atr_expand"),
      ("mtf_trend_bull","elliott_wave3_bull","macd_accel_up","atr_expand"),
      ("mtf_trend_bull","range_compression","breakout20_up","atr_expand"),
      ("volatility","adx25","range_compression","breakout20_up"),
      ("volatility","adx_up3","range_compression","breakout20_up"),
      ("volatility","mtf_volatility","adx25"),
      ("volatility","mtf_volatility","breakout20_up"),
      ("atr_expand","mtf_volatility","breakout20_up"),
      ("atr_expand","mtf_volatility","adx25"),
    ]
    out=[];seen=set()
    for c in explicit+combos:
        if "pin_bull" in c: continue
        if c not in seen:seen.add(c);out.append(c)
    return out

def main():
    Path("reports").mkdir(exist_ok=True)
    rows=[]; robust=[]
    for tf in TFS:
        print("BUILD",tf,flush=True)
        data=mtf.build_dataset(tf)
        for h in HORIZONS[tf]:
            combos=common_candidates(data,h)
            cache={c:mask(data,c) for c in combos}
            for th in LADDER:
                target=f"hit{th}_h{h}"
                disc=data[data.timestamp.dt.year<=2024]
                val=data[data.timestamp.dt.year==2025]
                oos=data[data.timestamp.dt.year>=2026]
                base=float(disc[target].mean())
                vb=float(val[target].mean()); ob=float(oos[target].mean())
                ranked=[]
                for c,m in cache.items():
                    md=m & (data.timestamp.dt.year<=2024); n=int(md.sum())
                    if n<120: continue
                    dh=float(data.loc[md,target].mean())
                    dl=dh/base if base else np.nan
                    if np.isfinite(dl): ranked.append((c,n,dh,dl))
                # Keep a bounded common set, identical across all targets.
                ranked=sorted(ranked,key=lambda x:(x[3],x[1]),reverse=True)[:120]
                for c,dn,dh,dl in ranked:
                    m=cache[c]
                    mv=m & (data.timestamp.dt.year==2025); vn=int(mv.sum())
                    vh=float(data.loc[mv,target].mean()) if vn else np.nan
                    mo=m & (data.timestamp.dt.year>=2026); on=int(mo.sum())
                    oh=float(data.loc[mo,target].mean()) if on else np.nan
                    rows.append([tf,h,th,"+".join(c),dn,dh,dl,vn,vh,vh/vb if vn and vb else np.nan,
                                 wilson(int(data.loc[mv,target].sum()),vn),on,oh,oh/ob if on and ob else np.nan,
                                 wilson(int(data.loc[mo,target].sum()),on)])
                    ps=[float(x[target].mean()) for _,x in data.loc[mo].groupby("pair") if len(x)>=5]
                    if ps:
                        robust.append([tf,h,th,"+".join(c),dn,on,dl,oh/ob if on and ob else np.nan,
                                       len(ps),sum(x>=ob for x in ps),float(np.mean(ps)),float(min(ps))])
    cols=["timeframe","horizon_bars","target_pips","conditions","discovery_n","discovery_hit","discovery_lift",
          "validation_n","validation_hit","validation_lift","validation_wilson95","oos_n","oos_hit","oos_lift","oos_wilson95"]
    df=pd.DataFrame(rows,columns=cols)
    rob=pd.DataFrame(robust,columns=["timeframe","horizon_bars","target_pips","conditions","discovery_n","oos_n",
                                      "discovery_lift","oos_lift","oos_pairs_n_ge5","oos_pairs_above_base",
                                      "oos_pair_hit_mean","oos_min_pair_hit"])
    df.to_csv("reports/pin_independent_move_ladder_oos.csv",index=False,float_format="%.8f")
    rob.to_csv("reports/pin_independent_move_ladder_robustness.csv",index=False,float_format="%.8f")
    # Focus shortlist: OOS support + lower Wilson bound, across all targets.
    top=df[(df.oos_n>=30)].sort_values(["oos_wilson95","oos_lift","oos_n"],ascending=False).head(200)
    top.to_csv("reports/pin_independent_move_ladder_shortlist.csv",index=False,float_format="%.8f")
    print("DONE rows",len(df),"robust",len(rob),flush=True)
    print(top.head(80).to_string(index=False),flush=True)

if __name__=="__main__":main()
