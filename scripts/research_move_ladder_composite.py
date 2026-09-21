#!/usr/bin/env python3
"""FX move ladder research: +50/+100/+150/+200/+300 pip reach from early-origin composites.

Discovery <=2024, validation=2025, OOS>=2026. Candidate selection uses discovery only.
Labels are forward-only and use completed signal candles; MTF features come from
completed higher-timeframe candles in the shared dataset builder.
"""
from pathlib import Path
import importlib.util
from itertools import combinations, product
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("mtf","scripts/research_50pip_mtf_structure.py")
mtf=importlib.util.module_from_spec(spec); spec.loader.exec_module(mtf)
mod=mtf.mod
PAIRS=mod.PAIRS
TFS=mod.TFS
HORIZONS={"h4":[1,3,6,12],"d1":[1,3,5,10]}
LADDER=[50,100,150,200,300]

FAMILIES={
"trend":["trend_bull","price20_bull","price200_bull","ema20_rising","ema200_rising","sma_stack_bull","ichimoku_cloud_bull"],
"momentum":["rsi60","rsi_up1","rsi_up3","rsi_cross55_up","macd_bull","macd_rising","macd_cross_up","macd_accel_up","dmi_adx_bull","dmi_adx_strong_bull","adx25","adx_rising","adx_up3","di_bull","di_strong_bull","di_cross_up","di_spread_up3","rci_bull","rci_strong_bull"],
"volatility":["volatility","atr_expand","bb_expand","bb_bull","bb_upper","superbb_bull","superbb_expand","squeeze_release_up","range_expansion"],
"structure":["breakout20_up","breakout_up20","horizontal_break_up","channel_breakout_up","hh20","structure_bull_transition","triangle_breakout_up","range_compression","range_tight","inside_bar","inside_2","nr7","nr4"],
"price_action":["retest_up","pin_bull","engulf_bull","strong_close_bull","impulse_bull","close_near_high","pullback_bull"],
"wave":["elliott_wave3_bull","elliott_wave5_bull","elliott_impulse_bull","elliott_pullback_bull"],
"mtf":["mtf_trend_bull","mtf_price20_bull","mtf_price200_bull","mtf_ema20_rising","mtf_macd_bull","mtf_macd_rising","mtf_rsi60","mtf_adx25","mtf_adx_rising","mtf_di_bull","mtf_di_strong_bull","mtf_volatility","mtf_bb_bull","mtf_breakout_up"],
"cross":["cross_group_bull","cross_group_return_up","cross_target_relative_up"],
}

def mask(g,combo):
    m=pd.Series(True,index=g.index)
    for n in combo:
        if n not in g.columns:return pd.Series(False,index=g.index)
        s=g[n]
        if not pd.api.types.is_bool_dtype(s):
            s=pd.to_numeric(s,errors="coerce").fillna(0).ne(0)
        else:s=s.fillna(False)
        m &= s
    return m

def candidate_pool(data,target,top_k=3,min_n=120):
    disc=data[data.timestamp.dt.year<=2024]
    base=float(disc[target].mean())
    selected={}
    for fam,features in FAMILIES.items():
        scores=[]
        for name in features:
            if name not in data.columns:continue
            s=data[name]
            s=s.fillna(False) if pd.api.types.is_bool_dtype(s) else pd.to_numeric(s,errors="coerce").fillna(0).ne(0)
            m=s & (data.timestamp.dt.year<=2024)
            n=int(m.sum())
            if n<min_n:continue
            hit=float(data.loc[m,target].mean())
            lift=hit/base if base else np.nan
            if np.isfinite(lift):scores.append((name,n,hit,lift))
        selected[fam]=[x[0] for x in sorted(scores,key=lambda x:(x[3],x[1]),reverse=True)[:top_k]]
    combos=[]
    for k in (2,3,4):
        for fams in combinations(selected.keys(),k):
            for picked in product(*(selected[f] for f in fams)):
                if len(set(picked))==k:combos.append(tuple(picked))
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
      ("mtf_trend_bull","pin_bull","adx_rising","atr_expand"),
      ("mtf_trend_bull","elliott_wave3_bull","macd_accel_up","atr_expand"),
      ("mtf_trend_bull","range_compression","breakout20_up","atr_expand"),
    ]
    seen=set();out=[]
    for c in explicit+combos:
        if c not in seen:seen.add(c);out.append(c)
    return out

def wilson(k,n):
    if n<=0:return np.nan
    z=1.959963984540054;p=k/n;den=1+z*z/n
    return ((p+z*z/(2*n))-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def main():
    Path("reports").mkdir(exist_ok=True)
    rows=[]; robust=[]
    for tf in TFS:
        print("BUILD",tf,flush=True)
        data=mtf.build_dataset(tf)
        disc=data[data.timestamp.dt.year<=2024]
        val=data[data.timestamp.dt.year==2025]
        oos=data[data.timestamp.dt.year>=2026]
        for th in LADDER:
            for h in HORIZONS[tf]:
                target=f"hit{th}_h{h}"
                base=float(disc[target].mean())
                combos=candidate_pool(data,target)
                cache={c:mask(data,c) for c in combos}
                cand=[]
                for c,m in cache.items():
                    md=m & (data.timestamp.dt.year<=2024)
                    n=int(md.sum())
                    if n<120:continue
                    hit=float(data.loc[md,target].mean())
                    lift=hit/base if base else np.nan
                    if np.isfinite(lift) and lift>=1.08:
                        cand.append((c,n,hit,lift))
                cand=sorted(cand,key=lambda x:(x[3],x[1]),reverse=True)[:100]
                print("TARGET",tf,th,h,"BASE",round(base,6),"CAND",len(cand),flush=True)
                for c,dn,dh,dl in cand:
                    m=cache[c]
                    mg=m & (data.timestamp.dt.year==2025); nn=int(mg.sum())
                    vh=float(data.loc[mg,target].mean()) if nn else np.nan
                    vb=float(val[target].mean())
                    og=m & (data.timestamp.dt.year>=2026); on=int(og.sum())
                    oh=float(data.loc[og,target].mean()) if on else np.nan
                    ob=float(oos[target].mean())
                    rows.append([tf,th,h,"+".join(c),dn,dh,dl,nn,vh,vh/vb if vb else np.nan,
                                 wilson(int(data.loc[mg,target].sum()),nn),on,oh,oh/ob if ob else np.nan,
                                 wilson(int(data.loc[og,target].sum()),on)])
                    ps=[]
                    for pair,x in data.loc[og].groupby("pair"):
                        if len(x)>=5:ps.append(float(x[target].mean()))
                    if ps:
                        robust.append([tf,th,h,"+".join(c),dn,on,dl,oh/ob if ob else np.nan,
                                       len(ps),sum(x>=ob for x in ps),float(np.mean(ps)),float(min(ps))])
    cols=["timeframe","target_pips","horizon_bars","conditions","discovery_n","discovery_hit","discovery_lift",
          "validation_n","validation_hit","validation_lift","validation_wilson95",
          "oos_n","oos_hit","oos_lift","oos_wilson95"]
    df=pd.DataFrame(rows,columns=cols)
    rob=pd.DataFrame(robust,columns=["timeframe","target_pips","horizon_bars","conditions","discovery_n","oos_n",
                                      "discovery_lift","oos_lift","oos_pairs_n_ge5","oos_pairs_above_base",
                                      "oos_pair_hit_mean","oos_min_pair_hit"])
    df.to_csv("reports/move_ladder_composite_oos.csv",index=False,float_format="%.8f")
    rob.to_csv("reports/move_ladder_composite_robustness.csv",index=False,float_format="%.8f")
    top=df.sort_values(["oos_wilson95","oos_lift","oos_n"],ascending=False).head(150)
    top.to_csv("reports/move_ladder_composite_shortlist.csv",index=False,float_format="%.8f")
    print("DONE rows",len(df),"robust",len(rob),flush=True)
    print(top.head(60).to_string(index=False),flush=True)

if __name__=="__main__":main()
