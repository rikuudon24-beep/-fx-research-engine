#!/usr/bin/env python3
"""Realized P/L comparison of hold-vs-exit strategies after +A.

Uses fixed exit candidates identified in the independent exit study, but evaluates
them without re-selecting on OOS. Exit occurs at the condition candle close.
Baseline hold exits at +B if reached, otherwise at the final close of the
remaining window. Also reports missed upside and drawdown from +A.
"""
from pathlib import Path
import importlib.util, pandas as pd, numpy as np
spec=importlib.util.spec_from_file_location("mtf","scripts/research_50pip_mtf_structure.py")
mtf=importlib.util.module_from_spec(spec); spec.loader.exec_module(mtf)
TFS=mtf.TFS
TRANSITIONS=[(50,100),(100,150),(150,200),(200,300)]
WINDOWS={"h4":6,"d1":5}
CANDS=[
("price20_cross_down+di_spread_down3",("price20_cross_down","di_spread_down3")),
("price20_cross_down",("price20_cross_down",)),
("adx_down3+di_spread_down3",("adx_down3","di_spread_down3")),
("adx_down3",("adx_down3",)),
]
def ab(s):
    if pd.api.types.is_bool_dtype(s): return s.fillna(False).astype(bool)
    return pd.to_numeric(s,errors="coerce").fillna(0).ne(0)
def cond(g,j,c):
    return all(n in g.columns and bool(ab(g[n]).iloc[j]) for n in c)
def main():
    Path("reports").mkdir(exist_ok=True); rows=[]
    for tf in TFS:
        ds=mtf.build_dataset(tf)
        for pair,g in ds.groupby("pair",sort=False):
            g=g.sort_values("timestamp").reset_index(drop=True)
            pip=.01 if pair.endswith("jpy") else .0001
            horizon=12 if "h4" in str(g.timeframe.iloc[0]) else 10
            win=WINDOWS["h4" if "h4" in str(g.timeframe.iloc[0]) else "d1"]
            for i in range(len(g)):
                entry=float(g.close.iloc[i])
                for a,b in TRANSITIONS:
                    al=entry+a*pip; bl=entry+b*pip
                    target=None
                    for j in range(i+1,min(len(g),i+1+horizon)):
                        if float(g.high.iloc[j])>=al: target=j; break
                    if target is None or float(g.high.iloc[target])>=bl: continue
                    end=min(len(g),target+1+win)
                    if end<=target+1: continue
                    fut=g.iloc[target+1:end]
                    # Baseline hold: hit +B at the first qualifying candle, otherwise close at window end.
                    bhit=np.where(fut.high.astype(float).values>=bl)[0]
                    if len(bhit):
                        hj=target+1+int(bhit[0]); hold_px=bl; hold_hit=True
                    else:
                        hj=end-1; hold_px=float(g.close.iloc[hj]); hold_hit=False
                    hold_pips=(hold_px-al)/pip
                    hold_mae=float((fut.low.astype(float).min()-al)/pip)
                    hold_mfe=float((fut.high.astype(float).max()-al)/pip)
                    for name,c in CANDS:
                        hits=[j for j in range(target+1,end) if cond(g,j,c)]
                        if not hits:
                            continue
                        ej=hits[0]; exit_px=float(g.close.iloc[ej])
                        exit_pips=(exit_px-al)/pip
                        after=g.iloc[ej+1:end]
                        post_mae=float((after.low.astype(float).min()-exit_px)/pip) if len(after) else 0.0
                        post_mfe=float((after.high.astype(float).max()-exit_px)/pip) if len(after) else 0.0
                        # Counterfactual value: difference vs simply holding to the baseline exit.
                        opportunity=hold_pips-exit_pips
                        rows.append([tf,pair,a,b,name,g.timestamp.iloc[target],g.timestamp.iloc[ej],
                            ej-target,exit_pips,hold_pips,opportunity,hold_hit,hold_mae,hold_mfe,
                            post_mae,post_mfe, g.timestamp.iloc[i]])
    df=pd.DataFrame(rows,columns=["timeframe","pair","from_target","to_target","condition",
        "target_timestamp","exit_timestamp","bars_after_target","exit_pips","hold_pips",
        "opportunity_cost_pips","hold_hit_target","hold_mae_pips","hold_mfe_pips",
        "post_exit_mae_pips","post_exit_mfe_pips","signal_timestamp"])
    out=[]
    for tf in TFS:
        for a,b in TRANSITIONS:
            for c in [x[0] for x in CANDS]:
                x=df[(df.timeframe==tf)&(df.from_target==a)&(df.to_target==b)&(df.condition==c)].copy()
                for period,name in [(x[x.exit_timestamp.dt.year<=2024],"discovery"),
                                    (x[x.exit_timestamp.dt.year==2025],"validation"),
                                    (x[x.exit_timestamp.dt.year>=2026],"oos")]:
                    n=len(period)
                    if not n: continue
                    out.append([tf,name,a,b,c,n,float(period.exit_pips.mean()),float(period.hold_pips.mean()),
                        float(period.opportunity_cost_pips.mean()),float((period.exit_pips>0).mean()),
                        float(period.hold_pips.mean()-period.exit_pips.mean()),
                        float(period.bars_after_target.mean()),float(period.post_exit_mae_pips.mean()),
                        float(period.post_exit_mfe_pips.mean())])
    pd.DataFrame(out,columns=["timeframe","period","from_target","to_target","condition","samples",
        "exit_pips_mean","hold_pips_mean","opportunity_cost_pips_mean","exit_positive_rate",
        "hold_minus_exit_mean","bars_after_target_mean","post_exit_mae_mean","post_exit_mfe_mean"]).to_csv(
        "reports/exit_pnl_comparison.csv",index=False,float_format="%.8f")
    df.to_csv("reports/exit_pnl_events.csv",index=False,float_format="%.8f")
    print("DONE",len(df),len(out),flush=True)
if __name__=="__main__": main()
