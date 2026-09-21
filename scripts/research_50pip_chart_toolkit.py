#!/usr/bin/env python3
"""Chart-toolkit feature study: SMA, Ichimoku, Stochastic, RCI, Fibonacci,
trendline/channel and objective shape/compression proxies.

All conditions are causal and evaluated against directional +50pip labels.
This is signal research, not a complete trading win-rate.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("direct","scripts/research_50pip_direct_entry.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

PAIRS=mod.PAIRS; TFS=mod.TFS; HORIZONS=mod.HORIZONS

def mask(g,names):
    m=pd.Series(True,index=g.index)
    for n in names:
        if n not in g.columns: return pd.Series(False,index=g.index)
        m &= g[n].fillna(False)
    return m

def main():
    Path("reports").mkdir(exist_ok=True)
    chunks=[]
    for tf in TFS:
        for pair in PAIRS:
            df=mod.load_market(tf,pair)
            f=mod.build_features(df); lab=mod.add_labels(df,pair,tf)
            z=pd.concat([df[["timestamp","open","high","low","close"]],f,lab],axis=1)
            z["pair"]=pair; z["timeframe"]=tf; chunks.append(z)
    all_df=pd.concat(chunks,ignore_index=True)

    rows=[]
    for tf in TFS:
        gtf=all_df[all_df.timeframe==tf]
        for direction,features in (("bull",mod.TOOLKIT_BULL),("bear",mod.TOOLKIT_BEAR)):
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}" if direction=="bull" else f"hit_short50_h{h}"
                for feature in features:
                    vals=[tf,direction,h,feature]
                    for g in (gtf[gtf.timestamp.dt.year<=2024],
                              gtf[gtf.timestamp.dt.year==2025],
                              gtf[gtf.timestamp.dt.year>=2026]):
                        m=g[feature].fillna(False)
                        a=g[m]; b=g[~m]
                        if len(a)<100 or len(b)<100:
                            vals += [len(a),len(b),np.nan,np.nan,np.nan,np.nan]
                        else:
                            ha=float(a[target].mean()); hb=float(b[target].mean())
                            vals += [len(a),len(b),ha,hb,ha-hb,ha/hb if hb else np.nan]
                    oos=gtf[gtf.timestamp.dt.year>=2026]
                    base=float(oos[target].mean())
                    vals += [float(gtf[gtf.timestamp.dt.year>=2026].loc[gtf[gtf.timestamp.dt.year>=2026][feature].fillna(False),target].mean())/base if base else np.nan]
                    rows.append(vals)
    cols=["timeframe","direction","horizon_bars","feature",
          "disc_true_n","disc_false_n","disc_true_hit","disc_false_hit","disc_diff","disc_true_false_ratio",
          "val_true_n","val_false_n","val_true_hit","val_false_hit","val_diff","val_true_false_ratio",
          "oos_true_n","oos_false_n","oos_true_hit","oos_false_hit","oos_diff","oos_true_false_ratio","oos_base_lift"]
    out=pd.DataFrame(rows,columns=cols).sort_values(["oos_base_lift","oos_diff","oos_true_n"],ascending=False)
    out.to_csv("reports/50pip_chart_toolkit_effects.csv",index=False,float_format="%.8f")

    # Early-entry confirmation: test toolkit feature + one existing trigger.
    inc=[]
    for tf in TFS:
        gtf=all_df[all_df.timeframe==tf]
        for direction,tools,base_features in (
            ("bull",mod.TOOLKIT_BULL,mod.BULL_BASE),
            ("bear",mod.TOOLKIT_BEAR,mod.BEAR_BASE),
        ):
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}" if direction=="bull" else f"hit_short50_h{h}"
                disc=gtf[gtf.timestamp.dt.year<=2024]; base=float(disc[target].mean())
                strong=[]
                for a in base_features:
                    m=disc[a].fillna(False); n=int(m.sum())
                    if n>=150:
                        rate=float(disc.loc[m,target].mean())
                        if rate/base>=1.10: strong.append((a,rate/base))
                strong=sorted(strong,key=lambda x:x[1],reverse=True)[:12]
                for a,_ in strong:
                    for b in tools:
                        m=disc[a].fillna(False)&disc[b].fillna(False)
                        n=int(m.sum())
                        if n<50: continue
                        rate=float(disc.loc[m,target].mean())
                        for period,g in (("disc",disc),("val",gtf[gtf.timestamp.dt.year==2025]),("oos",gtf[gtf.timestamp.dt.year>=2026])):
                            mm=g[a].fillna(False)&g[b].fillna(False); nn=int(mm.sum())
                            rr=float(g.loc[mm,target].mean()) if nn else np.nan
                            inc.append([tf,direction,h,a,b,period,nn,rr])
    incdf=pd.DataFrame(inc,columns=["timeframe","direction","horizon_bars","base_feature","toolkit_feature","period","samples","hit_rate"])
    incdf.to_csv("reports/50pip_chart_toolkit_incremental.csv",index=False,float_format="%.8f")

    print("chart_toolkit_effect_rows",len(out),"incremental_rows",len(incdf))
    print(out[["timeframe","direction","horizon_bars","feature","disc_true_hit","val_true_hit","oos_true_hit","oos_base_lift"]].head(50).to_string(index=False))

if __name__=="__main__":
    main()
