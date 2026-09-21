#!/usr/bin/env python3
"""Feature/transition effect study for direct +50 entry research.

Measures which current-state or transition features distinguish +50-hit candles
from non-hit candles, with discovery/validation/OOS splits. This is evidence
for signal construction, not a complete trading win-rate.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

spec=importlib.util.spec_from_file_location("direct","scripts/research_50pip_direct_entry.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

PAIRS=mod.PAIRS; TFS=mod.TFS; HORIZONS=mod.HORIZONS
BULL_BASE=mod.BULL_BASE; BEAR_BASE=mod.BEAR_BASE

def effect(g, feature, target):
    x=g[g[feature].fillna(False)]
    y=g[~g[feature].fillna(False)]
    if len(x)<100 or len(y)<100: return None
    hx=float(x[target].mean()); hy=float(y[target].mean())
    return len(x),len(y),hx,hy,hx-hy,hx/hy if hy else np.nan

def main():
    chunks=[]
    for tf in TFS:
        for pair in PAIRS:
            df=mod.load_market(tf,pair)
            f=mod.build_features(df); lab=mod.add_labels(df,pair,tf)
            z=pd.concat([df[["timestamp"]],f,lab],axis=1)
            z["pair"]=pair; z["timeframe"]=tf; chunks.append(z)
    all_df=pd.concat(chunks,ignore_index=True)
    rows=[]
    for tf in TFS:
        tfdf=all_df[all_df.timeframe==tf]
        for direction in ("bull","bear"):
            features=BULL_BASE if direction=="bull" else BEAR_BASE
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}" if direction=="bull" else f"hit_short50_h{h}"
                for feature in features:
                    vals=[tf,direction,h,feature]
                    for g in (tfdf[tfdf.timestamp.dt.year<=2024],
                              tfdf[tfdf.timestamp.dt.year==2025],
                              tfdf[tfdf.timestamp.dt.year>=2026]):
                        e=effect(g,feature,target)
                        if e is None: vals += [np.nan]*7
                        else:
                            ntrue,nfalse,hrtrue,hrfalse,diff,ratio=e
                            base=float(g[target].mean())
                            vals += [ntrue,nfalse,hrtrue,hrfalse,diff,ratio,ratio if base else np.nan]
                    rows.append(vals)
    cols=["timeframe","direction","horizon_bars","feature",
          "disc_true_n","disc_false_n","disc_true_hit","disc_false_hit","disc_diff","disc_true_false_ratio","disc_ratio",
          "val_true_n","val_false_n","val_true_hit","val_false_hit","val_diff","val_true_false_ratio","val_ratio",
          "oos_true_n","oos_false_n","oos_true_hit","oos_false_hit","oos_diff","oos_true_false_ratio","oos_ratio"]
    out=pd.DataFrame(rows,columns=cols)
    # The ratio columns above are intentionally feature-state ratios; keep a
    # separate OOS lift against the full OOS base for easy screening.
    out["oos_base_lift"]=np.nan
    for i,r in out.iterrows():
        g=all_df[(all_df.timeframe==r.timeframe)&(all_df.timestamp.dt.year>=2026)]
        base=float(g[f"hit50_h{r.horizon_bars}" if r.direction=="bull" else f"hit_short50_h{r.horizon_bars}"].mean())
        if np.isfinite(r.oos_true_hit): out.at[i,"oos_base_lift"]=r.oos_true_hit/base if base else np.nan
    out=out.sort_values(["oos_base_lift","oos_diff","oos_true_n"],ascending=False)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/50pip_entry_feature_effects.csv",index=False,float_format="%.8f")

    # Near-miss diagnostic: for each strong discovery feature, find pairs where
    # adding a second feature changes the hit rate materially in validation/OOS.
    inc=[]
    for tf in TFS:
        tfdf=all_df[all_df.timeframe==tf]
        for direction in ("bull","bear"):
            features=BULL_BASE if direction=="bull" else BEAR_BASE
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}"
                disc=tfdf[tfdf.timestamp.dt.year<=2024]
                candidates=[]
                for a in features:
                    sa=disc[disc[a].fillna(False)]
                    if len(sa)<150: continue
                    ha=float(sa[target].mean()); base=float(disc[target].mean())
                    if ha/base>=1.10: candidates.append((a,ha/base))
                candidates=sorted(candidates,key=lambda z:z[1],reverse=True)[:20]
                for a,_ in candidates:
                    for b in features:
                        if b==a: continue
                        vals=[tf,direction,h,a,b]
                        for g in (disc,tfdf[tfdf.timestamp.dt.year==2025],tfdf[tfdf.timestamp.dt.year>=2026]):
                            sa=g[g[a].fillna(False)]
                            sb=sa[g.loc[sa.index,b].fillna(False)]
                            if len(sb)<50:
                                vals += [len(sb),np.nan,np.nan]
                            else:
                                rate=float(sb[target].mean()); arate=float(sa[target].mean())
                                vals += [len(sb),rate,rate-arate]
                        inc.append(vals)
    incdf=pd.DataFrame(inc,columns=["timeframe","direction","horizon_bars","base_feature","added_feature",
                                    "disc_n","disc_rate","disc_increment","val_n","val_rate","val_increment",
                                    "oos_n","oos_rate","oos_increment"])
    incdf.to_csv("reports/50pip_entry_incremental_effects.csv",index=False,float_format="%.8f")
    print("feature_effect_rows",len(out),"incremental_rows",len(incdf))
    print(out[["timeframe","direction","horizon_bars","feature","disc_true_hit","val_true_hit","oos_true_hit","oos_base_lift"]].head(40).to_string(index=False))

if __name__=="__main__": main()
