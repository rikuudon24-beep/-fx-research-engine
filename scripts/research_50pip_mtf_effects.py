#!/usr/bin/env python3
"""MTF/structure feature-effect and near-miss study.

Compares feature-present vs feature-absent +50/-50 reachability and tests
incremental confirmation features. This is evidence for signal construction,
not a complete trading win-rate.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

spec=importlib.util.spec_from_file_location("mtf","scripts/research_50pip_mtf_structure.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

PAIRS=mod.PAIRS; TFS=mod.TFS; HORIZONS=mod.HORIZONS

def effect(g, feature, target):
    if feature not in g.columns: return None
    x=g[g[feature].fillna(False)]
    y=g[~g[feature].fillna(False)]
    if len(x)<100 or len(y)<100: return None
    hx=float(x[target].mean()); hy=float(y[target].mean())
    return len(x),len(y),hx,hy,hx-hy,hx/hy if hy else np.nan

def feature_pool(direction):
    local=mod.BULL_LOCAL if direction=="bull" else mod.BEAR_LOCAL
    mtf=mod.MTF_BULL if direction=="bull" else mod.MTF_BEAR
    struct=mod.STRUCT_BULL if direction=="bull" else mod.STRUCT_BEAR
    cross=mod.CROSS_BULL if direction=="bull" else mod.CROSS_BEAR
    toolkit=mod.TOOLKIT_BULL if direction=="bull" else mod.TOOLKIT_BEAR
    # Local chart-toolkit features plus completed higher-timeframe equivalents.
    mtf_toolkit=[f"mtf_{x}" for x in toolkit]
    return list(dict.fromkeys(local+toolkit+mtf+mtf_toolkit+struct+cross))

def main():
    Path("reports").mkdir(exist_ok=True)
    print("START MTF/structure feature effects")
    datasets={tf:mod.build_dataset(tf) for tf in TFS}

    rows=[]
    for tf,all_df in datasets.items():
        for direction in ("bull","bear"):
            features=feature_pool(direction)
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}" if direction=="bull" else f"hit_short50_h{h}"
                for feature in features:
                    vals=[tf,direction,h,feature]
                    for g in (
                        all_df[all_df.timestamp.dt.year<=2024],
                        all_df[all_df.timestamp.dt.year==2025],
                        all_df[all_df.timestamp.dt.year>=2026],
                    ):
                        e=effect(g,feature,target)
                        if e is None:
                            vals += [np.nan]*7
                        else:
                            ntrue,nfalse,hrtrue,hrfalse,diff,ratio=e
                            base=float(g[target].mean())
                            vals += [ntrue,nfalse,hrtrue,hrfalse,diff,ratio,
                                     hrtrue/base if base else np.nan]
                    rows.append(vals)
    cols=["timeframe","direction","horizon_bars","feature",
          "disc_true_n","disc_false_n","disc_true_hit","disc_false_hit","disc_diff","disc_true_false_ratio","disc_base_lift",
          "val_true_n","val_false_n","val_true_hit","val_false_hit","val_diff","val_true_false_ratio","val_base_lift",
          "oos_true_n","oos_false_n","oos_true_hit","oos_false_hit","oos_diff","oos_true_false_ratio","oos_base_lift"]
    out=pd.DataFrame(rows,columns=cols)
    out=out.sort_values(["oos_base_lift","oos_diff","oos_true_n"],ascending=False)
    out.to_csv("reports/50pip_mtf_feature_effects.csv",index=False,float_format="%.8f")

    inc=[]
    for tf,all_df in datasets.items():
        for direction in ("bull","bear"):
            features=feature_pool(direction)
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}" if direction=="bull" else f"hit_short50_h{h}"
                disc=all_df[all_df.timestamp.dt.year<=2024]
                base=float(disc[target].mean())
                candidates=[]
                for a in features:
                    sa=disc[disc[a].fillna(False)] if a in disc.columns else disc.iloc[0:0]
                    if len(sa)<150: continue
                    ha=float(sa[target].mean())
                    if ha/base>=1.10:
                        candidates.append((a,ha/base))
                candidates=sorted(candidates,key=lambda z:z[1],reverse=True)[:20]
                for a,_ in candidates:
                    sa_cache={}
                    for b in features:
                        if b==a or b not in all_df.columns: continue
                        vals=[tf,direction,h,a,b]
                        for g in (disc,all_df[all_df.timestamp.dt.year==2025],all_df[all_df.timestamp.dt.year>=2026]):
                            sa=g[g[a].fillna(False)]
                            sb=sa[sa[b].fillna(False)]
                            if len(sb)<50:
                                vals += [len(sb),np.nan,np.nan]
                            else:
                                rate=float(sb[target].mean())
                                arate=float(sa[target].mean())
                                vals += [len(sb),rate,rate-arate]
                        inc.append(vals)
    incdf=pd.DataFrame(inc,columns=[
        "timeframe","direction","horizon_bars","base_feature","added_feature",
        "disc_n","disc_rate","disc_increment","val_n","val_rate","val_increment",
        "oos_n","oos_rate","oos_increment"])
    incdf.to_csv("reports/50pip_mtf_incremental_effects.csv",index=False,float_format="%.8f")

    print("MTF feature effect rows:",len(out))
    print("MTF incremental rows:",len(incdf))
    print(out[["timeframe","direction","horizon_bars","feature","disc_base_lift","val_base_lift","oos_base_lift","oos_true_hit"]].head(40).to_string(index=False))

if __name__=="__main__":
    main()
