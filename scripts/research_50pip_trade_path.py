#!/usr/bin/env python3
"""Analyze continuation paths for the strongest H4 bearish 100pip research family."""
from pathlib import Path
import pandas as pd
import numpy as np

PAIRS=["usdjpy","eurjpy","gbpjpy","audjpy","eurusd","gbpusd","audusd","nzdusd","usdcad","usdchf","audnzd","eurgbp"]

def main():
    events=pd.read_csv("reports/50pip_events.csv")
    focus=pd.read_csv("reports/50pip_focus_pair_detail.csv")
    focus_pairs=focus["pair"].drop_duplicates().tolist()
    focus_conditions=focus["conditions"].drop_duplicates().tolist()
    base=events[(events.timeframe=="h4")&(events.direction=="bear")&(events.threshold_pips==50)].copy()
    rows=[]
    for cond in focus_conditions:
        names=cond.split("+")
        for pair in focus_pairs:
            e=base[base.pair==pair].copy()
            # Reuse the already generated pair-level OOS candidate membership.
            detail=focus[(focus.conditions==cond)&(focus.pair==pair)]
            if detail.empty: continue
            # Map candidate events through the source condition dataset.
            feat=pd.read_csv("reports/50pip_feature_dataset.csv")
            feat=feat[(feat.timeframe=="h4")&(feat.pair==pair)]
            if not all(n in feat.columns for n in names): continue
            mask=np.ones(len(feat),dtype=bool)
            for n in names: mask &= feat[n].fillna(False).astype(bool).to_numpy()
            cand=feat.loc[mask,["event_timestamp"]]
            e=e.merge(cand,on="event_timestamp",how="inner")
            for _,x in e.iterrows():
                key=["pair","timeframe","direction","event_timestamp"]
                sub=events[(events.pair==pair)&(events.timeframe=="h4")&(events.direction=="bear")&(events.event_timestamp==x.event_timestamp)]
                hit={}
                for th in [100,150,200,300]:
                    hit[th]=not sub[sub.threshold_pips==th].empty
                rows.append([cond,pair,x.event_timestamp,int(hit[100]),int(hit[150]),int(hit[200]),int(hit[300])])
    out=pd.DataFrame(rows,columns=["conditions","pair","event_timestamp","hit100","hit150","hit200","hit300"])
    out.to_csv("reports/50pip_trade_path_focus.csv",index=False)
    if out.empty:
        return
    summary=(out.groupby(["conditions","pair"])
        .agg(events=("event_timestamp","size"),hit100_rate=("hit100","mean"),
             hit150_rate=("hit150","mean"),hit200_rate=("hit200","mean"),
             hit300_rate=("hit300","mean"))
        .reset_index())
    summary.to_csv("reports/50pip_trade_path_focus_summary.csv",index=False)
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
