#!/usr/bin/env python3
"""Analyze continuation paths for the strongest H4 bearish 100pip research family."""
from pathlib import Path
import pandas as pd
import numpy as np
from research_50pip_continuation import load_events, event_features, directional_conditions

PAIRS=["usdjpy","eurjpy","gbpjpy","audjpy","eurusd","gbpusd","audusd","nzdusd","usdcad","usdchf","audnzd","eurgbp"]

def main():
    all_events=load_events()
    focus=pd.read_csv("reports/50pip_focus_pair_detail.csv")
    focus_pairs=focus["pair"].drop_duplicates().tolist()
    focus_conditions=focus["conditions"].drop_duplicates().tolist()
    base=all_events[(all_events.timeframe=="h4")&(all_events.direction=="bear")&(all_events.threshold_pips==50)].copy()
    events=event_features(base)
    rows=[]
    for cond in focus_conditions:
        names=cond.split("+")
        for pair in focus_pairs:
            e=base[base.pair==pair].copy()
            # Reuse the already generated pair-level OOS candidate membership.
            detail=focus[(focus.conditions==cond)&(focus.pair==pair)]
            if detail.empty: continue
            conds=directional_conditions("bear")
            if not all(n in conds for n in names): continue
            mask=pd.Series(True,index=e.index)
            for n in names: mask &= conds[n](e).fillna(False)
            e=e[mask].copy()
            for _,x in e.iterrows():
                key=["pair","timeframe","direction","event_timestamp"]
                sub=all_events[(all_events.pair==pair)&(all_events.timeframe=="h4")&(all_events.direction=="bear")&(all_events.event_timestamp==x.event_timestamp)]
                hit={}
                for th in [100,150,200,300]:
                    hit[th]=not sub[sub.threshold_pips==th].empty
                rows.append([cond,pair,x.event_timestamp,int(hit[100]),int(hit[150]),int(hit[200]),int(hit[300])])
    out=pd.DataFrame(rows,columns=["conditions","pair","event_timestamp","hit100","hit150","hit200","hit300"])
    out.to_csv("reports/50pip_trade_path_focus.csv",index=False)
    summary=(out.groupby(["conditions","pair"])
        .agg(events=("event_timestamp","size"),hit100_rate=("hit100","mean"),
             hit150_rate=("hit150","mean"),hit200_rate=("hit200","mean"),
             hit300_rate=("hit300","mean"))
        .reset_index())
    summary.to_csv("reports/50pip_trade_path_focus_summary.csv",index=False)
    print("trade_path_rows",len(out))
    print(summary.to_string(index=False) if not summary.empty else "no trade path rows")

if __name__=="__main__":
    main()
