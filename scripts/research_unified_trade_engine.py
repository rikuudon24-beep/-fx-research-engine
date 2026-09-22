#!/usr/bin/env python3
"""Unified operational trade engine: direction-specific entry, adaptive target, managed exit, costs."""
from pathlib import Path
import pandas as pd,numpy as np
P=Path("reports/final_candidate_events.csv")
if not P.exists(): raise SystemExit("missing final_candidate_events.csv")
df=pd.read_csv(P,parse_dates=["entry_timestamp"])
rules={"long":"sma_stack_bull+di_strong_bull","short":"strong_close_bear"}
costs=[0,1,2,3]
rows=[]
for direction,filt in rules.items():
 z=df[(df.direction==direction)&df[filt].astype(bool)].copy()
 for period in ["discovery","validation","oos"]:
  q=z[z.period==period]
  for cost in costs:
   if not len(q): continue
   v=q.managed_pips-cost
   rows.append([direction,filt,period,cost,len(q),v.mean(),(v>0).mean(),v.sum(),v.std(ddof=0)])
# OOS pair robustness
pair=[]
for direction,filt in rules.items():
 z=df[(df.direction==direction)&(df.period=="oos")&df[filt].astype(bool)]
 for p,q in z.groupby("pair"):
  for cost in [0,1,2,3]:
   v=q.managed_pips-cost
   pair.append([direction,filt,p,cost,len(q),v.mean(),(v>0).mean(),v.sum()])
Path("reports").mkdir(exist_ok=True)
pd.DataFrame(rows,columns=["direction","filter","period","cost_pips","trades","mean_pips","positive_rate","total_pips","std_pips"]).to_csv("reports/unified_engine_results.csv",index=False,float_format="%.6f")
pd.DataFrame(pair,columns=["direction","filter","pair","cost_pips","trades","mean_pips","positive_rate","total_pips"]).to_csv("reports/unified_engine_pair_robustness.csv",index=False,float_format="%.6f")
print("DONE")
