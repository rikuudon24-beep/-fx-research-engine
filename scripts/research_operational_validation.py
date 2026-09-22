#!/usr/bin/env python3
"""Operational candidate validation: pair robustness + cost/slippage sensitivity."""
from pathlib import Path
import pandas as pd,numpy as np
P=Path("reports/final_candidate_events.csv")
if not P.exists(): raise SystemExit("missing final_candidate_events.csv")
df=pd.read_csv(P,parse_dates=["entry_timestamp"])
C=[("long","sma_stack_bull+di_strong_bull"),("short","strong_close_bear")]
# Cost is charged round-trip in pips. Test realistic sensitivity without changing signal timing.
costs=[0,0.5,1,1.5,2,3]
rows=[]
for direction,filt in C:
 z=df[(df.direction==direction)&(df[filt].astype(bool))]
 for period in ["discovery","validation","oos"]:
  q=z[z.period==period]
  for c in costs:
   v=q.managed_pips.astype(float)-c
   rows.append([direction,filt,period,"managed",c,len(q),v.mean() if len(q) else np.nan,(v>0).mean() if len(q) else np.nan,v.sum() if len(q) else np.nan])
  # pair robustness in OOS
  if period=="oos":
   for pair,q2 in q.groupby("pair"):
    v=q2.managed_pips.astype(float)
    rows.append([direction,filt,"oos_pair",pair,0,len(q2),v.mean(),(v>0).mean(),v.sum()])
pd.DataFrame(rows,columns=["direction","filter","period","metric","cost_or_pair","trades","mean_pips","positive_rate","total_pips"]).to_csv("reports/operational_validation.csv",index=False,float_format="%.6f")
# Pair-level target robustness for OOS
pr=[]
for direction,filt in C:
 z=df[(df.direction==direction)&(df.period=="oos")&df[filt].astype(bool)]
 for pair,q in z.groupby("pair"):
  pr.append([direction,filt,pair,len(q),*(q[f"pnl{t}"].mean() for t in [100,150,200,300]),q.managed_pips.mean(),(q.managed_pips>0).mean()])
pd.DataFrame(pr,columns=["direction","filter","pair","trades","mean100","mean150","mean200","mean300","managed_mean","managed_positive_rate"]).to_csv("reports/operational_pair_robustness.csv",index=False,float_format="%.6f")
print("DONE")
