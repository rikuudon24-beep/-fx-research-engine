#!/usr/bin/env python3
"""Freeze monitoring universe using OOS consistency across targets; no optimization on 2026 data beyond predefined rule."""
from pathlib import Path
import pandas as pd,numpy as np
p=Path("reports/frozen_rule_oos_pair_robustness.csv")
if not p.exists(): raise SystemExit("missing robustness report")
z=pd.read_csv(p)
rows=[]
for (direction,pair),q in z.groupby(["direction","pair"]):
    q=q[q.target.isin([100,150,200,300])]
    rows.append([direction,pair,len(q),q["mean"].mean(),q["positive_rate"].mean(),(q["mean"]>0).sum(),(q["positive_rate"]>=0.5).sum()])
a=pd.DataFrame(rows,columns=["direction","pair","targets","mean_of_target_means","mean_positive_rate","positive_mean_targets","positive_rate_ge50_targets"])
# Monitoring eligibility is descriptive, not a promise of future performance:
# require >=3/4 positive target means AND >=2/4 target positive rates >=50%.
a["monitor_candidate"]=(a.positive_mean_targets>=3)&(a.positive_rate_ge50_targets>=2)
Path("reports").mkdir(exist_ok=True)
a.to_csv("reports/monitoring_universe_screen.csv",index=False,float_format="%.6f")
print("DONE")
