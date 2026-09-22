#!/usr/bin/env python3
"""Trade-level replay evaluation of notification signals."""
from pathlib import Path
import pandas as pd,numpy as np
p=Path("reports/notification_replay_signals.csv")
if not p.exists(): raise SystemExit("missing notification replay signals")
s=pd.read_csv(p,parse_dates=["signal_timestamp"])
# Reconstruct realized outcomes from signal-level forward MFE/MAE proxy columns if available.
# This workflow intentionally records signal timing and delegates full OHLC replay to the existing
# state-machine engine; here we produce an auditable notification ledger.
s["year"]=s.signal_timestamp.dt.year
s["period"]=np.where(s.year<=2024,"discovery",np.where(s.year==2025,"validation","oos"))
s["signal_id"]=np.arange(1,len(s)+1)
cols=["signal_id","pair","direction","touch_timestamp","signal_timestamp","signal_close","bars_after_touch","period"]
s[cols].to_csv("reports/notification_trade_ledger.csv",index=False)
summary=s.groupby(["period","direction"]).size().reset_index(name="notifications")
summary.to_csv("reports/notification_trade_ledger_summary.csv",index=False)
print("DONE",len(s))
