#!/usr/bin/env python3
"""Build an auditable notification-ready signal specification from the validated engine."""
from pathlib import Path
import pandas as pd
Path("reports").mkdir(exist_ok=True)
rows=[
["LONG","H4","20EMA>200EMA GC後","最初の20EMAタッチ","基準足高値を実体上抜け","SMA stack bull + DI strong bull","次足始値候補","managed exit; pre-entry invalidation requires 2 consecutive closes beyond touch low"],
["SHORT","H4","20EMA<200EMA DC後","最初の20EMAタッチ","基準足安値を実体下抜け","strong close bear","次足始値候補","managed exit; pre-entry invalidation requires 2 consecutive closes beyond touch high"],
]
pd.DataFrame(rows,columns=["direction","timeframe","state","setup","trigger","filter","entry","exit"]).to_csv("reports/notification_rule_spec.csv",index=False)
# Notification state machine: WAIT -> ARMED -> TRIGGERED -> MANAGED -> CLOSED
states=[
["WAIT","No valid setup","Do not notify"],
["ARMED","EMA state + first touch + candidate filter context valid","Notify only when trigger is confirmed"],
["TRIGGERED","Completed candle closes beyond reference level","Entry candidate notification"],
["MANAGED","Trade active; +50 reached and deterioration condition appears","Exit/reduce-risk notification"],
["CLOSED","Position exited/targeted/stopped","Record result; return WAIT"],
]
pd.DataFrame(states,columns=["state","condition","action"]).to_csv("reports/notification_state_machine.csv",index=False)
print("DONE")
