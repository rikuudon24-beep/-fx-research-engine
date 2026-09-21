import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("direct",ROOT/"scripts"/"research_50pip_direct_entry.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

PAIRS=mod.PAIRS
HORIZONS=[1,3,6,12,24]
LAGS=range(0,7)
CONF=["adx_rising","adx25","di_strong_bear","macd_bear","atr_expand","bb_bear","breakout_down20","retest_down","elliott_pullback_bear","range_tight"]
BASE=["price20_bear","price200_bear","trend_bear","volatility","body"]
TRANS=["adx_up3","macd_accel_bear","rsi_down1","di_cross_down","ema20_slope_down","price20_cross_down","price200_cross_down","atr_expand","bb_expand","range_expand","breakout_down20","retest_down","false_breakout_down"]

def build(p):
    df=mod.load_market("h4",p)
    z=mod.build_features(df).copy()
    labels=mod.add_labels(df,p,"h4")
    z=pd.concat([df, z, labels],axis=1)
    if "pin_bear" not in z: raise RuntimeError("pin_bear missing")
    return z

def transition(s):
    b=s.fillna(False).astype(bool)
    return b & ~b.shift(1).fillna(False)

def evaluate(g, mask, h):
    target=f"hit_short50_h{h}"
    x=g[mask & g[target].notna()]
    return len(x), float(x[target].mean()) if len(x) else np.nan

rows=[]; seqrows=[]
for p in PAIRS:
    z=build(p)
    z["pair"]=p
    for c in TRANS:
        if c in z: z[c+"_tr"]=transition(z[c])
    for h in HORIZONS:
        y=z[z["pin_bear"].fillna(False)].copy()
        if y.empty: continue
        for lag in LAGS:
            for c in CONF+BASE+TRANS:
                col=c+"_tr" if c in TRANS else c
                if col not in z: continue
                m=z[col].shift(lag).reindex(y.index).fillna(False).astype(bool)
                n,hit=evaluate(y,m,h)
                if n: rows.append({"pair":p,"h":h,"lag":lag,"feature":c,"n":n,"hit":hit})
        for a in TRANS:
            ca=a+"_tr"
            if ca not in z: continue
            for b in TRANS:
                cb=b+"_tr"
                if cb not in z or a==b: continue
                for gap in (1,2,3):
                    m=(z[ca].shift(gap).fillna(False)&z[cb].fillna(False)).reindex(y.index).fillna(False)
                    n,hit=evaluate(y,m,h)
                    if n>=5: seqrows.append({"pair":p,"h":h,"a":a,"b":b,"gap":gap,"n":n,"hit":hit})

out=ROOT/"reports"; out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out/"50pip_pin_bear_precursor_lags.csv",index=False)
pd.DataFrame(seqrows).to_csv(out/"50pip_pin_bear_precursor_sequences.csv",index=False)

# aggregate only candidates with meaningful sample sizes, separated by discovery/validation/OOS
allrows=pd.DataFrame(rows)
if not allrows.empty:
    allrows["date"]=pd.to_datetime(allrows.get("date"),errors="coerce")
    # pair-level raw results are retained; this aggregate is descriptive.
    agg=(allrows.groupby(["h","lag","feature"],as_index=False)
         .agg(n=("n","sum"), weighted_hit=("hit",lambda x: np.nan)))
    # Recompute weighted hit from pair-level hit/n.
    tmp=allrows.assign(w=allrows["hit"]*allrows["n"])
    agg=(tmp.groupby(["h","lag","feature"],as_index=False)
         .agg(n=("n","sum"),wins=("w","sum")))
    agg["hit"]=agg["wins"]/agg["n"]
    agg["rank_score"]=agg["hit"]*np.log1p(agg["n"])
    agg.sort_values(["h","rank_score"],ascending=[True,False]).to_csv(out/"50pip_pin_bear_precursor_summary.csv",index=False)
