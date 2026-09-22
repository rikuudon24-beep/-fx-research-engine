#!/usr/bin/env python3
"""State-machine backtest of the established H4 20/200 EMA pullback entry.

Entry (completed candles only):
1) 20 EMA crosses 200 EMA.
2) Only the first subsequent touch of 20 EMA is eligible.
3) The touch candle is the reference candle.
4) Long: a later candle must CLOSE above the reference high.
   Short: a later candle must CLOSE below the reference low.
5) A reverse 20/200 cross before entry invalidates the setup.

After entry, compare three exit policies:
- target_hold: fixed +100/-100 pip target, otherwise close at max 24 H4 bars.
- ladder_exit: after +50, use the fixed independent-exit condition to exit;
  before +50, use a reference-candle stop.
- hold_plus_exit: after +50, use the same exit condition, otherwise hold to
  +100 or the 24-bar time limit.

This is a causal research backtest, not an execution recommendation.
"""
from pathlib import Path
import importlib.util, numpy as np, pandas as pd

spec=importlib.util.spec_from_file_location("d","scripts/research_50pip_direct_entry.py")
d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
PAIRS=d.PAIRS; PIP=d.PIP

def features(df):
    c,h,l=df.close.astype(float),df.high.astype(float),df.low.astype(float)
    e20=c.ewm(span=20,adjust=False).mean()
    e200=c.ewm(span=200,adjust=False).mean()
    return e20,e200

def exit_cond(row):
    return bool(row.get("price20_cross_down",False) and row.get("di_spread_down3",False))

def build(tf="h4"):
    rows=[]
    for pair in PAIRS:
        g=d.load_market(tf,pair).sort_values("timestamp").reset_index(drop=True)
        # Reuse causal feature builder for exit state.
        f=d.build_features(g)
        e20,e200=features(g)
        g["ema20"]=e20; g["ema200"]=e200
        g["cross_up"]=(e20.shift(1)<=e200.shift(1))&(e20>e200)
        g["cross_down"]=(e20.shift(1)>=e200.shift(1))&(e20<e200)
        g["exit_cond"]=exit_cond_series(f)
        pip=PIP[pair]
        i=1
        while i<len(g)-25:
            cross_dir=None
            if bool(g.cross_up.iloc[i]): cross_dir="long"
            elif bool(g.cross_down.iloc[i]): cross_dir="short"
            if cross_dir is None:
                i+=1; continue
            # Find first touch after the cross; reverse cross invalidates.
            touch=None
            for j in range(i+1,min(len(g),i+25)):
                if (cross_dir=="long" and bool(g.cross_down.iloc[j])) or (cross_dir=="short" and bool(g.cross_up.iloc[j])):
                    break
                touched = float(g.low.iloc[j])<=float(g.ema20.iloc[j])<=float(g.high.iloc[j])
                if touched:
                    touch=j; break
            if touch is None:
                i+=1; continue
            ref_hi=float(g.high.iloc[touch]); ref_lo=float(g.low.iloc[touch])
            entry=None
            for j in range(touch+1,min(len(g),touch+13)):
                if (cross_dir=="long" and bool(g.cross_down.iloc[j])) or (cross_dir=="short" and bool(g.cross_up.iloc[j])):
                    break
                if cross_dir=="long" and float(g.close.iloc[j])>ref_hi:
                    entry=j; break
                if cross_dir=="short" and float(g.close.iloc[j])<ref_lo:
                    entry=j; break
            if entry is None:
                i=touch+1; continue
            entry_px=float(g.close.iloc[entry])
            stop_px=ref_lo if cross_dir=="long" else ref_hi
            end=min(len(g),entry+25)
            if end<=entry+1: break
            x=g.iloc[entry+1:end]
            direction=1 if cross_dir=="long" else -1
            stop_idx=None; target_idx=None; exit_idx=None
            for k in x.index:
                hi=float(g.high.iloc[k]); lo=float(g.low.iloc[k])
                if direction==1:
                    if lo<=stop_px: stop_idx=k; break
                    if hi>=entry_px+100*pip: target_idx=k; break
                else:
                    if hi>=stop_px: stop_idx=k; break
                    if lo<=entry_px-100*pip: target_idx=k; break
            # Conservative same-candle ordering: stop wins if both are touched.
            if stop_idx is not None: base_idx=stop_idx; base_px=stop_px
            elif target_idx is not None: base_idx=target_idx; base_px=entry_px+direction*100*pip
            else: base_idx=x.index[-1]; base_px=float(g.close.iloc[base_idx])
            base_pips=direction*(base_px-entry_px)/pip
            # Exit-policy: first exit condition after +50, but only after +50 has
            # actually been reached. If stop occurs first, exit at stop.
            ladder_idx=None; ladder_px=None; reached50=False
            for k in x.index:
                hi=float(g.high.iloc[k]); lo=float(g.low.iloc[k])
                if direction==1:
                    if lo<=stop_px: ladder_idx=k; ladder_px=stop_px; break
                    if hi>=entry_px+50*pip: reached50=True
                    if reached50 and bool(g.exit_cond.iloc[k]):
                        ladder_idx=k; ladder_px=float(g.close.iloc[k]); break
                    if hi>=entry_px+100*pip: ladder_idx=k; ladder_px=entry_px+100*pip; break
                else:
                    if hi>=stop_px: ladder_idx=k; ladder_px=stop_px; break
                    if lo<=entry_px-50*pip: reached50=True
                    if reached50 and bool(g.exit_cond.iloc[k]):
                        ladder_idx=k; ladder_px=float(g.close.iloc[k]); break
                    if lo<=entry_px-100*pip: ladder_idx=k; ladder_px=entry_px-100*pip; break
            if ladder_idx is None: ladder_idx=x.index[-1]; ladder_px=float(g.close.iloc[ladder_idx])
            ladder_pips=direction*(ladder_px-entry_px)/pip
            rows.append([tf,pair,cross_dir,g.timestamp.iloc[i],g.timestamp.iloc[touch],g.timestamp.iloc[entry],
                         entry_px,stop_px,base_pips,ladder_pips,
                         int(base_idx-entry),int(ladder_idx-entry)])
            i=entry+1
    return pd.DataFrame(rows,columns=["timeframe","pair","direction","cross_timestamp","touch_timestamp","entry_timestamp",
        "entry_price","stop_price","target100_pips","managed_exit_pips","bars_base","bars_managed"])

def exit_cond_series(f):
    # independent-exit condition, computed causally from the same completed candle.
    c=f["close"].astype(float); e20=c.ewm(span=20,adjust=False).mean()
    d=c.diff(); up=f.high.diff(); dn=-f.low.diff()
    atr=pd.concat([f.high-f.low,(f.high-c.shift()).abs(),(f.low-c.shift()).abs()],axis=1).max(axis=1).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    plus=pd.Series(np.where((up>dn)&(up>0),up,0),index=f.index).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    minus=pd.Series(np.where((dn>up)&(dn>0),dn,0),index=f.index).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    pdi=100*plus/atr; mdi=100*minus/atr
    spread=pdi-mdi
    return (c.shift(1)>=e20.shift(1))&(c<e20)&(spread.rolling(3).mean().diff()<0)

def main():
    out=build("h4"); Path("reports").mkdir(exist_ok=True)
    out["period"]=np.where(out.entry_timestamp.dt.year<=2024,"discovery",np.where(out.entry_timestamp.dt.year==2025,"validation","oos"))
    out.to_csv("reports/state_machine_trade_events.csv",index=False)
    summary=[]
    for p,x in out.groupby(["period","direction"],sort=False):
        summary.append([*p,len(x),x.target100_pips.mean(),(x.target100_pips>0).mean(),x.managed_exit_pips.mean(),(x.managed_exit_pips>0).mean(),x.target100_pips.sum(),x.managed_exit_pips.sum(),x.target100_pips.min(),x.managed_exit_pips.min()])
    pd.DataFrame(summary,columns=["period","direction","trades","target100_mean","target100_positive_rate","managed_mean","managed_positive_rate","target100_total","managed_total","target100_min","managed_min"]).to_csv("reports/state_machine_trade_summary.csv",index=False,float_format="%.6f")
    print("DONE",len(out))
if __name__=="__main__": main()
