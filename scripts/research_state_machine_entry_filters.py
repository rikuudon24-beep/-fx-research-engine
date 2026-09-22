#!/usr/bin/env python3
"""Trade-level entry-filter research for the H4 20/200 EMA state machine.

Discovery <=2024 selects causal, non-pin entry filters. Validation=2025 confirms
direction-specific candidates. OOS >=2026 is held out until final scoring.
All filters are evaluated on the completed entry candle.
"""
from pathlib import Path
import importlib.util, itertools, numpy as np, pandas as pd

spec=importlib.util.spec_from_file_location("d","scripts/research_50pip_direct_entry.py")
d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
PAIRS=d.PAIRS; PIP=d.PIP

CANDS={
"long":["price200_bull","sma_stack_bull","volatility","adx25","adx_up3","atr_expand","bb_expand",
        "strong_close_bull","elliott_impulse_bull","breakout20_up","horizontal_break_up",
        "rsi_cross55_up","di_strong_bull","ema20_rising","trendline_up"],
"short":["price200_bear","sma_stack_bear","volatility","adx25","adx_down3","atr_expand","bb_expand",
         "strong_close_bear","elliott_impulse_bear","breakout20_down","horizontal_break_down",
         "rsi_cross45_down","di_strong_bear","ema20_falling","trendline_down"]
}

def make_trades():
    rows=[]
    for pair in PAIRS:
        g=d.load_market("h4",pair).sort_values("timestamp").reset_index(drop=True)
        f=d.build_features(g)
        e20=g.close.ewm(span=20,adjust=False).mean()
        e200=g.close.ewm(span=200,adjust=False).mean()
        g["ema20"]=e20; g["ema200"]=e200
        g["cross_up"]=(e20.shift(1)<=e200.shift(1))&(e20>e200)
        g["cross_down"]=(e20.shift(1)>=e200.shift(1))&(e20<e200)
        g["exit_long"]=f["price20_cross_down"]&f["di_spread_down3"]
        g["exit_short"]=f["price20_cross_up"]&f["di_spread_up3"]
        for name in f.columns: g[name]=f[name]
        pip=PIP[pair]; i=1
        while i<len(g)-25:
            direction="long" if bool(g.cross_up.iloc[i]) else ("short" if bool(g.cross_down.iloc[i]) else None)
            if direction is None: i+=1; continue
            touch=None
            for j in range(i+1,min(len(g),i+25)):
                if (direction=="long" and bool(g.cross_down.iloc[j])) or (direction=="short" and bool(g.cross_up.iloc[j])): break
                if float(g.low.iloc[j])<=float(g.ema20.iloc[j])<=float(g.high.iloc[j]):
                    touch=j; break
            if touch is None: i+=1; continue
            ref_hi=float(g.high.iloc[touch]); ref_lo=float(g.low.iloc[touch]); entry=None
            for j in range(touch+1,min(len(g),touch+13)):
                if (direction=="long" and bool(g.cross_down.iloc[j])) or (direction=="short" and bool(g.cross_up.iloc[j])): break
                if direction=="long" and float(g.close.iloc[j])>ref_hi: entry=j; break
                if direction=="short" and float(g.close.iloc[j])<ref_lo: entry=j; break
            if entry is None: i=touch+1; continue
            entry_px=float(g.close.iloc[entry]); stop_px=ref_lo if direction=="long" else ref_hi
            x=g.iloc[entry+1:min(len(g),entry+25)]; sign=1 if direction=="long" else -1
            if len(x)==0: break
            stop_idx=target_idx=None
            for k in x.index:
                hi=float(g.high.iloc[k]); lo=float(g.low.iloc[k])
                if (direction=="long" and lo<=stop_px) or (direction=="short" and hi>=stop_px): stop_idx=k; break
                if (direction=="long" and hi>=entry_px+100*pip) or (direction=="short" and lo<=entry_px-100*pip): target_idx=k; break
            if stop_idx is not None: base_idx=stop_idx; base_px=stop_px
            elif target_idx is not None: base_idx=target_idx; base_px=entry_px+sign*100*pip
            else: base_idx=x.index[-1]; base_px=float(g.close.iloc[base_idx])
            reached=False; mx_idx=None; mx_px=None
            for k in x.index:
                hi=float(g.high.iloc[k]); lo=float(g.low.iloc[k])
                if (direction=="long" and lo<=stop_px) or (direction=="short" and hi>=stop_px):
                    mx_idx=k; mx_px=stop_px; break
                if direction=="long" and hi>=entry_px+50*pip: reached=True
                if direction=="short" and lo<=entry_px-50*pip: reached=True
                if reached and bool(g.exit_long.iloc[k] if direction=="long" else g.exit_short.iloc[k]):
                    mx_idx=k; mx_px=float(g.close.iloc[k]); break
                if (direction=="long" and hi>=entry_px+100*pip) or (direction=="short" and lo<=entry_px-100*pip):
                    mx_idx=k; mx_px=entry_px+sign*100*pip; break
            if mx_idx is None: mx_idx=x.index[-1]; mx_px=float(g.close.iloc[mx_idx])
            row={"pair":pair,"direction":direction,"entry_timestamp":g.timestamp.iloc[entry],
                 "target100_pips":sign*(base_px-entry_px)/pip,"managed_exit_pips":sign*(mx_px-entry_px)/pip}
            for c in CANDS[direction]: row[c]=bool(g[c].iloc[entry])
            rows.append(row); i=entry+1
    out=pd.DataFrame(rows)
    out["period"]=np.where(out.entry_timestamp.dt.year<=2024,"discovery",np.where(out.entry_timestamp.dt.year==2025,"validation","oos"))
    return out

def stats(x):
    if len(x)==0: return [0,np.nan,np.nan,np.nan,np.nan]
    vals=x.target100_pips
    return [len(x),vals.mean(),(vals>0).mean(),vals.sum(),(x.managed_exit_pips).mean()]

def main():
    out=make_trades(); Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/state_machine_entry_filter_events.csv",index=False)
    rows=[]
    for direction in ["long","short"]:
        base={p:stats(out[(out.period==p)&(out.direction==direction)]) for p in ["discovery","validation","oos"]}
        cands=CANDS[direction]
        combos=[(c,) for c in cands]+list(itertools.combinations(cands,2))
        for combo in combos:
            name="+".join(combo)
            for p in ["discovery","validation","oos"]:
                z=out[(out.period==p)&(out.direction==direction)]
                for c in combo: z=z[z[c]]
                a=stats(z); b=base[p]
                rows.append([direction,name,p,*a, a[1]-b[1] if np.isfinite(a[1]) and np.isfinite(b[1]) else np.nan])
    res=pd.DataFrame(rows,columns=["direction","filter","period","trades","target100_mean","positive_rate","total_pips","managed_mean","mean_delta_vs_baseline"])
    res.to_csv("reports/state_machine_entry_filter_results.csv",index=False,float_format="%.6f")
    # Selection uses discovery + validation only; OOS remains held out.
    sel=[]
    for direction in ["long","short"]:
        for name in sorted(res[res.direction==direction].filter.unique()):
            a=res[(res.direction==direction)&(res.filter==name)&(res.period=="discovery")].iloc[0]
            b=res[(res.direction==direction)&(res.filter==name)&(res.period=="validation")].iloc[0]
            if a.trades>=25 and b.trades>=10 and a.mean_delta_vs_baseline>0 and b.mean_delta_vs_baseline>0:
                sel.append([direction,name,a.trades,b.trades,a.target100_mean,b.target100_mean])
    pd.DataFrame(sel,columns=["direction","filter","discovery_trades","validation_trades","discovery_mean","validation_mean"]).to_csv(
        "reports/state_machine_entry_filter_shortlist.csv",index=False,float_format="%.6f")
    print("DONE",len(out),"trades",len(res),"candidate rows",len(sel),"selected")

if __name__=="__main__": main()
