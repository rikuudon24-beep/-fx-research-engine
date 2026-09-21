#!/usr/bin/env python3
"""Focused pin-bear early-entry research.

The chart-toolkit study found H4 bear pin_bear to be a strong +50pip
reachability subgroup. This study asks the practical question: can we make
that subgroup earlier/purer without selecting on future information?

All filters use only the current and completed prior H4 candles.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("direct","scripts/research_50pip_direct_entry.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

PAIRS=mod.PAIRS
HORIZONS=mod.HORIZONS["h4"]

def build_pair(pair):
    df=mod.load_market("h4",pair).copy()
    f=mod.build_features(df)
    lab=mod.add_labels(df,pair,"h4")
    z=pd.concat([df[["timestamp","open","high","low","close"]],f,lab],axis=1)
    z["pair"]=pair

    # Causal timing/late-entry context. These never inspect future candles.
    c=z.close; h=z.high; l=z.low
    atr=pd.to_numeric(z.get("atr14"),errors="coerce")
    if atr.isna().all():
        # build_features currently exposes ATR under one of these names.
        for n in ("atr","ATR14","atr_14"):
            if n in z.columns:
                atr=pd.to_numeric(z[n],errors="coerce"); break
    if atr.isna().all():
        tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        atr=tr.rolling(14).mean()

    prior_high3=h.shift(1).rolling(3).max()
    prior_low3=l.shift(1).rolling(3).min()
    prior_high10=h.shift(1).rolling(10).max()
    prior_low10=l.shift(1).rolling(10).min()

    # How much bearish travel has already occurred before the pin candle?
    prior_drop=(prior_high3-c).clip(lower=0)
    prior_drop10=(prior_high10-c).clip(lower=0)
    # How close are we to the recent high: a proxy for "not already extended down".
    from_high=(prior_high10-c).clip(lower=0)

    z["pin_bear"]=z["pin_bear"].fillna(False)
    z["prior_drop_atr3"]=prior_drop/atr.replace(0,np.nan)
    z["prior_drop_atr10"]=prior_drop10/atr.replace(0,np.nan)
    z["from_high_atr10"]=from_high/atr.replace(0,np.nan)
    z["range_atr"]=(h-l)/atr.replace(0,np.nan)
    z["atr_expand"]=z["atr_expand"].fillna(False) if "atr_expand" in z else (atr>atr.shift(3)*1.05)
    z["adx25"]=z["adx25"].fillna(False) if "adx25" in z else False
    z["adx_rising"]=z["adx_rising"].fillna(False) if "adx_rising" in z else False
    z["di_strong_bear"]=z["di_strong_bear"].fillna(False) if "di_strong_bear" in z else False
    z["macd_bear"]=z["macd_bear"].fillna(False) if "macd_bear" in z else False
    z["price200_bear"]=z["price200_bear"].fillna(False) if "price200_bear" in z else False
    z["bb_bear"]=z["bb_bear"].fillna(False) if "bb_bear" in z else False
    z["breakout_down20"]=z["breakout_down20"].fillna(False) if "breakout_down20" in z else False
    z["retest_down"]=z["retest_down"].fillna(False) if "retest_down" in z else False
    z["elliott_pullback_bear"]=z["elliott_pullback_bear"].fillna(False) if "elliott_pullback_bear" in z else False
    z["range_tight"]=z["range_tight"].fillna(False) if "range_tight" in z else False
    return z

def eval_mask(g, mask, target):
    n=int(mask.sum())
    if n==0: return n,np.nan
    return n,float(g.loc[mask,target].mean())

def main():
    Path("reports").mkdir(exist_ok=True)
    frames=[build_pair(p) for p in PAIRS]
    all_df=pd.concat(frames,ignore_index=True)

    # Candidate filters: first test each alone, then compact combinations.
    filters={
      "early_drop_atr3<=0.5": all_df.prior_drop_atr3<=0.5,
      "early_drop_atr3<=1.0": all_df.prior_drop_atr3<=1.0,
      "early_drop_atr3<=1.5": all_df.prior_drop_atr3<=1.5,
      "early_drop_atr10<=1.0": all_df.prior_drop_atr10<=1.0,
      "near_recent_high_atr10<=1.0": all_df.from_high_atr10<=1.0,
      "near_recent_high_atr10<=1.5": all_df.from_high_atr10<=1.5,
      "range_atr<=1.5": all_df.range_atr<=1.5,
      "atr_expand": all_df.atr_expand,
      "adx25": all_df.adx25,
      "adx_rising": all_df.adx_rising,
      "di_strong_bear": all_df.di_strong_bear,
      "macd_bear": all_df.macd_bear,
      "price200_bear": all_df.price200_bear,
      "bb_bear": all_df.bb_bear,
      "breakout_down20": all_df.breakout_down20,
      "retest_down": all_df.retest_down,
      "elliott_pullback_bear": all_df.elliott_pullback_bear,
      "range_tight": all_df.range_tight,
    }
    combos={
      "pin": [],
    }
    names=list(filters)
    for a in names:
        combos["pin+"+a]=[a]
    # High-value practical confirmations, deliberately bounded to avoid a
    # combinatorial search that encourages overfit.
    confirms=["atr_expand","adx25","adx_rising","di_strong_bear","macd_bear",
              "price200_bear","bb_bear","breakout_down20","retest_down",
              "elliott_pullback_bear","range_tight"]
    timing=["early_drop_atr3<=0.5","early_drop_atr3<=1.0","early_drop_atr10<=1.0",
            "near_recent_high_atr10<=1.0","near_recent_high_atr10<=1.5"]
    for t in timing:
        for c in confirms:
            combos["pin+"+t+"+"+c]=[t,c]

    rows=[]
    for h in HORIZONS:
      target=f"hit_short50_h{h}"
      for label,parts in combos.items():
        mask=all_df.pin_bear.fillna(False)
        for p in parts: mask &= filters[p].fillna(False)
        vals=[h,label]
        for period,g in (("disc",all_df[all_df.timestamp.dt.year<=2024]),
                         ("val",all_df[all_df.timestamp.dt.year==2025]),
                         ("oos",all_df[all_df.timestamp.dt.year>=2026])):
            gm=mask.loc[g.index]
            n,rate=eval_mask(g,gm,target)
            base=float(g[target].mean())
            vals += [n,rate,rate/base if base and np.isfinite(rate) else np.nan]
        # OOS pair robustness
        oos=all_df[all_df.timestamp.dt.year>=2026]
        gm=mask.loc[oos.index]
        pair_rates=[]
        pair_above=0
        pair_n=0
        for pair,pg in oos.groupby("pair"):
            pm=gm.loc[pg.index]
            if int(pm.sum())>=5:
                pr=float(pg.loc[pm,target].mean()); pair_rates.append(pr)
                pair_n+=1
                if pr>float(pg[target].mean()): pair_above+=1
        vals += [pair_n,pair_above,float(np.mean(pair_rates)) if pair_rates else np.nan,
                 float(np.min(pair_rates)) if pair_rates else np.nan]
        rows.append(vals)

    cols=["horizon_bars","condition",
          "disc_n","disc_hit","disc_lift","val_n","val_hit","val_lift",
          "oos_n","oos_hit","oos_lift","oos_pairs_ge5","oos_pairs_above_base",
          "oos_pair_mean","oos_pair_min"]
    out=pd.DataFrame(rows,columns=cols)
    out=out.sort_values(["horizon_bars","oos_lift","oos_n"],ascending=[True,False,False])
    out.to_csv("reports/50pip_pin_bear_focus.csv",index=False,float_format="%.8f")

    # Extension ladder for the strongest early-entry candidates at h=1.
    ladder=[]
    top=out[out.horizon_bars==1].query("oos_n>=50").head(25)
    for _,r in top.iterrows():
        label=r.condition
        mask=all_df.pin_bear.fillna(False)
        for p in combos[label]: mask &= filters[p].fillna(False)
        for th in (50,100,150,200,300):
            target=f"hit_short{th}_h1"
            oos=all_df[all_df.timestamp.dt.year>=2026]
            gm=mask.loc[oos.index]
            ladder.append([label,th,int(gm.sum()),float(oos.loc[gm,target].mean()) if int(gm.sum()) else np.nan])
    pd.DataFrame(ladder,columns=["condition","threshold_pips","oos_n","oos_hit"]).to_csv(
        "reports/50pip_pin_bear_extension.csv",index=False,float_format="%.8f")

    print("PIN_BEAR_FOCUS_DONE",len(out))
    print(out[out.horizon_bars==1].head(35).to_string(index=False))

if __name__=="__main__":
    main()
