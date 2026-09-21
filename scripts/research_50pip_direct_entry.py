#!/usr/bin/env python3
"""Direct +50 entry research.

A signal is evaluated on every eligible candle. Success means price reaches
+50 pips after entry; 100/150/200/300 are extension metrics. This first pass
keeps the search bounded and supports both long and short directions.
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

PAIRS=["usdjpy","eurjpy","gbpjpy","audjpy","eurusd","gbpusd","audusd","nzdusd","usdcad","usdchf","audnzd","eurgbp"]
TFS=["h4","d1"]
PIP={p:0.01 if "jpy" in p else 0.0001 for p in PAIRS}
HORIZONS={"h4":[1,3,6,12,24],"d1":[1,3,5,10,20]}
LADDER=[50,100,150,200,300]

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def build_features(df):
    c,h,l,o=df.close,df.high,df.low,df.open
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0); loss=-d.clip(upper=0)
    rsi=100-100/(1+rma(g,14)/rma(loss,14).replace(0,np.nan))
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); mh=macd-macd.ewm(span=9,adjust=False).mean()
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=rma(tr,14)
    up=h.diff(); dn=-l.diff()
    plus=pd.Series(np.where((up>dn)&(up>0),up,0),index=df.index)
    minus=pd.Series(np.where((dn>up)&(dn>0),dn,0),index=df.index)
    pdi=100*rma(plus,14)/atr; mdi=100*rma(minus,14)/atr
    adx=rma(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan),14)
    atr_pct=atr.rolling(100).rank(pct=True)
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); bbpos=(c-(mid-2*sd))/(4*sd).replace(0,np.nan)
    bbwidth=(4*sd/mid.replace(0,np.nan))
    body=(c-o).abs()/(h-l).replace(0,np.nan)

    # State features: explicitly directional. Never derive bear signals by negating bull signals.
    f=pd.DataFrame({
        "trend_bull":(e20>e50)&(e50>e200), "trend_bear":(e20<e50)&(e50<e200),
        "price20_bull":c>e20, "price20_bear":c<e20,
        "price200_bull":c>e200, "price200_bear":c<e200,
        "ema20_rising":e20.diff()>0, "ema20_falling":e20.diff()<0,
        "ema200_rising":e200.diff()>0, "ema200_falling":e200.diff()<0,
        "momentum_bull":(rsi>=55)&(rsi<75), "momentum_bear":(rsi<=45)&(rsi>25),
        "rsi60":rsi>=60, "rsi40":rsi<=40,
        "macd_bull":mh>0, "macd_bear":mh<0,
        "macd_rising":mh.diff()>0, "macd_falling":mh.diff()<0,
        "adx25":adx>25, "adx_rising":adx.diff()>0,
        "di_bull":pdi>mdi, "di_bear":mdi>pdi,
        "di_strong_bull":(pdi-mdi)>5, "di_strong_bear":(mdi-pdi)>5,
        "volatility":atr_pct>=.7,
        "bb_bull":bbpos>.5, "bb_bear":bbpos<.5,
        "bb_upper":bbpos>=.8, "bb_lower":bbpos<=.2,
        "body_bull":(c>o)&(body>.6), "body_bear":(c<o)&(body>.6),
        # Change / transition features: what changed immediately before entry.
        "rsi_up1":rsi.diff(1)>3, "rsi_down1":rsi.diff(1)<-3,
        "rsi_up3":rsi.diff(3)>5, "rsi_down3":rsi.diff(3)<-5,
        "rsi_cross50_up":(rsi.shift(1)<50)&(rsi>=50), "rsi_cross50_down":(rsi.shift(1)>50)&(rsi<=50),
        "rsi_cross55_up":(rsi.shift(1)<55)&(rsi>=55), "rsi_cross45_down":(rsi.shift(1)>45)&(rsi<=45),
        "macd_cross_up":(mh.shift(1)<=0)&(mh>0), "macd_cross_down":(mh.shift(1)>=0)&(mh<0),
        "macd_accel_up":mh.diff(1)>0, "macd_accel_down":mh.diff(1)<0,
        "adx_up3":adx.diff(3)>3, "adx_down3":adx.diff(3)<-3,
        "adx_cross25_up":(adx.shift(1)<25)&(adx>=25),
        "di_cross_up":(pdi.shift(1)<=mdi.shift(1))&(pdi>mdi),
        "di_cross_down":(mdi.shift(1)<=pdi.shift(1))&(mdi>pdi),
        "di_spread_up3":(pdi-mdi).diff(3)>5, "di_spread_down3":(pdi-mdi).diff(3)<-5,
        "ema20_slope_up3":e20.diff(3)>0, "ema20_slope_down3":e20.diff(3)<0,
        "price20_cross_up":(c.shift(1)<=e20.shift(1))&(c>e20),
        "price20_cross_down":(c.shift(1)>=e20.shift(1))&(c<e20),
        "price200_cross_up":(c.shift(1)<=e200.shift(1))&(c>e200),
        "price200_cross_down":(c.shift(1)>=e200.shift(1))&(c<e200),
        "breakout20_up":c>h.shift(1).rolling(20).max(),
        "breakout20_down":c<l.shift(1).rolling(20).min(),
        "atr_expand":atr/atr.shift(3)>1.15,
        "bb_expand":bbwidth/bbwidth.shift(3)>1.15,
        "body_expand":body-body.shift(3)>.20,
    },index=df.index)
    return f

def load_market(tf,pair):
    df=pd.read_csv(Path("data/market")/tf/f"{pair}.csv")
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms",utc=True,errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def add_labels(df,pair,tf):
    pip=PIP[pair]; n=len(df); close=df.close.to_numpy(); high=df.high.to_numpy(); low=df.low.to_numpy()
    out={f"hit{th}_h{h}":[] for th in LADDER for h in HORIZONS[tf]}
    out.update({f"hit_short{th}_h{h}":[] for th in LADDER for h in HORIZONS[tf]})
    out.update({f"mfe_h{h}":[] for h in HORIZONS[tf]}); out.update({f"mae_h{h}":[] for h in HORIZONS[tf]})
    for i in range(n):
        for h in HORIZONS[tf]:
            hi=high[i+1:min(n,i+h+1)]; lo=low[i+1:min(n,i+h+1)]
            up=(hi-close[i])/pip; dn=(close[i]-lo)/pip
            for th in LADDER:
                out[f"hit{th}_h{h}"].append(bool(len(hi) and np.max(up)>=th))
                out[f"hit_short{th}_h{h}"].append(bool(len(lo) and np.max(dn)>=th))
            out[f"mfe_h{h}"].append(float(np.max(up)) if len(up) else np.nan)
            out[f"mae_h{h}"].append(float(np.max(dn)) if len(dn) else np.nan)
    return pd.DataFrame(out,index=df.index)

def wilson_lower(k,n,z=1.959963984540054):
    if n<=0:return np.nan
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    return center-z*np.sqrt((p*(1-p)/n)+(z*z/(4*n*n)))/den

BULL_BASE=[
    "trend_bull","price20_bull","price200_bull","ema20_rising","ema200_rising","momentum_bull",
    "rsi60","macd_bull","macd_rising","adx25","adx_rising","di_bull","di_strong_bull","volatility",
    "bb_bull","bb_upper","body_bull","rsi_up1","rsi_up3","rsi_cross50_up","rsi_cross55_up",
    "macd_cross_up","macd_accel_up","adx_up3","adx_cross25_up","di_cross_up","di_spread_up3",
    "ema20_slope_up3","price20_cross_up","price200_cross_up","breakout20_up","atr_expand","bb_expand","body_expand"
]
BEAR_BASE=[
    "trend_bear","price20_bear","price200_bear","ema20_falling","ema200_falling","momentum_bear",
    "rsi40","macd_bear","macd_falling","adx25","adx_rising","di_bear","di_strong_bear","volatility",
    "bb_bear","bb_lower","body_bear","rsi_down1","rsi_down3","rsi_cross50_down","rsi_cross45_down",
    "macd_cross_down","macd_accel_down","adx_up3","adx_cross25_up","di_cross_down","di_spread_down3",
    "ema20_slope_down3","price20_cross_down","price200_cross_down","breakout20_down","atr_expand","bb_expand","body_expand"
]

CORE_BULL=[
    ("price20_bull","price200_bull"),("price20_bull","price200_bull","macd_bull"),
    ("price20_bull","price200_bull","volatility"),("price20_bull","price200_bull","macd_bull","volatility"),
    ("price200_bull","macd_bull","volatility"),("price200_bull","macd_bull","di_bull","volatility"),
    ("price200_bull","volatility"),("price200_bull","volatility","bb_bull"),("trend_bull","volatility","body_bull"),
    ("trend_bull","price20_bull","macd_bull","volatility"),("macd_bull","volatility"),("volatility","body_bull"),
    ("macd_bull","di_bull","volatility","bb_bull"),("macd_cross_up","adx_up3","atr_expand"),
    ("price20_cross_up","macd_cross_up","atr_expand"),("breakout20_up","atr_expand","bb_expand"),
    ("ema20_slope_up3","macd_accel_up","di_spread_up3"),("rsi_cross50_up","macd_cross_up","adx_cross25_up")
]
CORE_BEAR=[
    ("price20_bear","price200_bear"),("price20_bear","price200_bear","macd_bear"),
    ("price20_bear","price200_bear","volatility"),("price20_bear","price200_bear","macd_bear","volatility"),
    ("price200_bear","macd_bear","volatility"),("price200_bear","macd_bear","di_bear","volatility"),
    ("price200_bear","volatility"),("price200_bear","volatility","bb_bear"),("trend_bear","volatility","body_bear"),
    ("trend_bear","price20_bear","macd_bear","volatility"),("macd_bear","volatility"),("volatility","body_bear"),
    ("macd_bear","di_bear","volatility","bb_bear"),("macd_cross_down","adx_up3","atr_expand"),
    ("price20_cross_down","macd_cross_down","atr_expand"),("breakout20_down","atr_expand","bb_expand"),
    ("ema20_slope_down3","macd_accel_down","di_spread_down3"),("rsi_cross50_down","macd_cross_down","adx_cross25_up")
]

def mask(g,names):
    m=pd.Series(True,index=g.index)
    for n in names: m &= g[n].fillna(False)
    return m

def candidate_conditions(direction):
    base=BULL_BASE if direction=="bull" else BEAR_BASE
    core=CORE_BULL if direction=="bull" else CORE_BEAR
    seen=set(); out=[]
    for k in (1,2):
        for c in combinations(base,k):
            if c not in seen: seen.add(c); out.append(c)
    for c in core:
        if c not in seen: seen.add(c); out.append(c)
    return out

def main():
    chunks=[]
    for tf in TFS:
        for pair in PAIRS:
            df=load_market(tf,pair); f=build_features(df); lab=add_labels(df,pair,tf)
            z=pd.concat([df[["timestamp","open","high","low","close"]],f,lab],axis=1)
            z["pair"]=pair; z["timeframe"]=tf; chunks.append(z)
    all_df=pd.concat(chunks,ignore_index=True)
    rows=[]
    for tf in TFS:
        tfdf=all_df[all_df.timeframe==tf]
        for direction in ("bull","bear"):
            g=tfdf.copy()
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}" if direction=="bull" else f"hit_short50_h{h}"; base=float(g[target].mean())
                for combo in candidate_conditions(direction):
                    sub=g.loc[mask(g,combo)]
                    if len(sub)<75: continue
                    hit=float(sub[target].mean())
                    rows.append([tf,direction,h,"+".join(combo),len(sub),base,hit,hit-base,hit/base if base else np.nan])
    summary=pd.DataFrame(rows,columns=["timeframe","direction","horizon_bars","conditions","samples","base_rate","hit50_rate","rate_diff","lift"])
    Path("reports").mkdir(exist_ok=True); summary.to_csv("reports/50pip_direct_entry_conditions.csv",index=False,float_format="%.8f")

    scored=[]
    for tf in TFS:
        tfdf=all_df[all_df.timeframe==tf]
        disc=tfdf[tfdf.timestamp.dt.year<=2024]; val=tfdf[tfdf.timestamp.dt.year==2025]; oos=tfdf[tfdf.timestamp.dt.year>=2026]
        for direction in ("bull","bear"):
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}"; base=float(disc[target].mean()); candidates=[]
                for combo in candidate_conditions(direction):
                    sub=disc.loc[mask(disc,combo)]
                    if len(sub)<150: continue
                    hit=float(sub[target].mean()); lift=hit/base if base else np.nan
                    if np.isfinite(lift) and lift>=1.10: candidates.append((combo,len(sub),hit,lift))
                candidates=sorted(candidates,key=lambda x:(x[3],x[1]),reverse=True)[:60]
                for combo,dn,dh,dl in candidates:
                    vals=[tf,direction,h,"+".join(combo),dn,dh,dl]
                    for name,g in (("validation",val),("oos",oos),("all",tfdf)):
                        sub=g.loc[mask(g,combo)]; n=len(sub); hit=float(sub[target].mean()) if n else np.nan; b=float(g[target].mean()) if len(g) else np.nan
                        vals += [n,hit,hit/b if b else np.nan,wilson_lower(int(sub[target].sum()),n)]
                    scored.append(vals)
    cols=["timeframe","direction","horizon_bars","conditions","discovery_samples","discovery_hit_rate","discovery_lift",
          "validation_samples","validation_hit_rate","validation_lift","validation_wilson95_lower",
          "oos_samples","oos_hit_rate","oos_lift","oos_wilson95_lower","all_samples","all_hit_rate","all_lift","all_wilson95_lower"]
    oos=pd.DataFrame(scored,columns=cols); oos.to_csv("reports/50pip_direct_entry_oos.csv",index=False,float_format="%.8f")

    robust=[]
    for r in oos.itertuples(index=False):
        if r.oos_samples<50: continue
        g=all_df[(all_df.timeframe==r.timeframe)&(all_df.timestamp.dt.year>=2026)]
        sub=g.loc[mask(g,r.conditions.split("+"))]; target=f"hit50_h{r.horizon_bars}" if r.direction=="bull" else f"hit_short50_h{r.horizon_bars}"
        base=float(g[target].mean()); pairs=[]
        for pair,x in sub.groupby("pair"):
            if len(x)>=5: pairs.append((pair,len(x),float(x[target].mean())))
        if pairs:
            robust.append([r.timeframe,r.direction,r.horizon_bars,r.conditions,r.discovery_samples,r.validation_samples,r.oos_samples,
                           r.discovery_lift,r.validation_lift,r.oos_lift,r.oos_wilson95_lower,len(pairs),
                           sum(x[2]>=base for x in pairs),np.mean([x[2] for x in pairs]),min(x[2] for x in pairs)])
    rob=pd.DataFrame(robust,columns=["timeframe","direction","horizon_bars","conditions","discovery_samples","validation_samples","oos_samples",
                                     "discovery_lift","validation_lift","oos_lift","oos_wilson95_lower","oos_pairs_n_ge5",
                                     "oos_pairs_above_oos_base","oos_pair_hit_mean","oos_min_pair_hit"])
    rob.to_csv("reports/50pip_direct_entry_robustness.csv",index=False,float_format="%.8f")
    print("direct_entry_rows",len(summary),"oos_candidates",len(oos),"robustness_rows",len(rob))
    print(rob.sort_values(["oos_wilson95_lower","oos_lift","oos_samples"],ascending=False).head(40).to_string(index=False))

if __name__=="__main__": main()
