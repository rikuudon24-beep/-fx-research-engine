#!/usr/bin/env python3
"""Direct entry research: test long-entry conditions on every eligible H4/D1 candle.

Success is defined strictly as reaching +50 pips or more after the signal.
100/150/200/300 pips are extension metrics, not alternative success criteria.
No time-based exit is used here; horizons are measurement windows only.
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

PAIRS=["usdjpy","eurjpy","gbpjpy","audjpy","eurusd","gbpusd","audusd","nzdusd","usdcad","usdchf","audnzd","eurgbp"]
TFS=["h4","d1"]
PIP={p:0.01 if "jpy" in p else 0.0001 for p in PAIRS}
HORIZONS={"h4":[1,3,6,12,24],"d1":[1,3,5,10,20]}
LADDER=[50,60,70,80,100,120,150,200,300]
SPLITS={"discovery_end":"2024-12-31","validation_start":"2025-01-01","validation_end":"2025-12-31","oos_start":"2026-01-01"}

def rma(s,n):
    return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def build_features(df):
    c,h,l,o=df.close,df.high,df.low,df.open
    e20=c.ewm(span=20,adjust=False).mean()
    e50=c.ewm(span=50,adjust=False).mean()
    e200=c.ewm(span=200,adjust=False).mean()
    d=c.diff()
    g=d.clip(lower=0); loss=-d.clip(upper=0)
    rsi=100-100/(1+rma(g,14)/rma(loss,14).replace(0,np.nan))
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    mh=macd-macd.ewm(span=9,adjust=False).mean()
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=rma(tr,14)
    up=h.diff(); dn=-l.diff()
    plus=pd.Series(np.where((up>dn)&(up>0),up,0),index=df.index)
    minus=pd.Series(np.where((dn>up)&(dn>0),dn,0),index=df.index)
    pdi=100*rma(plus,14)/atr
    mdi=100*rma(minus,14)/atr
    adx=rma(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan),14)
    atr_pct=atr.rolling(100).rank(pct=True)
    mid=c.rolling(20).mean(); sd=c.rolling(20).std()
    bbpos=(c-(mid-2*sd))/(4*sd).replace(0,np.nan)
    body=(c-o).abs()/(h-l).replace(0,np.nan)
    # Slope/state features use only current and prior candles.
    e20_slope=e20.diff()
    e200_slope=e200.diff()
    mh_slope=mh.diff()
    adx_slope=adx.diff()
    di_spread=pdi-mdi
    return pd.DataFrame({
        "trend":(e20>e50)&(e50>e200),
        "price20":c>e20,
        "price200":c>e200,
        "ema20_rising":e20_slope>0,
        "ema200_rising":e200_slope>0,
        "momentum":(rsi>=45)&(rsi<70),
        "rsi55":rsi>=55,
        "rsi60":rsi>=60,
        "rsi_lt45":rsi<45,
        "macd":mh>0,
        "macd_rising":mh_slope>0,
        "adx25":adx>25,
        "adx_rising":adx_slope>0,
        "di":pdi>mdi,
        "di_strong":di_spread>5,
        "volatility":atr_pct>=.7,
        "bb":bbpos>.5,
        "bb_upper":bbpos>=.8,
        "body":body>.6,
    },index=df.index)

def load_market(tf,pair):
    path=Path("data/market")/tf/f"{pair}.csv"
    df=pd.read_csv(path)
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms",utc=True,errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def add_labels(df,pair,tf):
    pip=PIP[pair]
    max_h=max(HORIZONS[tf])
    close=df.close.to_numpy()
    high=df.high.to_numpy()
    low=df.low.to_numpy()
    n=len(df)
    out=[]
    for i in range(n):
        j=min(n,i+max_h+1)
        if i+1>=j:
            out.append({})
            continue
        fh=high[i+1:j]; fl=low[i+1:j]
        up=(np.maximum.accumulate(fh)-close[i])/pip
        down=(close[i]-np.minimum.accumulate(fl))/pip
        row={}
        for th in LADDER:
            row[f"mfe_{th}"]=float(np.max(up))>=th
            row[f"mfe_max"] = float(np.max(up))
        for h in HORIZONS[tf]:
            hh=high[i+1:min(n,i+h+1)]
            if len(hh):
                row[f"hit50_h{h}"]=float(np.max(np.maximum.accumulate(hh)-close[i])/pip)>=50
                row[f"mfe_h{h}"]=float(np.max((hh-close[i])/pip))
                row[f"mae_h{h}"]=float(np.max((close[i]-low[i+1:min(n,i+h+1)])/pip))
            else:
                row[f"hit50_h{h}"]=False
                row[f"mfe_h{h}"]=np.nan
                row[f"mae_h{h}"]=np.nan
        out.append(row)
    return pd.DataFrame(out,index=df.index)

def condition_map():
    return [
        "trend","price20","price200","ema20_rising","ema200_rising",
        "momentum","rsi55","rsi60","macd","macd_rising",
        "adx25","adx_rising","di","di_strong","volatility","bb","bb_upper","body"
    ]

def condition_mask(df,names):
    m=pd.Series(True,index=df.index)
    for n in names:
        m &= df[n].fillna(False)
    return m

def wilson_lower(k,n,z=1.959963984540054):
    if n<=0: return np.nan
    p=k/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt((p*(1-p)/n)+(z*z/(4*n*n)))/den
    return center-half

def main():
    chunks=[]
    for tf in TFS:
        for pair in PAIRS:
            df=load_market(tf,pair)
            f=build_features(df)
            labels=add_labels(df,pair,tf)
            z=pd.concat([df[["timestamp","open","high","low","close"]],f,labels],axis=1)
            z["pair"]=pair; z["timeframe"]=tf
            chunks.append(z)
    all_df=pd.concat(chunks,ignore_index=True)
    conds=condition_map()
    rows=[]
    # Analyze every eligible candle, not only candles that already became 50pip events.
    for tf in TFS:
        for h in HORIZONS[tf]:
            target=f"hit50_h{h}"
            for k in range(1,5):
                for combo in combinations(conds,k):
                    m=condition_mask(all_df[all_df.timeframe==tf],combo)
                    sub=all_df[all_df.timeframe==tf].loc[m].copy()
                    if len(sub)<75: continue
                    hit=float(sub[target].mean())
                    base=float(all_df.loc[all_df.timeframe==tf,target].mean())
                    rows.append([tf,h,"+".join(combo),len(sub),base,hit,hit-base,hit/base if base else np.nan])
    summary=pd.DataFrame(rows,columns=["timeframe","horizon_bars","conditions","samples","base_rate","hit50_rate","rate_diff","lift"])
    Path("reports").mkdir(exist_ok=True)
    summary.to_csv("reports/50pip_direct_entry_conditions.csv",index=False,float_format="%.8f")

    # Time-split validation of candidates. Candidate discovery requires >=150 samples and lift >=1.10.
    scored=[]
    for tf in TFS:
        tfdf=all_df[all_df.timeframe==tf].copy()
        for h in HORIZONS[tf]:
            target=f"hit50_h{h}"
            disc=tfdf[tfdf.timestamp.dt.year<=2024]
            candidates=[]
            for k in range(1,5):
                for combo in combinations(conds,k):
                    m=condition_mask(disc,combo)
                    sub=disc.loc[m]
                    if len(sub)<150: continue
                    hit=float(sub[target].mean())
                    base=float(disc[target].mean())
                    lift=hit/base if base else np.nan
                    if np.isfinite(lift) and lift>=1.10:
                        candidates.append((combo,len(sub),hit,lift))
            candidates=sorted(candidates,key=lambda x:(x[3],x[1]),reverse=True)[:100]
            for combo,dn,dh,dl in candidates:
                vals=[tf,h,"+".join(combo),dn,dh,dl]
                for label,mask in [
                    ("validation",(tfdf.timestamp.dt.year==2025)),
                    ("oos",(tfdf.timestamp.dt.year>=2026)),
                    ("all",(tfdf.timestamp.dt.year>=2021))
                ]:
                    g=tfdf.loc[mask]
                    m=condition_mask(g,combo)
                    sub=g.loc[m]
                    n=len(sub); hit=float(sub[target].mean()) if n else np.nan
                    base=float(g[target].mean()) if len(g) else np.nan
                    vals += [n,hit,hit/base if base else np.nan,wilson_lower(int(sub[target].sum()),n)]
                scored.append(vals)
    cols=["timeframe","horizon_bars","conditions","discovery_samples","discovery_hit_rate","discovery_lift",
          "validation_samples","validation_hit_rate","validation_lift","validation_wilson95_lower",
          "oos_samples","oos_hit_rate","oos_lift","oos_wilson95_lower",
          "all_samples","all_hit_rate","all_lift","all_wilson95_lower"]
    oos=pd.DataFrame(scored,columns=cols)
    oos.to_csv("reports/50pip_direct_entry_oos.csv",index=False,float_format="%.8f")

    # Pair robustness for OOS candidates.
    robust=[]
    for r in oos.itertuples(index=False):
        if r.oos_samples<50: continue
        tfdf=all_df[(all_df.timeframe==r.timeframe)&(all_df.timestamp.dt.year>=2026)].copy()
        combo=r.conditions.split("+")
        m=condition_mask(tfdf,combo)
        sub=tfdf.loc[m]
        pair_rows=[]
        for pair,g in sub.groupby("pair"):
            if len(g)<5: continue
            hit=float(g[f"hit50_h{r.horizon_bars}"].mean())
            pair_rows.append((pair,len(g),hit))
        if not pair_rows: continue
        robust.append([
            r.timeframe,r.horizon_bars,r.conditions,r.discovery_samples,r.validation_samples,r.oos_samples,
            r.discovery_lift,r.validation_lift,r.oos_lift,r.oos_wilson95_lower,
            len(pair_rows),sum(x[2]>=float(all_df[(all_df.timeframe==r.timeframe)&(all_df.timestamp.dt.year>=2026)][f"hit50_h{r.horizon_bars}"].mean()) for x in pair_rows),
            np.mean([x[2] for x in pair_rows]),min(x[2] for x in pair_rows)
        ])
    rob=pd.DataFrame(robust,columns=[
        "timeframe","horizon_bars","conditions","discovery_samples","validation_samples","oos_samples",
        "discovery_lift","validation_lift","oos_lift","oos_wilson95_lower",
        "oos_pairs_n_ge5","oos_pairs_above_oos_base","oos_pair_hit_mean","oos_min_pair_hit"
    ])
    rob.to_csv("reports/50pip_direct_entry_robustness.csv",index=False,float_format="%.8f")
    print("direct_entry_rows",len(summary),"oos_candidates",len(oos),"robustness_rows",len(rob))
    print(rob.sort_values(["oos_wilson95_lower","oos_lift","oos_samples"],ascending=False).head(40).to_string(index=False))

if __name__=="__main__":
    main()
