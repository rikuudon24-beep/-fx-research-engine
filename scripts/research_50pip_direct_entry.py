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
    plus=pd.Series(np.where((up>dn)&(up>0),up,0),index=df.index); minus=pd.Series(np.where((dn>up)&(dn>0),dn,0),index=df.index)
    pdi=100*rma(plus,14)/atr; mdi=100*rma(minus,14)/atr; adx=rma(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan),14)
    atr_pct=atr.rolling(100).rank(pct=True)
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); bbpos=(c-(mid-2*sd))/(4*sd).replace(0,np.nan)
    body=(c-o).abs()/(h-l).replace(0,np.nan)
    return pd.DataFrame({
        "trend":(e20>e50)&(e50>e200),"price20":c>e20,"price200":c>e200,
        "ema20_rising":e20.diff()>0,"ema200_rising":e200.diff()>0,
        "momentum":(rsi>=45)&(rsi<70),"rsi55":rsi>=55,"rsi60":rsi>=60,"rsi_lt45":rsi<45,
        "macd":mh>0,"macd_rising":mh.diff()>0,"adx25":adx>25,"adx_rising":adx.diff()>0,
        "di":pdi>mdi,"di_strong":(pdi-mdi)>5,"volatility":atr_pct>=.7,
        "bb":bbpos>.5,"bb_upper":bbpos>=.8,"body":body>.6,
    },index=df.index)

def load_market(tf,pair):
    df=pd.read_csv(Path("data/market")/tf/f"{pair}.csv")
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms",utc=True,errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def add_labels(df,pair,tf):
    pip=PIP[pair]; n=len(df); close=df.close.to_numpy(); high=df.high.to_numpy(); low=df.low.to_numpy()
    out={f"hit{th}_h{h}":[] for th in LADDER for h in HORIZONS[tf]}
    out.update({f"mfe_h{h}":[] for h in HORIZONS[tf]}); out.update({f"mae_h{h}":[] for h in HORIZONS[tf]})
    for i in range(n):
        for h in HORIZONS[tf]:
            hi=high[i+1:min(n,i+h+1)]; lo=low[i+1:min(n,i+h+1)]
            up=(hi-close[i])/pip; dn=(close[i]-lo)/pip
            for th in LADDER: out[f"hit{th}_h{h}"].append(bool(len(hi) and np.max(up)>=th))
            out[f"mfe_h{h}"].append(float(np.max(up)) if len(up) else np.nan)
            out[f"mae_h{h}"].append(float(np.max(dn)) if len(dn) else np.nan)
    return pd.DataFrame(out,index=df.index)

def wilson_lower(k,n,z=1.959963984540054):
    if n<=0:return np.nan
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    return center-z*np.sqrt((p*(1-p)/n)+(z*z/(4*n*n)))/den

def mask(g,names,direction):
    m=pd.Series(True,index=g.index)
    for n in names:
        m &= g[n].fillna(False) if direction=="bull" else (~g[n].fillna(False))
    return m

# Deliberately bounded: singles + pairs + selected 3/4-condition families.
BASE=["trend","price20","price200","ema20_rising","ema200_rising","momentum","rsi55","rsi60","rsi_lt45","macd","macd_rising","adx25","adx_rising","di","di_strong","volatility","bb","bb_upper","body"]
CORE=[
    ("price20","price200"),("price20","price200","macd"),("price20","price200","volatility"),
    ("price20","price200","macd","volatility"),("price200","macd","volatility"),
    ("price200","macd","di","volatility"),("price200","volatility"),("price200","volatility","bb"),
    ("trend","volatility","body"),("trend","price20","macd","volatility"),
    ("macd","volatility"),("volatility","body"),("macd","di","volatility","bb"),
]

def candidate_conditions():
    seen=set(); out=[]
    for k in (1,2):
        for c in combinations(BASE,k):
            if c not in seen: seen.add(c); out.append(c)
    for c in CORE:
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
    combos=candidate_conditions(); rows=[]
    for tf in TFS:
        tfdf=all_df[all_df.timeframe==tf]
        for direction in ("bull","bear"):
            g=tfdf.copy()
            for h in HORIZONS[tf]:
                target=f"hit50_h{h}"; base=float(g[target].mean())
                for combo in combos:
                    sub=g.loc[mask(g,combo,direction)]
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
                for combo in combos:
                    sub=disc.loc[mask(disc,combo,direction)]
                    if len(sub)<150: continue
                    hit=float(sub[target].mean()); lift=hit/base if base else np.nan
                    if np.isfinite(lift) and lift>=1.10: candidates.append((combo,len(sub),hit,lift))
                candidates=sorted(candidates,key=lambda x:(x[3],x[1]),reverse=True)[:60]
                for combo,dn,dh,dl in candidates:
                    vals=[tf,direction,h,"+".join(combo),dn,dh,dl]
                    for name,g in (("validation",val),("oos",oos),("all",tfdf)):
                        sub=g.loc[mask(g,combo,direction)]; n=len(sub); hit=float(sub[target].mean()) if n else np.nan; b=float(g[target].mean()) if len(g) else np.nan
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
        sub=g.loc[mask(g,r.conditions.split("+"),r.direction)]; target=f"hit50_h{r.horizon_bars}"
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
