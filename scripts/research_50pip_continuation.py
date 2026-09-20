#!/usr/bin/env python3
"""Research continuation from 50pip events with directional indicator combinations and time OOS validation."""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

PAIRS=["usdjpy","eurjpy","gbpjpy","audjpy","eurusd","gbpusd","audusd","nzdusd","usdcad","usdchf","audnzd","eurgbp"]
TFS=["h4","d1"]
PIP={p:0.01 if "jpy" in p else 0.0001 for p in PAIRS}
LADDER=[50,100,150,200,300]
SPLITS={"discovery_end":"2024-12-31","validation_start":"2025-01-01","validation_end":"2025-12-31","oos_start":"2026-01-01"}

def rma(s,n):
    return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def features(df):
    c,h,l,o=df.close,df.high,df.low,df.open
    e20=c.ewm(span=20,adjust=False).mean()
    e50=c.ewm(span=50,adjust=False).mean()
    e200=c.ewm(span=200,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0); loss=-d.clip(upper=0)
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
    return pd.DataFrame({
        "ema_bull":(e20>e50)&(e50>e200),
        "ema_bear":(e20<e50)&(e50<e200),
        "price_ema20_bull":c>e20,
        "price_ema20_bear":c<e20,
        "price_200_bull":c>e200,
        "price_200_bear":c<e200,
        "rsi30_45":(rsi>=30)&(rsi<45),
        "rsi45_55":(rsi>=45)&(rsi<55),
        "rsi55_70":(rsi>=55)&(rsi<70),
        "rsi_lt30":rsi<30,
        "rsi_gt70":rsi>70,
        "macd_bull":mh>0,
        "macd_bear":mh<0,
        "adx25":adx>25,
        "di_bull":pdi>mdi,
        "di_bear":mdi>pdi,
        "atr_high":atr_pct>=.7,
        "atr_low":atr_pct<=.3,
        "bb_above_mid":bbpos>.5,
        "bb_below_mid":bbpos<.5,
        "body_gt60":body>.6,
    }, index=df.index)

def load_events():
    e=pd.read_csv("reports/50pip_events.csv")
    e["event_timestamp"]=pd.to_datetime(e["event_timestamp"],utc=True,errors="coerce")
    e["hit_timestamp"]=pd.to_datetime(e["hit_timestamp"],utc=True,errors="coerce")
    e=e.dropna(subset=["event_timestamp","hit_timestamp"]).copy()
    return e

def event_features(events):
    chunks=[]
    for tf in TFS:
        for pair in PAIRS:
            path=Path("data/market")/tf/f"{pair}.csv"
            df=pd.read_csv(path)
            df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms",utc=True,errors="coerce")
            df=df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
            f=features(df)
            z=pd.concat([df[["timestamp","close"]],f],axis=1)
            z["pair"]=pair; z["timeframe"]=tf
            chunks.append(z)
    allf=pd.concat(chunks,ignore_index=True)
    e=events.copy()
    e["pair"]=e.pair.astype(str); e["timeframe"]=e.timeframe.astype(str)
    e["event_timestamp"]=pd.to_datetime(e["event_timestamp"],utc=True,errors="coerce")
    allf["timestamp"]=pd.to_datetime(allf["timestamp"],utc=True,errors="coerce")
    out=e.merge(
        allf,
        left_on=["pair","timeframe","event_timestamp"],
        right_on=["pair","timeframe","timestamp"],
        how="left",
        validate="many_to_one",
    )
    return out

def directional_conditions(direction):
    if direction=="bull":
        return {
            "trend":lambda d:d.ema_bull,
            "price20":lambda d:d.price_ema20_bull,
            "price200":lambda d:d.price_200_bull,
            "momentum":lambda d:(d.rsi45_55|d.rsi55_70),
            "rsi_extreme":lambda d:d.rsi_lt30,
            "macd":lambda d:d.macd_bull,
            "adx":lambda d:d.adx25,
            "di":lambda d:d.di_bull,
            "volatility":lambda d:d.atr_high,
            "bb":lambda d:d.bb_above_mid,
            "body":lambda d:d.body_gt60,
        }
    return {
        "trend":lambda d:d.ema_bear,
        "price20":lambda d:d.price_ema20_bear,
        "price200":lambda d:d.price_200_bear,
        "momentum":lambda d:(d.rsi30_45|d.rsi45_55),
        "rsi_extreme":lambda d:d.rsi_gt70,
        "macd":lambda d:d.macd_bear,
        "adx":lambda d:d.adx25,
        "di":lambda d:d.di_bear,
        "volatility":lambda d:d.atr_high,
        "bb":lambda d:d.bb_below_mid,
        "body":lambda d:d.body_gt60,
    }

def evaluate(group, names, conds, target):
    m=pd.Series(True,index=group.index)
    for n in names: m &= conds[n](group).fillna(False)
    n=int(m.sum())
    if n<50: return None
    hit=float(group.loc[m,target].mean())
    return n,hit

def main():
    all_events=load_events()
    base_events=all_events[all_events.threshold_pips==50].copy()
    events=event_features(base_events)
    missing=int(events[[c for c in ["rsi30_45","adx25","di_bull"] if c in events]].isna().all(axis=1).sum())
    if missing:
        print(f"WARNING feature rows missing={missing}")
    rows=[]
    for tf in TFS:
        for direction in ("bull","bear"):
            d=events[(events.timeframe==tf)&(events.direction==direction)].copy()
            for th in LADDER[1:]:
                target=f"hit_{th}"
                # Conditional continuation: among events that first reached 50p, did the same event also reach th?
                d[target]=False
                # event file has one row per threshold, so reconstruct by event key
                key=["pair","timeframe","direction","event_timestamp"]
                hits=all_events[all_events.threshold_pips==th][key].drop_duplicates().assign(**{target:True})
                d=d.merge(hits,on=key,how="left",suffixes=("","_y"))
                d[target]=d[target+"_y"].fillna(False) if target+"_y" in d else d[target]
                if target+"_y" in d: d.drop(columns=[target+"_y"],inplace=True)
                base=float(d[target].mean())
                conds=directional_conditions(direction)
                # singles and 2/3/4-way combinations, limited to interpretable directional features
                names=list(conds)
                combos=[]
                for k in range(1,5): combos.extend(combinations(names,k))
                for combo in combos:
                    r=evaluate(d,combo,conds,target)
                    if r is None: continue
                    n,hit=r
                    rows.append([tf,direction,th,"+".join(combo),n,base,hit,hit-base,hit/base if base else np.nan])
    res=pd.DataFrame(rows,columns=["timeframe","direction","target_pips","conditions","samples","base_rate","hit_rate","rate_diff","lift"])
    Path("reports").mkdir(exist_ok=True)
    res.to_csv("reports/50pip_continuation_combinations.csv",index=False,float_format="%.8f")
    # Rank only combinations discovered on 2021-2024, then score on 2025 and 2026 OOS.
    scored=[]
    for tf in TFS:
        for direction in ("bull","bear"):
            base=events[(events.timeframe==tf)&(events.direction==direction)&(events.threshold_pips==50)].copy()
            base["date"]=base.event_timestamp.dt.date
            for th in LADDER[1:]:
                key=["pair","timeframe","direction","event_timestamp"]
                hits=all_events[all_events.threshold_pips==th][key].drop_duplicates().assign(hit=True)
                d=base.merge(hits,on=key,how="left"); d["hit"]=d.hit.fillna(False)
                conds=directional_conditions(direction)
                names=list(conds)
                candidates=[]
                disc=d[d.event_timestamp.dt.year<=2024]
                for k in range(1,5):
                    for combo in combinations(names,k):
                        r=evaluate(disc,combo,conds,"hit")
                        if r is None: continue
                        n,hit=r
                        if n<75: continue
                        # Keep discovery candidates with positive lift; no ranking claim is made in output.
                        lift=hit/(float(disc.hit.mean()) or np.nan)
                        if np.isfinite(lift) and lift>=1.20:
                            candidates.append((combo,n,hit,lift))
                candidates=sorted(candidates,key=lambda x:(x[3],x[1]),reverse=True)[:50]
                for combo,disc_n,disc_hit,disc_lift in candidates:
                    row=[tf,direction,th,"+".join(combo),disc_n,disc_hit,disc_lift]
                    for label,mask in [
                        ("validation",(d.event_timestamp.dt.year==2025)),
                        ("oos",(d.event_timestamp.dt.year>=2026)),
                        ("all",(d.event_timestamp.dt.year>=2021))]:
                        g=d[mask]
                        rr=evaluate(g,combo,conds,"hit")
                        row.extend([0,np.nan,np.nan] if rr is None else [rr[0],rr[1],rr[1]/(float(g.hit.mean()) or np.nan)])
                    scored.append(row)
    cols=["timeframe","direction","target_pips","conditions","discovery_samples","discovery_hit_rate","discovery_lift",
          "validation_samples","validation_hit_rate","validation_lift","oos_samples","oos_hit_rate","oos_lift",
          "all_samples","all_hit_rate","all_lift"]
    sc=pd.DataFrame(scored,columns=cols)
    sc.to_csv("reports/50pip_continuation_oos.csv",index=False,float_format="%.8f")
    print("continuation_rows",len(res),"oos_candidates",len(sc))
    if not res.empty:
        print("target_base_rates")
        print(res.groupby(["timeframe","direction","target_pips"])["base_rate"].first().to_string())
    print(sc.head(50).to_string(index=False))

if __name__=="__main__":
    main()
