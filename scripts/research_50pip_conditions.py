#!/usr/bin/env python3
"""Research directional 50/100/150/200/300 pip events and indicator conditions."""
from pathlib import Path
import numpy as np
import pandas as pd

PAIRS=["usdjpy","eurjpy","gbpjpy","audjpy","eurusd","gbpusd","audusd","nzdusd","usdcad","usdchf","audnzd","eurgbp"]
TFS=["h4","d1"]; PIP={p:0.01 if "jpy" in p else 0.0001 for p in PAIRS}
HORIZONS={"h4":[1,3,6,12,24],"d1":[1,3,5,10,20]}; THRESHOLDS=[50,100,150,200,300]

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def features(df):
 c,h,l,o=df.close,df.high,df.low,df.open
 e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean()
 d=c.diff(); g=d.clip(lower=0); loss=-d.clip(upper=0); rsi=100-100/(1+rma(g,14)/rma(loss,14).replace(0,np.nan))
 macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); mh=macd-macd.ewm(span=9,adjust=False).mean()
 tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=rma(tr,14)
 up=h.diff(); dn=-l.diff(); plus=pd.Series(np.where((up>dn)&(up>0),up,0),index=df.index); minus=pd.Series(np.where((dn>up)&(dn>0),dn,0),index=df.index)
 pdi=100*rma(plus,14)/atr; mdi=100*rma(minus,14)/atr; adx=rma(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan),14)
 mid=c.rolling(20).mean(); sd=c.rolling(20).std()
 return pd.DataFrame({"ema_stack_bull":(e20>e50)&(e50>e200),"ema_stack_bear":(e20<e50)&(e50<e200),
 "price_above_ema20":c>e20,"price_above_200":c>e200,"rsi14":rsi,"macd_hist":mh,"adx14":adx,"pdi":pdi,"mdi":mdi,
 "atr_pct":atr.rolling(100).rank(pct=True),"bbpos":(c-(mid-2*sd))/(4*sd).replace(0,np.nan),
 "body_ratio":(c-o).abs()/(h-l).replace(0,np.nan)},index=df.index)

def future(s,n,kind):
 shifted=s.shift(-1)
 r=shifted.rolling(n,min_periods=n)
 x=r.max() if kind=="max" else r.min()
 return x.shift(-(n-1))

def targets(df,pair,tf):
 out=pd.DataFrame(index=df.index); pip=PIP[pair]
 for n in HORIZONS[tf]:
  hi=future(df.high,n,"max"); lo=future(df.low,n,"min"); e=df.close
  up=(hi-e)/pip; dn=(e-lo)/pip
  for th in THRESHOLDS: out[f"bull_{th}_h{n}"]=up>=th; out[f"bear_{th}_h{n}"]=dn>=th
 return out

CONDS={
"ema_stack_bull":lambda x:x.ema_stack_bull,"ema_stack_bear":lambda x:x.ema_stack_bear,
"price_above_ema20":lambda x:x.price_above_ema20,"price_above_200":lambda x:x.price_above_200,
"rsi_lt30":lambda x:x.rsi14<30,"rsi_30_45":lambda x:(x.rsi14>=30)&(x.rsi14<45),
"rsi_45_55":lambda x:(x.rsi14>=45)&(x.rsi14<55),"rsi_55_70":lambda x:(x.rsi14>=55)&(x.rsi14<70),
"rsi_gt70":lambda x:x.rsi14>70,"macd_hist_pos":lambda x:x.macd_hist>0,"adx_gt25":lambda x:x.adx14>25,
"pdi_gt_mdi":lambda x:x.pdi>x.mdi,"mdi_gt_pdi":lambda x:x.mdi>x.pdi,
"atr_high":lambda x:x.atr_pct>=.7,"atr_low":lambda x:x.atr_pct<=.3,
"bb_below_mid":lambda x:x.bbpos<.5,"bb_above_mid":lambda x:x.bbpos>.5,"body_ratio_gt60":lambda x:x.body_ratio>.6}

def main():
 ds=[]
 for tf in TFS:
  for pair in PAIRS:
   df=pd.read_csv(Path("data/market")/tf/f"{pair}.csv").sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
   z=pd.concat([features(df),targets(df,pair,tf)],axis=1); z["pair"]=pair; z["timeframe"]=tf; ds.append(z)
 data=pd.concat(ds,ignore_index=True); rows=[]; pair_rows=[]
 for tf in TFS:
  d=data[data.timeframe==tf]
  for direction in ("bull","bear"):
   for n in HORIZONS[tf]:
    for th in THRESHOLDS:
     target=f"{direction}_{th}_h{n}"; q=d.drop(columns=["timeframe"]).dropna(subset=[target]); base=float(q[target].mean())
     for name,fn in CONDS.items():
      m=fn(q).fillna(False); samples=int(m.sum())
      if samples<100: continue
      rate=float(q.loc[m,target].mean())
      rows.append([tf,direction,th,n,name,samples,base,rate,rate-base,rate/base if base else np.nan])
      for pair,g in q.loc[m].groupby("pair"):
       if len(g)<50: continue
       pb=float(q.loc[q.pair==pair,target].mean()); pr=float(g[target].mean())
       pair_rows.append([pair,tf,direction,th,n,name,len(g),pb,pr,pr-pb,pr/pb if pb else np.nan])
 summary=pd.DataFrame(rows,columns=["timeframe","direction","threshold_pips","horizon_bars","condition","samples","base_rate","hit_rate","rate_diff","lift"])
 ps=pd.DataFrame(pair_rows,columns=["pair","timeframe","direction","threshold_pips","horizon_bars","condition","samples","base_rate","hit_rate","rate_diff","lift"])
 Path("reports").mkdir(exist_ok=True); summary.to_csv("reports/50pip_condition_summary.csv",index=False,float_format="%.8f")
 ps.to_csv("reports/50pip_condition_pair_summary.csv",index=False,float_format="%.8f"); data.to_csv("reports/50pip_feature_dataset.csv",index=False)
 print("rows",len(summary),"pair_rows",len(ps)); print(summary.sort_values("lift",ascending=False).head(50).to_string(index=False))
if __name__=="__main__": main()
