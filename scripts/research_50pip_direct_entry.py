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

def rolling_slope(s, n):
    x=np.arange(n,dtype=float)
    xm=x.mean()
    den=((x-xm)**2).sum()
    return s.rolling(n).apply(lambda y: float(np.dot(x-xm, y-y.mean())/den) if np.isfinite(y).all() else np.nan, raw=True)

def rci(s,n=14):
    ranks_t=np.arange(1,n+1,dtype=float)
    def calc(y):
        if not np.isfinite(y).all(): return np.nan
        ranks=pd.Series(y).rank(method="average").to_numpy()
        d=ranks-ranks_t
        return 100.0*(1.0-6.0*np.sum(d*d)/(n*(n*n-1)))
    return s.rolling(n).apply(calc,raw=True)

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

    # Additional chart/indicator families from the user's charting toolkit.
    # All are causal: rolling swing levels use shift(1), and Ichimoku cloud
    # values at time t are the forward-shifted values that were already known.
    sma20=c.rolling(20).mean(); sma50=c.rolling(50).mean(); sma200=c.rolling(200).mean()
    bb_std=sd
    bb_upper_band=mid+2*bb_std; bb_lower_band=mid-2*bb_std
    # Super Bollinger-style expansion/position proxies.
    bb3_upper=mid+3*bb_std; bb3_lower=mid-3*bb_std
    bb_width=bbwidth
    bb_squeeze=bb_width < bb_width.rolling(100).quantile(.20)

    tenkan=(h.rolling(9).max()+l.rolling(9).min())/2
    kijun=(h.rolling(26).max()+l.rolling(26).min())/2
    span_a=((tenkan+kijun)/2).shift(26)
    span_b=((h.rolling(52).max()+l.rolling(52).min())/2).shift(26)
    cloud_top=pd.concat([span_a,span_b],axis=1).max(axis=1)
    cloud_bottom=pd.concat([span_a,span_b],axis=1).min(axis=1)

    stoch_low=l.rolling(14).min(); stoch_high=h.rolling(14).max()
    stoch_k=100*(c-stoch_low)/(stoch_high-stoch_low).replace(0,np.nan)
    stoch_d=stoch_k.rolling(3).mean()

    rci14=rci(c,14)

    # Fibonacci retracement of the preceding 50 completed candles.
    fib_hi=h.shift(1).rolling(50).max(); fib_lo=l.shift(1).rolling(50).min()
    fib_rng=(fib_hi-fib_lo).replace(0,np.nan)
    fib236=fib_hi-fib_rng*.236; fib382=fib_hi-fib_rng*.382
    fib500=fib_hi-fib_rng*.500; fib618=fib_hi-fib_rng*.618; fib786=fib_hi-fib_rng*.786
    fib_pos=(c-fib_lo)/fib_rng

    # Fibonacci time-zone / fan / arc proxies. These are intentionally
    # deterministic time/price geometry derived from prior completed swings.
    swing_age=np.arange(len(df),dtype=float)
    fib_time_5=(np.arange(len(df)) % 5 == 0)
    fib_time_8=(np.arange(len(df)) % 8 == 0)
    fib_time_13=(np.arange(len(df)) % 13 == 0)
    fib_time_zone=fib_time_5|fib_time_8|fib_time_13
    fan_base=(fib_hi-fib_lo).replace(0,np.nan)
    fan_ratio=(c-fib_lo)/fan_base
    fib_fan_up=fan_ratio > .618
    fib_fan_down=fan_ratio < .382
    fib_arc_mid=((fib_hi+fib_lo)/2)
    fib_arc_bull=c>fib_arc_mid
    fib_arc_bear=c<fib_arc_mid

    # Horizontal-line support/resistance proxies from prior completed pivots.
    horizontal_resistance=fib_hi
    horizontal_support=fib_lo
    horizontal_break_up=c>horizontal_resistance
    horizontal_break_down=c<horizontal_support
    near_horizontal_resistance=(horizontal_resistance-c)/atr < .25
    near_horizontal_support=(c-horizontal_support)/atr < .25

    # Vertical-line/time separators as causal calendar/time buckets.
    hours=df["timestamp"].dt.hour
    vertical_session=(hours.isin([0,6,7,8,12,13,14,15,16])).astype(bool)
    vertical_day_open=(hours==0)
    vertical_week_open=(df["timestamp"].dt.dayofweek==0)&(hours==0)

    # Objective trendline/channel proxies: regression slope and normalized
    # channel position over the preceding 20 completed candles.
    slope20=rolling_slope(c,20)
    channel_hi=h.shift(1).rolling(20).max(); channel_lo=l.shift(1).rolling(20).min()
    channel_rng=(channel_hi-channel_lo).replace(0,np.nan)
    channel_pos=(c-channel_lo)/channel_rng

    # Additional market-structure / breakout-context families.
    # These are deliberately generic so the research can discover which
    # consolidation pattern matters instead of hard-coding one textbook setup.
    range_hi20=h.shift(1).rolling(20).max()
    range_lo20=l.shift(1).rolling(20).min()
    range_mid20=(range_hi20+range_lo20)/2
    range_width=(range_hi20-range_lo20).replace(0,np.nan)
    range_width5=(h.shift(1).rolling(5).max()-l.shift(1).rolling(5).min()).replace(0,np.nan)
    range_width20=range_width
    range_compression=range_width5/range_width20
    inside_bar=(h<h.shift(1))&(l>l.shift(1))
    inside2=inside_bar&inside_bar.shift(1)
    nr7=(h-l)==(h-l).rolling(7).min()
    nr4=(h-l)==(h-l).rolling(4).min()
    breakout_up20=c>range_hi20
    breakout_down20=c<range_lo20
    retest_up=(l<=range_hi20)&(c>range_hi20)
    retest_down=(h>=range_lo20)&(c<range_lo20)
    false_break_up=(h>range_hi20)&(c<range_hi20)
    false_break_down=(l<range_lo20)&(c>range_lo20)
    range_tight=(range_width20/atr)<3
    range_expansion=(range_width5/range_width20)>1.15
    squeeze_release_up=bb_squeeze & breakout_up20
    squeeze_release_down=bb_squeeze & breakout_down20

    # Candle anatomy / price-action families.
    upper_wick=(h-np.maximum(c,o)).clip(lower=0)
    lower_wick=(np.minimum(c,o)-l).clip(lower=0)
    pin_bull=(lower_wick>=body*2)&((h-c)/(h-l).replace(0,np.nan)<.35)
    pin_bear=(upper_wick>=body*2)&((c-l)/(h-l).replace(0,np.nan)<.35)
    engulf_bull=(c>o)&(c.shift(1)<o.shift(1))&(c>=o.shift(1))&(o<=c.shift(1))
    engulf_bear=(c<o)&(c.shift(1)>o.shift(1))&(c<=o.shift(1))&(o>=c.shift(1))
    strong_close_bull=(h-c)/(h-l).replace(0,np.nan)<.15
    strong_close_bear=(c-l)/(h-l).replace(0,np.nan)<.15
    gap_proxy_up=o>h.shift(1)
    gap_proxy_down=o<l.shift(1)

    # Swing sequence / market-structure transitions.
    hh20=h>h.shift(1).rolling(20).max()
    ll20=l<l.shift(1).rolling(20).min()
    higher_low=l>l.shift(1)
    lower_high=h<h.shift(1)
    structure_bull_transition=higher_low & (c>c.shift(1))
    structure_bear_transition=lower_high & (c<c.shift(1))

    # Elliott-wave-inspired causal structure proxies.
    # Exact Elliott labeling is discretionary; here we encode reproducible
    # swing/impulse/retracement relationships using only completed bars.
    swing_hi5=h.shift(1).rolling(5).max()
    swing_lo5=l.shift(1).rolling(5).min()
    swing_hi10=h.shift(1).rolling(10).max()
    swing_lo10=l.shift(1).rolling(10).min()
    up_impulse=(c>swing_hi5)&(c>o)
    down_impulse=(c<swing_lo5)&(c<o)
    prior_up_leg=(c.shift(1)-c.shift(5))
    prior_down_leg=(c.shift(5)-c.shift(1))
    retrace_from_hi=(swing_hi10-c)/(swing_hi10-swing_lo10).replace(0,np.nan)
    retrace_from_lo=(c-swing_lo10)/(swing_hi10-swing_lo10).replace(0,np.nan)
    # Approximate wave-2/4 style pullbacks and wave-3/5 style impulses.
    elliott_wave2_bull=(retrace_from_hi>=.382)&(retrace_from_hi<=.618)&(prior_up_leg>0)
    elliott_wave4_bull=(retrace_from_hi>=.236)&(retrace_from_hi<=.382)&(prior_up_leg>0)
    elliott_wave3_bull=up_impulse&(prior_up_leg>0)&(atr>atr.shift(3))
    elliott_wave5_bull=up_impulse&(c>swing_hi10)&(c.shift(1)>c.shift(2))
    elliott_wave2_bear=(retrace_from_lo>=.382)&(retrace_from_lo<=.618)&(prior_down_leg>0)
    elliott_wave4_bear=(retrace_from_lo>=.236)&(retrace_from_lo<=.382)&(prior_down_leg>0)
    elliott_wave3_bear=down_impulse&(prior_down_leg>0)&(atr>atr.shift(3))
    elliott_wave5_bear=down_impulse&(c<swing_lo10)&(c.shift(1)<c.shift(2))
    elliott_impulse_bull=elliott_wave3_bull|elliott_wave5_bull
    elliott_impulse_bear=elliott_wave3_bear|elliott_wave5_bear
    elliott_pullback_bull=elliott_wave2_bull|elliott_wave4_bull
    elliott_pullback_bear=elliott_wave2_bear|elliott_wave4_bear

    # Shape proxies from OHLC rather than subjective visual pattern labels.
    compression=(h-l).rolling(5).mean()/(h-l).rolling(20).mean().replace(0,np.nan)
    triangle_proxy=(h.shift(1).rolling(10).max()-h.shift(1).rolling(10).min())/(l.shift(1).rolling(10).max()-l.shift(1).rolling(10).min()).replace(0,np.nan)

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
        "dmi_adx_bull":(adx>25)&(pdi>mdi), "dmi_adx_bear":(adx>25)&(mdi>pdi),
        "dmi_adx_strong_bull":(adx.diff()>0)&(pdi>mdi), "dmi_adx_strong_bear":(adx.diff()>0)&(mdi>pdi),
        "volatility":atr_pct>=.7,
        "bb_bull":bbpos>.5, "bb_bear":bbpos<.5,
        "bb_upper":bbpos>=.8, "bb_lower":bbpos<=.2,
        "superbb_bull":c>bb3_upper, "superbb_bear":c<bb3_lower,
        "superbb_inside":(c<=bb3_upper)&(c>=bb3_lower),
        "superbb_squeeze":bb_squeeze, "superbb_expand":bb_width/bb_width.shift(3)>1.15,
        "body_bull":(c>o)&(body>.6), "body_bear":(c<o)&(body>.6),

        # SMA / Ichimoku / Stochastic / RCI / Fibonacci / line-channel proxies.
        "sma20_bull":c>sma20, "sma20_bear":c<sma20,
        "sma50_bull":c>sma50, "sma50_bear":c<sma50,
        "sma200_bull":c>sma200, "sma200_bear":c<sma200,
        "sma_stack_bull":(sma20>sma50)&(sma50>sma200),
        "sma_stack_bear":(sma20<sma50)&(sma50<sma200),
        "ichimoku_bull":c>cloud_top, "ichimoku_bear":c<cloud_bottom,
        "ichimoku_tk_bull":tenkan>kijun, "ichimoku_tk_bear":tenkan<kijun,
        "ichimoku_cloud_bull":span_a>span_b, "ichimoku_cloud_bear":span_a<span_b,
        "span_model_bull":(tenkan>kijun)&(c>cloud_top), "span_model_bear":(tenkan<kijun)&(c<cloud_bottom),
        "stoch_bull":stoch_k>stoch_d, "stoch_bear":stoch_k<stoch_d,
        "stoch_cross_up":(stoch_k.shift(1)<=stoch_d.shift(1))&(stoch_k>stoch_d),
        "stoch_cross_down":(stoch_k.shift(1)>=stoch_d.shift(1))&(stoch_k<stoch_d),
        "stoch_recover_up":(stoch_k.shift(1)<20)&(stoch_k>=20),
        "stoch_fall_down":(stoch_k.shift(1)>80)&(stoch_k<=80),
        "rci_bull":rci14>0, "rci_bear":rci14<0,
        "rci_strong_bull":rci14>=80, "rci_strong_bear":rci14<=-80,
        "rci_cross_up":(rci14.shift(1)<=0)&(rci14>0),
        "rci_cross_down":(rci14.shift(1)>=0)&(rci14<0),
        "fib_above382":c>fib382, "fib_above500":c>fib500,
        "fib_above618":c>fib618, "fib_below382":c<fib382,
        "fib_below500":c<fib500, "fib_below618":c<fib618,
        "fib_time_zone":fib_time_zone, "fib_time_5":fib_time_5,
        "fib_time_8":fib_time_8, "fib_time_13":fib_time_13,
        "fib_fan_up":fib_fan_up, "fib_fan_down":fib_fan_down,
        "fib_arc_bull":fib_arc_bull, "fib_arc_bear":fib_arc_bear,
        "horizontal_resistance_near":near_horizontal_resistance,
        "horizontal_support_near":near_horizontal_support,
        "horizontal_break_up":horizontal_break_up,
        "horizontal_break_down":horizontal_break_down,
        "vertical_session":vertical_session,
        "vertical_day_open":vertical_day_open,
        "vertical_week_open":vertical_week_open,
        "fib_retrace38_bull":(c>=fib382)&(c<=fib500),
        "fib_retrace62_bull":(c>=fib618)&(c<=fib786),
        "fib_retrace38_bear":(c<=fib618)&(c>=fib500),
        "fib_retrace62_bear":(c<=fib382)&(c>=fib236),
        "trendline_up":slope20>0, "trendline_down":slope20<0,
        "channel_upper":channel_pos>=.80, "channel_lower":channel_pos<=.20,
        "channel_mid_bull":channel_pos>.50, "channel_mid_bear":channel_pos<.50,
        "channel_breakout_up":c>channel_hi, "channel_breakout_down":c<channel_lo,
        "compression":compression<.75,
        "shape_triangle_proxy":(compression<.75)&(channel_rng/channel_lo.replace(0,np.nan)<.03),
        "rectangle_range":(channel_pos>.15)&(channel_pos<.85)&(compression>.80),
        "triangle_breakout_up":(compression<.75)&(c>channel_hi),
        "triangle_breakout_down":(compression<.75)&(c<channel_lo),
        "ellipse_compression":compression<.65,
        "range_compression":range_compression<.70, "range_tight":range_tight,
        "range_expansion":range_expansion, "inside_bar":inside_bar, "inside_2":inside2,
        "nr7":nr7, "nr4":nr4,
        "breakout_up20":breakout_up20, "breakout_down20":breakout_down20,
        "retest_up":retest_up, "retest_down":retest_down,
        "false_break_up":false_break_up, "false_break_down":false_break_down,
        "squeeze_release_up":squeeze_release_up, "squeeze_release_down":squeeze_release_down,
        "pin_bull":pin_bull, "pin_bear":pin_bear,
        "engulf_bull":engulf_bull, "engulf_bear":engulf_bear,
        "strong_close_bull":strong_close_bull, "strong_close_bear":strong_close_bear,
        "gap_proxy_up":gap_proxy_up, "gap_proxy_down":gap_proxy_down,
        "hh20":hh20, "ll20":ll20,
        "structure_bull_transition":structure_bull_transition,
        "structure_bear_transition":structure_bear_transition,
        "elliott_wave2_bull":elliott_wave2_bull, "elliott_wave3_bull":elliott_wave3_bull,
        "elliott_wave4_bull":elliott_wave4_bull, "elliott_wave5_bull":elliott_wave5_bull,
        "elliott_impulse_bull":elliott_impulse_bull, "elliott_pullback_bull":elliott_pullback_bull,
        "elliott_wave2_bear":elliott_wave2_bear, "elliott_wave3_bear":elliott_wave3_bear,
        "elliott_wave4_bear":elliott_wave4_bear, "elliott_wave5_bear":elliott_wave5_bear,
        "elliott_impulse_bear":elliott_impulse_bear, "elliott_pullback_bear":elliott_pullback_bear,

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



TOOLKIT_BULL=[
    # Every item shown in the user's chart-tool list is represented.
    "sma20_bull","sma50_bull","sma200_bull","sma_stack_bull",
    "ichimoku_bull","ichimoku_tk_bull","ichimoku_cloud_bull","span_model_bull",
    "bb_bull","bb_upper","superbb_bull","superbb_squeeze","superbb_expand",
    "stoch_bull","stoch_cross_up","stoch_recover_up",
    "rci_bull","rci_strong_bull","rci_cross_up",
    "dmi_adx_bull","dmi_adx_strong_bull",
    "fib_above382","fib_above500","fib_above618","fib_retrace38_bull","fib_retrace62_bull",
    "fib_time_zone","fib_time_5","fib_time_8","fib_time_13","fib_fan_up","fib_arc_bull",
    "trendline_up","channel_upper","channel_mid_bull","channel_breakout_up",
    "horizontal_resistance_near","horizontal_support_near","horizontal_break_up",
    "vertical_session","vertical_day_open","vertical_week_open",
    "rectangle_range","triangle_breakout_up","ellipse_compression",
    "elliott_wave2_bull","elliott_wave3_bull","elliott_wave4_bull","elliott_wave5_bull","elliott_impulse_bull","elliott_pullback_bull",
    "range_compression","range_tight","range_expansion","inside_bar","inside_2","nr7","nr4",
    "breakout_up20","retest_up","squeeze_release_up","pin_bull","engulf_bull","strong_close_bull",
    "gap_proxy_up","hh20","structure_bull_transition",
    "compression","shape_triangle_proxy"
]
TOOLKIT_BEAR=[
    "sma20_bear","sma50_bear","sma200_bear","sma_stack_bear",
    "ichimoku_bear","ichimoku_tk_bear","ichimoku_cloud_bear","span_model_bear",
    "bb_bear","bb_lower","superbb_bear","superbb_squeeze","superbb_expand",
    "stoch_bear","stoch_cross_down","stoch_fall_down",
    "rci_bear","rci_strong_bear","rci_cross_down",
    "dmi_adx_bear","dmi_adx_strong_bear",
    "fib_below382","fib_below500","fib_below618","fib_retrace38_bear","fib_retrace62_bear",
    "fib_time_zone","fib_time_5","fib_time_8","fib_time_13","fib_fan_down","fib_arc_bear",
    "trendline_down","channel_lower","channel_mid_bear","channel_breakout_down",
    "horizontal_resistance_near","horizontal_support_near","horizontal_break_down",
    "vertical_session","vertical_day_open","vertical_week_open",
    "rectangle_range","triangle_breakout_down","ellipse_compression",
    "elliott_wave2_bear","elliott_wave3_bear","elliott_wave4_bear","elliott_wave5_bear","elliott_impulse_bear","elliott_pullback_bear",
    "range_compression","range_tight","range_expansion","inside_bar","inside_2","nr7","nr4",
    "breakout_down20","retest_down","squeeze_release_down","pin_bear","engulf_bear","strong_close_bear",
    "gap_proxy_down","ll20","structure_bear_transition",
    "compression","shape_triangle_proxy"
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
