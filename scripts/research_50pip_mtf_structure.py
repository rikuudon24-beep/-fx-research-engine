#!/usr/bin/env python3
"""Multi-timeframe + market-structure research for direct +50 entry signals.

Causal alignment rule:
- An entry candle uses its own completed OHLC/indicator state.
- Higher/lower timeframe context is taken only from the latest *completed*
  opposite timeframe candle strictly before the entry candle timestamp.
  This avoids using a partially formed higher-timeframe candle.
- Cross-pair context is computed from information available at the same entry
  timestamp and uses only lagged returns/state.

This is signal research, not a complete trading win-rate.
"""
from pathlib import Path
from itertools import combinations
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("direct", "scripts/research_50pip_direct_entry.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PAIRS = mod.PAIRS
TFS = mod.TFS
PIP = mod.PIP
HORIZONS = mod.HORIZONS

BULL_LOCAL = mod.BULL_BASE
BEAR_LOCAL = mod.BEAR_BASE

# Higher-timeframe context names are prefixed mtf_.
MTF_BULL = [
    "mtf_trend_bull","mtf_price20_bull","mtf_price200_bull",
    "mtf_ema20_rising","mtf_macd_bull","mtf_macd_rising",
    "mtf_rsi60","mtf_adx25","mtf_adx_rising","mtf_di_bull",
    "mtf_di_strong_bull","mtf_volatility","mtf_bb_bull","mtf_breakout_up",
]
MTF_BEAR = [
    "mtf_trend_bear","mtf_price20_bear","mtf_price200_bear",
    "mtf_ema20_falling","mtf_macd_bear","mtf_macd_falling",
    "mtf_rsi40","mtf_adx25","mtf_adx_rising","mtf_di_bear",
    "mtf_di_strong_bear","mtf_volatility","mtf_bb_bear","mtf_breakout_down",
]

STRUCT_BULL = [
    "structure_hh","structure_hl","structure_breakout20_up","range_expand",
    "close_near_high","impulse_bull","pullback_bull","distance_high_ok",
]
STRUCT_BEAR = [
    "structure_lh","structure_ll","structure_breakout20_down","range_expand",
    "close_near_low","impulse_bear","pullback_bear","distance_low_ok",
]

CROSS_BULL = ["cross_group_bull","cross_group_return_up","cross_target_relative_up"]
CROSS_BEAR = ["cross_group_bear","cross_group_return_down","cross_target_relative_down"]

def load_market(tf, pair):
    return mod.load_market(tf, pair)

def structure_features(df):
    c,h,l,o = df.close, df.high, df.low, df.open
    rng = (h-l).replace(0, np.nan)
    prev_h20 = h.shift(1).rolling(20).max()
    prev_l20 = l.shift(1).rolling(20).min()
    prev_h5 = h.shift(1).rolling(5).max()
    prev_l5 = l.shift(1).rolling(5).min()
    atr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1).ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    # Causal swing/structure proxies: compare current completed bar with
    # previous rolling extrema, never future pivots.
    hh = h > prev_h5
    ll = l < prev_l5
    f = pd.DataFrame({
        "structure_hh": hh,
        "structure_hl": l > l.shift(1),
        "structure_lh": h < h.shift(1),
        "structure_ll": ll,
        "structure_breakout20_up": c > prev_h20,
        "structure_breakout20_down": c < prev_l20,
        "range_expand": rng > rng.shift(3)*1.20,
        "close_near_high": (h-c)/rng <= .20,
        "close_near_low": (c-l)/rng <= .20,
        "impulse_bull": (c-o) > atr*.50,
        "impulse_bear": (o-c) > atr*.50,
        "pullback_bull": (c < c.shift(1)) & (c > prev_l20),
        "pullback_bear": (c > c.shift(1)) & (c < prev_h20),
        "distance_high_ok": ((prev_h20-c)/atr > .25) & ((prev_h20-c)/atr < 3.0),
        "distance_low_ok": ((c-prev_l20)/atr > .25) & ((c-prev_l20)/atr < 3.0),
    }, index=df.index)
    return f

def aligned_mtf(local_df, mtf_df, mtf_features, local_tf, mtf_tf):
    # Repository timestamps are candle starts. A local signal is evaluated
    # at the local candle close, so the opposite-timeframe candle must have
    # COMPLETED by that close. This prevents using a still-forming D1 candle
    # inside an H4 signal (and vice versa).
    delta = {
        "h4": pd.Timedelta(hours=4),
        "d1": pd.Timedelta(days=1),
    }
    left = local_df[["timestamp"]].copy().sort_values("timestamp")
    left["entry_close"] = left["timestamp"] + delta[local_tf]

    right = pd.concat([mtf_df[["timestamp"]], mtf_features], axis=1).copy()
    right["completed_at"] = right["timestamp"] + delta[mtf_tf]
    right = right.sort_values("completed_at")
    right = right.rename(columns={c:f"mtf_{c}" for c in mtf_features.columns})

    merged = pd.merge_asof(
        left, right,
        left_on="entry_close", right_on="completed_at",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged[mtf_features.columns.map(lambda c:f"mtf_{c}")]

def normalized_cross_context(all_close, pair, tf):
    # Build target-relative context from logically related baskets. Returns
    # are sign-normalized so "up" means movement supportive of the target.
    if "jpy" in pair:
        basket = [p for p in PAIRS if "jpy" in p]
        signs = {p: 1 for p in basket}
    elif pair in {"eurusd","gbpusd","audusd","nzdusd"}:
        basket = ["eurusd","gbpusd","audusd","nzdusd"]
        signs = {p: 1 for p in basket}
    elif pair in {"usdcad","usdchf","usdjpy"}:
        basket = ["usdcad","usdchf","usdjpy"]
        signs = {p: -1 for p in basket}
    else:
        basket = [p for p in PAIRS if p != pair]
        signs = {p: 1 for p in basket}

    ret = []
    bull = []
    for p in basket:
        if p not in all_close:
            continue
        s = all_close[p]
        r = s.pct_change(3) * signs.get(p, 1)
        ret.append(r.rename(p))
        bull.append((r > 0).astype(float).rename(p))
    if not ret:
        return pd.DataFrame(index=all_close[pair].index)

    rr = pd.concat(ret, axis=1)
    bb = pd.concat(bull, axis=1)
    target_ret = all_close[pair].pct_change(3)
    group_ret = rr.mean(axis=1)
    group_bull = bb.mean(axis=1)
    relative = target_ret - group_ret

    # Do not use the current target candle's return as a feature.
    out = pd.DataFrame(index=all_close[pair].index)
    out["cross_group_bull"] = group_bull.shift(1) >= .60
    out["cross_group_bear"] = group_bull.shift(1) <= .40
    out["cross_group_return_up"] = group_ret.shift(1) > 0
    out["cross_group_return_down"] = group_ret.shift(1) < 0
    out["cross_target_relative_up"] = relative.shift(1) > 0
    out["cross_target_relative_down"] = relative.shift(1) < 0
    return out

def build_dataset(tf):
    rows = []
    raw = {}
    for pair in PAIRS:
        df = load_market(tf, pair)
        raw[pair] = df
    mtf_tf = "d1" if tf == "h4" else "h4"

    # Precompute opposite timeframe data and features.
    mtf_raw = {}
    mtf_feat = {}
    for pair in PAIRS:
        md = load_market(mtf_tf, pair)
        mtf_raw[pair] = md
        mtf_feat[pair] = mod.build_features(md)

    close_series = {}
    for pair, df in raw.items():
        s = df.set_index("timestamp").close.sort_index()
        close_series[pair] = s
    close_panel = pd.concat(close_series, axis=1)

    for pair in PAIRS:
        df = raw[pair]
        local = mod.build_features(df)
        structure = structure_features(df)
        aligned = aligned_mtf(df, mtf_raw[pair], mtf_feat[pair], tf, mtf_tf)

        # Cross-pair context is computed on a shared timestamp grid.
        cross = normalized_cross_context(close_panel, pair, tf)
        cross = cross.reindex(df.timestamp).reset_index(drop=True)

        lab = mod.add_labels(df, pair, tf)
        z = pd.concat([
            df[["timestamp","open","high","low","close"]],
            local, structure, aligned, cross, lab
        ], axis=1)
        z["pair"] = pair
        z["timeframe"] = tf
        rows.append(z)

    return pd.concat(rows, ignore_index=True)

# Keep feature namespaces unique: local direct-entry and structure features
# must never create duplicate DataFrame column names, because g[n] would
# otherwise return a DataFrame and Series boolean-mask operations can fail.
def mask(g, names):
    m = pd.Series(True, index=g.index)
    for n in names:
        if n not in g.columns:
            return pd.Series(False, index=g.index)
        m &= g[n].fillna(False)
    return m

def candidates(direction):
    local = BULL_LOCAL if direction == "bull" else BEAR_LOCAL
    mtf = MTF_BULL if direction == "bull" else MTF_BEAR
    struct = STRUCT_BULL if direction == "bull" else STRUCT_BEAR
    cross = CROSS_BULL if direction == "bull" else CROSS_BEAR
    out = []
    seen = set()

    # Single features are useful for screening but the main research emphasis
    # is confirmation: local trigger + higher-timeframe state + structure.
    pools = [
        [(x,) for x in mtf],
        [(x,) for x in struct],
        [(x,) for x in cross],
        [(a,b) for a,b in combinations(mtf,2)],
        [(a,b) for a,b in combinations(struct,2)],
        [(a,b) for a,b in combinations(local,2)],
    ]
    for pool in pools:
        for c in pool:
            if c not in seen:
                seen.add(c); out.append(c)

    for l in local:
        for m in mtf:
            c = (l,m)
            if c not in seen:
                seen.add(c); out.append(c)
    for l in local:
        for s in struct:
            c = (l,s)
            if c not in seen:
                seen.add(c); out.append(c)
    for m in mtf:
        for s in struct:
            c = (m,s)
            if c not in seen:
                seen.add(c); out.append(c)

    # Bounded 3-way search. Instead of the full local x MTF x structure
    # Cartesian product (thousands of masks per direction), test a curated
    # set of causal feature families covering trend, trigger, structure and
    # expansion. This keeps the study broad without exhausting the runner.
    mtf_core = [x for x in mtf if x in {
        "mtf_trend_bull","mtf_trend_bear",
        "mtf_price20_bull","mtf_price20_bear",
        "mtf_price200_bull","mtf_price200_bear",
        "mtf_macd_bull","mtf_macd_bear",
        "mtf_adx_rising","mtf_adx_rising",
        "mtf_di_strong_bull","mtf_di_strong_bear",
    }]
    local_core = [x for x in local if x in {
        "price20_cross_up","price20_cross_down",
        "price200_cross_up","price200_cross_down",
        "macd_cross_up","macd_cross_down",
        "macd_accel_up","macd_accel_down",
        "adx_up3","atr_expand","bb_expand",
        "breakout20_up","breakout20_down",
    }]
    struct_core = [x for x in struct if x in {
        "structure_hh","structure_hl","structure_lh","structure_ll",
        "structure_breakout20_up","structure_breakout20_down","range_expand",
        "impulse_bull","impulse_bear","pullback_bull","pullback_bear",
        "close_near_high","close_near_low",
    }]
    for l in local_core:
        for m in mtf_core:
            for s in struct_core:
                c = (l,m,s)
                if c not in seen:
                    seen.add(c); out.append(c)

    # Explicit high-value families: trend context -> local trigger -> expansion.
    if direction == "bull":
        families = [
            ("mtf_trend_bull","price20_cross_up","atr_expand"),
            ("mtf_price200_bull","macd_cross_up","atr_expand"),
            ("mtf_macd_bull","adx_up3","atr_expand"),
            ("mtf_adx_rising","breakout20_up","range_expand"),
            ("mtf_trend_bull","structure_hh","impulse_bull"),
            ("mtf_trend_bull","pullback_bull","macd_accel_up"),
            ("mtf_trend_bull","breakout20_up","cross_group_bull"),
        ]
    else:
        families = [
            ("mtf_trend_bear","price20_cross_down","atr_expand"),
            ("mtf_price200_bear","macd_cross_down","atr_expand"),
            ("mtf_macd_bear","adx_up3","atr_expand"),
            ("mtf_adx_rising","breakout20_down","range_expand"),
            ("mtf_trend_bear","structure_lh","impulse_bear"),
            ("mtf_trend_bear","pullback_bear","macd_accel_down"),
            ("mtf_trend_bear","breakout20_down","cross_group_bear"),
        ]
    for c in families:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def wilson_lower(k,n,z=1.959963984540054):
    if n <= 0: return np.nan
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    return center-z*np.sqrt((p*(1-p)/n)+(z*z/(4*n*n)))/den

def main():
    Path("reports").mkdir(exist_ok=True)
    print("START MTF/structure research")
    datasets = {tf: build_dataset(tf) for tf in TFS}
    rows = []
    scored = []
    for tf, all_df in datasets.items():
        # Cache candidate lists and boolean masks once per direction. The old
        # implementation rebuilt the same masks for every horizon, which made
        # the workflow much slower and more memory-hungry than necessary.
        for direction in ("bull","bear"):
            target_horizons = HORIZONS[tf]
            combos = candidates(direction)
            mask_cache = {}
            for combo in combos:
                m = mask(all_df, combo)
                if int(m.sum()) >= 100:
                    mask_cache[combo] = m

            disc = all_df[all_df.timestamp.dt.year <= 2024]
            val = all_df[all_df.timestamp.dt.year == 2025]
            oos = all_df[all_df.timestamp.dt.year >= 2026]

            for h in target_horizons:
                target = f"hit50_h{h}"
                base_all = float(all_df[target].mean())
                for combo, m in mask_cache.items():
                    n = int(m.sum())
                    if n < 100:
                        continue
                    hit = float(all_df.loc[m, target].mean())
                    rows.append([
                        tf,direction,h,"+".join(combo),n,base_all,hit,
                        hit-base_all,hit/base_all if base_all else np.nan
                    ])

                base = float(disc[target].mean())
                cand = []
                for combo, m in mask_cache.items():
                    md = m & (all_df.timestamp.dt.year <= 2024)
                    n = int(md.sum())
                    if n < 150:
                        continue
                    hit = float(all_df.loc[md, target].mean())
                    lift = hit/base if base else np.nan
                    if np.isfinite(lift) and lift >= 1.10:
                        cand.append((combo,n,hit,lift))
                cand = sorted(cand,key=lambda x:(x[3],x[1]),reverse=True)[:80]

                for combo,dn,dh,dl in cand:
                    m = mask_cache[combo]
                    vals=[tf,direction,h,"+".join(combo),dn,dh,dl]
                    for g in (val,oos):
                        mg = m & (all_df.timestamp.dt.year == (2025 if g is val else 2026))
                        # OOS is 2026+, not only 2026.
                        if g is oos:
                            mg = m & (all_df.timestamp.dt.year >= 2026)
                        n=int(mg.sum())
                        hit=float(all_df.loc[mg,target].mean()) if n else np.nan
                        b=float(g[target].mean()) if len(g) else np.nan
                        vals += [n,hit,hit/b if b else np.nan,
                                 wilson_lower(int(all_df.loc[mg,target].sum()),n)]
                    scored.append(vals)

    summary=pd.DataFrame(rows,columns=[
        "timeframe","direction","horizon_bars","conditions","samples",
        "base_rate","hit50_rate","rate_diff","lift"
    ])
    summary.to_csv("reports/50pip_mtf_structure_conditions.csv",index=False,float_format="%.8f")

    oos=pd.DataFrame(scored,columns=[
        "timeframe","direction","horizon_bars","conditions",
        "discovery_samples","discovery_hit_rate","discovery_lift",
        "validation_samples","validation_hit_rate","validation_lift",
        "validation_wilson95_lower","oos_samples","oos_hit_rate",
        "oos_lift","oos_wilson95_lower"
    ])
    oos.to_csv("reports/50pip_mtf_structure_oos.csv",index=False,float_format="%.8f")

    robust=[]
    for r in oos.itertuples(index=False):
        if r.oos_samples < 50: continue
        g=datasets[r.timeframe]
        sub=g.loc[mask(g,r.conditions.split("+"))]
        target=f"hit50_h{r.horizon_bars}"
        base=float(g.loc[g.timestamp.dt.year>=2026, target].mean())
        pairs=[]
        for pair,x in sub[sub.timestamp.dt.year>=2026].groupby("pair"):
            if len(x)>=5:
                pairs.append((pair,len(x),float(x[target].mean())))
        if pairs:
            robust.append([
                r.timeframe,r.direction,r.horizon_bars,r.conditions,
                r.discovery_samples,r.validation_samples,r.oos_samples,
                r.discovery_lift,r.validation_lift,r.oos_lift,
                r.oos_wilson95_lower,len(pairs),
                sum(x[2]>=base for x in pairs),
                np.mean([x[2] for x in pairs]),min(x[2] for x in pairs)
            ])
    rob=pd.DataFrame(robust,columns=[
        "timeframe","direction","horizon_bars","conditions",
        "discovery_samples","validation_samples","oos_samples",
        "discovery_lift","validation_lift","oos_lift",
        "oos_wilson95_lower","oos_pairs_n_ge5",
        "oos_pairs_above_oos_base","oos_pair_hit_mean","oos_min_pair_hit"
    ])
    rob.to_csv("reports/50pip_mtf_structure_robustness.csv",index=False,float_format="%.8f")

    print("MTF/structure summary rows:",len(summary))
    print("MTF/structure OOS candidates:",len(oos))
    print("MTF/structure robustness rows:",len(rob))
    if not rob.empty:
        print(rob.sort_values(
            ["oos_wilson95_lower","oos_lift","oos_samples"],
            ascending=False
        ).head(40).to_string(index=False))

if __name__=="__main__":
    main()
