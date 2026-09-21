#!/usr/bin/env python3
"""Origin-signal prevalence study for clustered +50 pip events.

Goal: identify signals that are common near the earliest causal source candle of
a clustered +50 pip move, then compare their prevalence with matched control
candles from the same pair/timeframe/year. This is a discovery study, not a
live-trade win-rate test.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("direct", "scripts/research_50pip_direct_entry.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PAIRS = mod.PAIRS
TFS = mod.TFS
PIP = {p: 0.01 if "jpy" in p else 0.0001 for p in PAIRS}
HORIZONS = {"h4": 24, "d1": 20}
OFFSETS = [0, -1, -2, -3, -4, -5, -6]
THRESHOLD = 50

def load_raw(tf, pair):
    return mod.load_market(tf, pair).reset_index(drop=True)

def cluster_origins(df, pair, tf):
    max_h = HORIZONS[tf]
    pip = PIP[pair]
    rows = []
    for i in range(len(df) - max_h):
        entry = float(df.at[i, "close"])
        future = df.iloc[i+1:i+max_h+1]
        up = (future["high"].cummax() - entry) / pip
        hit = up[up >= THRESHOLD]
        if hit.empty:
            continue
        idx = int(hit.index[0])
        rows.append((i, idx, df.at[idx, "timestamp"], float(up.max())))
    rows.sort()
    keep = []
    last_hit = None
    for i, hit_i, hit_ts, mx in rows:
        if last_hit is None or df.at[i, "timestamp"] > last_hit:
            keep.append((i, hit_i, hit_ts, mx))
            last_hit = hit_ts
    return keep

def event_origin_frame(df, features, pair, tf):
    origins = cluster_origins(df, pair, tf)
    if not origins:
        return pd.DataFrame()
    rows = []
    for i, hit_i, hit_ts, mx in origins:
        y = int(df.at[i, "timestamp"].year)
        for off in OFFSETS:
            j = i + off
            if j < 0 or j >= len(df):
                continue
            row = {
                "pair": pair, "timeframe": tf, "year": y,
                "origin_index": i, "hit_index": hit_i,
                "hit_timestamp": hit_ts, "max_excursion_pips": mx,
                "offset": off, "index": j,
            }
            for f in features:
                row[f] = features[f].iloc[j]
            rows.append(row)
    return pd.DataFrame(rows)

def feature_mask(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return pd.to_numeric(s, errors="coerce").fillna(0).ne(0)

def main():
    Path("reports").mkdir(exist_ok=True)
    bull_features = list(dict.fromkeys(mod.BULL_BASE + mod.TOOLKIT_BULL))
    # Use both directional feature families so the study can discover
    # asymmetric precursors instead of assuming a mirrored definition.
    feature_pool = list(dict.fromkeys(
        bull_features + mod.BEAR_BASE + mod.TOOLKIT_BEAR
    ))

    event_parts = []
    control_parts = []
    for tf in TFS:
        for pair in PAIRS:
            df = load_raw(tf, pair)
            feat = mod.build_features(df)
            feat = feat[[f for f in feature_pool if f in feat.columns]]
            z = event_origin_frame(df, feat, pair, tf)
            if not z.empty:
                event_parts.append(z)

            # Matched controls: every 10th candle by deterministic stride,
            # excluding any source candle inside a +/-6-bar event-origin band.
            origins = [x[0] for x in cluster_origins(df, pair, tf)]
            blocked = set()
            for i in origins:
                blocked.update(range(max(0, i-6), min(len(df), i+7)))
            stride = 10
            for j in range(0, len(df), stride):
                if j in blocked:
                    continue
                row = {"pair":pair, "timeframe":tf, "year":int(df.at[j,"timestamp"].year),
                       "offset":0, "index":j, "origin_index":-1}
                for f in feature_pool:
                    if f in feat.columns:
                        row[f] = feat[f].iloc[j]
                control_parts.append(row)

    events = pd.concat(event_parts, ignore_index=True)
    controls = pd.DataFrame(control_parts)
    events.to_csv("reports/50pip_origin_event_samples.csv", index=False, float_format="%.8f")

    rows = []
    for tf in TFS:
        for direction, features in (("bull", bull_features),
                                    ("bear", list(dict.fromkeys(mod.BEAR_BASE + mod.TOOLKIT_BEAR)))):
            features = [f for f in features if f in events.columns]
            for f in features:
                for off in OFFSETS:
                    e = events[(events.timeframe == tf) & (events.offset == off)]
                    c = controls[controls.timeframe == tf]
                    en = int(e[f].notna().sum())
                    cn = int(c[f].notna().sum())
                    ep = float(feature_mask(e[f]).mean()) if en else np.nan
                    cp = float(feature_mask(c[f]).mean()) if cn else np.nan
                    rows.append([tf,direction,f,off,en,ep,cn,cp,ep/cp if cp else np.nan,ep-cp if cn else np.nan])

    out = pd.DataFrame(rows, columns=[
        "timeframe","direction","feature","offset","event_n","event_presence",
        "control_n","control_presence","prevalence_lift","presence_delta"
    ])
    out.to_csv("reports/50pip_origin_signal_prevalence.csv", index=False, float_format="%.8f")

    # Transition-at-origin: condition turns on between offset -1 and 0.
    tr = []
    for tf in TFS:
        features = [f for f in bull_features if f in events.columns]
        for f in features:
            e0 = events[(events.timeframe == tf) & (events.offset == 0)]
            em1 = events[(events.timeframe == tf) & (events.offset == -1)]
            if e0.empty or em1.empty:
                continue
            # Match by pair/origin_index because event rows contain multiple offsets.
            a = e0.set_index(["pair","origin_index"])[f]
            b = em1.set_index(["pair","origin_index"])[f]
            idx = a.index.intersection(b.index)
            if len(idx):
                aa = feature_mask(a.loc[idx])
                bb = feature_mask(b.loc[idx])
                n = int((aa & ~bb).sum())
                rate = n / len(idx) if len(idx) else np.nan
                tr.append([tf,f,len(idx),n,rate])
    pd.DataFrame(tr, columns=["timeframe","feature","origin_pairs","turn_on_n","turn_on_rate"]).to_csv(
        "reports/50pip_origin_signal_transitions.csv", index=False, float_format="%.8f"
    )

    print("event_sample_rows", len(events), "control_rows", len(controls))
    print(out.sort_values(["prevalence_lift","event_presence"], ascending=False).head(80).to_string(index=False))

if __name__ == "__main__":
    main()
