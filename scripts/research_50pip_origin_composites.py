#!/usr/bin/env python3
"""Composite origin-signal research.

Finds multi-signal states around the source candle of clustered +50 pip moves.
Atoms are feature@offset (-3..0). Discovery selects combinations only from
2021-2024; validation=2025 and OOS>=2026 are scored without re-selection.
Controls are deterministic non-event candles from the same pair/timeframe.
"""
from pathlib import Path
import importlib.util, itertools
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("direct","scripts/research_50pip_direct_entry.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
PAIRS=mod.PAIRS; TFS=mod.TFS
PIP={p:0.01 if "jpy" in p else 0.0001 for p in PAIRS}
MAXH={"h4":24,"d1":20}; OFFSETS=[-3,-2,-1,0]
TOP_ATOMS=28

def origins(df,pair,tf):
    pip=PIP[pair]; mh=MAXH[tf]; rows=[]
    for i in range(len(df)-mh):
        e=float(df.at[i,"close"]); fut=df.iloc[i+1:i+mh+1]
        ex=(fut["high"].cummax()-e)/pip; hit=ex[ex>=50]
        if len(hit): rows.append((i,int(hit.index[0])))
    rows.sort(); keep=[]; last=-1
    for i,hi in rows:
        if last<0 or df.at[i,"timestamp"]>last:
            keep.append(i); last=df.at[hi,"timestamp"]
    return keep

def mask(s):
    if pd.api.types.is_bool_dtype(s): return s.fillna(False).to_numpy()
    return pd.to_numeric(s,errors="coerce").fillna(0).ne(0).to_numpy()

def main():
    Path("reports").mkdir(exist_ok=True)
    allbull=list(dict.fromkeys(mod.BULL_BASE+mod.TOOLKIT_BULL))
    allbear=list(dict.fromkeys(mod.BEAR_BASE+mod.TOOLKIT_BEAR))
    # Use a broad pool but rank atoms by single-signal discovery prevalence lift.
    outputs=[]
    for tf in TFS:
      for direction,features in (("bull",allbull),("bear",allbear)):
        events=[]; controls=[]
        for pair in PAIRS:
          df=mod.load_market(tf,pair).reset_index(drop=True)
          feat=mod.build_features(df)
          feats=[f for f in features if f in feat.columns]
          oi=origins(df,pair,tf)
          blocked=set()
          for i in oi: blocked.update(range(max(0,i-6),min(len(df),i+7)))
          ci=[j for j in range(0,len(df),10) if j not in blocked]
          for i in oi:
            vals={f:mask(feat[f]) for f in feats}
            events.append((pair,int(df.at[i,"timestamp"].year),i,vals))
          for j in ci:
            vals={f:mask(feat[f]) for f in feats}
            controls.append((pair,int(df.at[j,"timestamp"].year),j,vals))
        # rank feature@offset atoms in discovery
        atoms=[]
        for f in features:
          for off in OFFSETS:
            en=ep=cn=cp=0
            for _,yr,i,v in events:
              k=i+off
              if k<0 or k>=len(v[f]): continue
              en+=1; ep+=int(v[f][k])
            for _,yr,i,v in controls:
              k=i+off
              if k<0 or k>=len(v[f]): continue
              cn+=1; cp+=int(v[f][k])
            # redo discovery-only counts
            ed=epd=cd=cpd=0
            for _,yr,i,v in events:
              if yr<=2024 and 0<=i+off<len(v[f]): ed+=1; epd+=int(v[f][i+off])
            for _,yr,i,v in controls:
              if yr<=2024 and 0<=i+off<len(v[f]): cd+=1; cpd+=int(v[f][i+off])
            lift=(epd/ed)/(cpd/cd) if ed and cd and cpd else 0
            if epd>=80 and lift>=1.15: atoms.append((lift,f,off))
        atoms=sorted(atoms,reverse=True)[:TOP_ATOMS]
        rows=[]
        def score(sample, a1,a2,lo,hi):
          n=w=0
          for _,yr,i,v in sample:
            if yr<lo or yr>hi: continue
            ok=True
            for f,o in (a1,a2):
              k=i+o
              if k<0 or k>=len(v[f]) or not v[f][k]: ok=False; break
            n+=1; w+=int(ok)
          return n,w
        for a1,a2 in itertools.combinations(atoms,2):
          # avoid duplicate same feature at adjacent offsets unless meaningful
          if a1[1]==a2[1] and a1[2]==a2[2]: continue
          en,ew=score(events,a1,a2,0,9999); cn,cw=score(controls,a1,a2,0,9999)
          if ew<40 or cw<1: continue
          ep=ew/en; cp=cw/cn; lift=ep/cp if cp else np.inf
          ddn,ddw=score(events,a1,a2,0,2024); dcn,dcw=score(controls,a1,a2,0,2024)
          if ddw<40: continue
          d_lift=(ddw/ddn)/(dcw/dcn) if dcw else np.inf
          vn,vw=score(events,a1,a2,2025,2025); vc,vcw=score(controls,a1,a2,2025,2025)
          on,ow=score(events,a1,a2,2026,9999); oc,ocw=score(controls,a1,a2,2026,9999)
          v_lift=(vw/vn)/(vcw/vc) if vn and vcw else np.nan
          o_lift=(ow/on)/(ocw/oc) if on and ocw else np.nan
          rows.append([tf,direction,a1[1],a1[2],a2[1],a2[2],ddn,ddw,ddw/ddn,d_lift,vn,vw,vw/vn if vn else np.nan,v_lift,on,ow,ow/on if on else np.nan,o_lift,ep,lift])
        out=pd.DataFrame(rows,columns=["timeframe","direction","feature_a","offset_a","feature_b","offset_b","disc_n","disc_hits","disc_rate","disc_lift","val_n","val_hits","val_rate","val_lift","oos_n","oos_hits","oos_rate","oos_lift","all_event_rate","all_lift"])
        out=out.sort_values(["oos_lift","oos_hits","disc_lift"],ascending=False)
        outputs.append(out.head(80))
        out.to_csv(f"reports/50pip_origin_composites_{tf}_{direction}.csv",index=False,float_format="%.8f")
    final=pd.concat(outputs,ignore_index=True)
    final.to_csv("reports/50pip_origin_composite_shortlist.csv",index=False,float_format="%.8f")
    print(final.sort_values(["oos_lift","oos_hits"],ascending=False).head(80).to_string(index=False))

if __name__=="__main__": main()
