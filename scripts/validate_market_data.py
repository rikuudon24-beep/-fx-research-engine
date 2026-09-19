#!/usr/bin/env python3
import csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; files=sorted((ROOT/"data/market").glob("*/*.csv"))
if not files: print("[FAIL] no market data"); sys.exit(2)
bad=[]
for p in files:
    with p.open(newline="") as f: rows=list(csv.DictReader(f))
    if len(rows)<200: bad.append(f"{p}: rows={len(rows)}"); continue
    prev=None
    for r in rows:
        try:
            ts=int(r["timestamp"]); o,h,l,c=[float(r[x]) for x in ("open","high","low","close")]
            if min(o,h,l,c)<=0 or h<max(o,c,l) or l>min(o,c,h): bad.append(f"{p}: invalid OHLC"); break
            if prev is not None and ts<=prev: bad.append(f"{p}: timestamp order/duplicate"); break
            prev=ts
        except Exception: bad.append(f"{p}: malformed row"); break
if bad: print("\n".join(bad)); sys.exit(2)
print(f"[OK] validated {len(files)} market files")
