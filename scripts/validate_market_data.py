#!/usr/bin/env python3
import csv,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=sorted((ROOT/"data/market").glob("*/*.csv"))
if not files:
    print("[FAIL] no market data")
    sys.exit(2)

bad=[]
for p in files:
    with p.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)<200:
        bad.append(f"{p}: rows={len(rows)}")
        continue
    prev=None
    for i,r in enumerate(rows, start=2):
        try:
            ts=int(r["timestamp"])
            o,h,l,c=[float(r[x]) for x in ("open","high","low","close")]
            vals=(o,h,l,c)
            valid=all(math.isfinite(v) and v>0 for v in vals) and h>=max(o,c,l) and l<=min(o,c,h)
            if not valid:
                bad.append(f"{p}: invalid OHLC line={i} timestamp={ts} open={o} high={h} low={l} close={c}")
                break
            if prev is not None and ts<=prev:
                bad.append(f"{p}: timestamp order/duplicate line={i} timestamp={ts} prev={prev}")
                break
            prev=ts
        except Exception as e:
            bad.append(f"{p}: malformed row line={i}: {e}")
            break

if bad:
    print("\n".join(bad))
    sys.exit(2)
print(f"[OK] validated {len(files)} market files")
