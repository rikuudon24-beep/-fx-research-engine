#!/usr/bin/env python3
import csv,json,sys
from datetime import date,timedelta,datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"config/market_data.json").read_text())
start=date.fromisoformat(CFG["history_start"])
end=date.today()+timedelta(days=1)
bad=[]
for pair in CFG["pairs"]:
    for tf in CFG["timeframes"]:
        p=ROOT/"data/market"/tf/f"{pair}.csv"
        if not p.exists():
            bad.append(f"{pair}:{tf}:missing")
            continue
        with p.open(newline="") as f:
            ts=[int(r["timestamp"]) for r in csv.DictReader(f) if r.get("timestamp")]
        ts.sort()
        if not ts:
            bad.append(f"{pair}:{tf}:empty")
            continue
        first=datetime.fromtimestamp(ts[0]/1000,timezone.utc).date()
        last=datetime.fromtimestamp(ts[-1]/1000,timezone.utc).date()
        if first > start+timedelta(days=7):
            bad.append(f"{pair}:{tf}:starts {first}, expected near {start}")
        if last < end-timedelta(days=3):
            bad.append(f"{pair}:{tf}:ends {last}, expected near {end}")
        for a,b in zip(ts,ts[1:]):
            if b-a > 7*86400000:
                da=datetime.fromtimestamp(a/1000,timezone.utc).date()
                db=datetime.fromtimestamp(b/1000,timezone.utc).date()
                bad.append(f"{pair}:{tf}:gap {da} -> {db}")
                break
if bad:
    print("[FAIL] coverage gaps detected")
    print("\n".join(bad))
    sys.exit(2)
print(f"[OK] coverage complete for {len(CFG['pairs'])} pairs x {len(CFG['timeframes'])} timeframes")
