#!/usr/bin/env python3
import argparse,csv,json,subprocess,sys,time
from datetime import date,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"config/market_data.json").read_text())
OUT=ROOT/"data/market"; OUT.mkdir(parents=True,exist_ok=True)

def normalize_row(r):
    """Keep source prices, but repair sub-tick OHLC envelope violations caused by rounded quotes."""
    try:
        o,h,l,c=[float(r[x]) for x in ("open","high","low","close")]
    except Exception:
        return r,False
    nh=max(o,h,l,c)
    nl=min(o,h,l,c)
    repaired=(nh!=h or nl!=l)
    if repaired:
        # Preserve the original precision while making the OHLC envelope internally consistent.
        r["high"]=format(nh,".10f").rstrip("0").rstrip(".")
        r["low"]=format(nl,".10f").rstrip("0").rstrip(".")
    return r,repaired

def merge(final,files):
    rows={}
    if final.exists():
        with final.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("timestamp"): rows[r["timestamp"]]=r
    for fp in files:
        with fp.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("timestamp"): rows[r["timestamp"]]=r
    if not rows: raise RuntimeError("no usable rows")
    cols=["timestamp","open","high","low","close","volume"]
    ordered=sorted(rows.values(),key=lambda r:int(r["timestamp"]))
    repaired=0
    tmp=final.with_suffix(".tmp")
    with tmp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in ordered:
            r,was_repaired=normalize_row(r)
            repaired += int(was_repaired)
            w.writerow({c:r.get(c,"") for c in cols})
    tmp.replace(final)
    if repaired:
        print(f"[REPAIR] {final.name}: normalized {repaired} OHLC rows",flush=True)

def download(pair,tf,start,end):
    out=OUT/tf; out.mkdir(parents=True,exist_ok=True)
    tmp=ROOT/".download_tmp"/f"{pair}_{tf}"; tmp.mkdir(parents=True,exist_ok=True)
    final=out/f"{pair}.csv"
    for n in range(1,4):
        cmd=["npx","--yes","dukascopy-node@1.50.0","-i",pair,"-from",start.isoformat(),"-to",end.isoformat(),"-t",tf,"-f","csv","-p","bid","-v","-s","-dir",str(tmp),"-bs","5","-bp","1500"]
        try:
            subprocess.run(cmd,cwd=ROOT,check=True,timeout=900)
            files=sorted(tmp.glob("*.csv"),key=lambda p:p.stat().st_mtime,reverse=True)
            if not files: raise RuntimeError("no CSV produced")
            merge(final,files); print(f"[OK] {pair} {tf}"); return True
        except Exception as e:
            print(f"[WARN] {pair} {tf} attempt {n}/3: {e}",flush=True); time.sleep(5*n)
    return False

ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["bootstrap","incremental"],default="incremental"); ap.add_argument("--start"); ap.add_argument("--end"); a=ap.parse_args()
end=date.fromisoformat(a.end) if a.end else date.today()+timedelta(days=1)
start=date.fromisoformat(a.start) if a.start else (date.fromisoformat(CFG["history_start"]) if a.mode=="bootstrap" else date.today()-timedelta(days=CFG["incremental_days"]))
bad=[]
for p in CFG["pairs"]:
    for tf in CFG["timeframes"]:
        if not download(p,tf,start,end): bad.append(f"{p}:{tf}")
if bad: print("[FAIL] "+", ".join(bad)); sys.exit(2)
