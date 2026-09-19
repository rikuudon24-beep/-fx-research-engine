#!/usr/bin/env python3
import argparse,csv,json,subprocess,sys,time
from datetime import date,timedelta,datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"config/market_data.json").read_text())
OUT=ROOT/"data/market"
TMP=ROOT/".download_tmp_repair"

def ts_date(ms):
    return datetime.fromtimestamp(ms/1000, timezone.utc).date()

def load_ts(path):
    with path.open(newline="") as f:
        rows=csv.DictReader(f)
        return [int(r["timestamp"]) for r in rows if r.get("timestamp")]

def merge(final, files):
    rows={}
    if final.exists():
        with final.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("timestamp"): rows[r["timestamp"]]=r
    for fp in files:
        with fp.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("timestamp"): rows[r["timestamp"]]=r
    cols=["timestamp","open","high","low","close","volume"]
    ordered=sorted(rows.values(), key=lambda r:int(r["timestamp"]))
    with final.with_suffix(".tmp").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in ordered:
            o,h,l,c=[float(r[x]) for x in ("open","high","low","close")]
            r["high"]=format(max(o,h,l,c),".10f").rstrip("0").rstrip(".")
            r["low"]=format(min(o,h,l,c),".10f").rstrip("0").rstrip(".")
            w.writerow({c:r.get(c,"") for c in cols})
    final.with_suffix(".tmp").replace(final)

def download_range(pair,tf,start,end,idx):
    # Download in <=31-day chunks. Large missing ranges can otherwise return
    # incomplete results without a hard downloader error.
    out=OUT/tf; out.mkdir(parents=True,exist_ok=True)
    tmp=TMP/f"{pair}_{tf}_{idx}"; tmp.mkdir(parents=True,exist_ok=True)
    final=out/f"{pair}.csv"
    cur=start
    chunk_no=0
    while cur < end:
        chunk_end=min(cur+timedelta(days=31),end)
        chunk_no += 1
        cmd=["npx","--yes","dukascopy-node@1.50.0","-i",pair,"-from",cur.isoformat(),
             "-to",chunk_end.isoformat(),"-t",tf,"-f","csv","-p","bid","-v","-dir",str(tmp),
             "-bs","5","-bp","1500"]
        ok=False
        for attempt in range(1,4):
            try:
                subprocess.run(cmd,cwd=ROOT,check=True,timeout=900)
                files=sorted(tmp.glob("*.csv"),key=lambda p:p.stat().st_mtime,reverse=True)
                if not files: raise RuntimeError("no CSV produced")
                merge(final,files)
                ok=True
                break
            except Exception as e:
                print(f"[WARN] {pair} {tf} {cur}->{chunk_end} attempt {attempt}/3: {e}",flush=True)
                time.sleep(5*attempt)
        if not ok:
            return False
        cur=chunk_end
    print(f"[REPAIRED] {pair} {tf} {start} -> {end} in {chunk_no} chunks",flush=True)
    return True

def find_gaps(pair,tf,start,end):
    path=OUT/tf/f"{pair}.csv"
    if not path.exists(): return [("missing",start,end)]
    ts=sorted(load_ts(path))
    if not ts: return [("missing",start,end)]
    gaps=[]
    first=ts_date(ts[0]); last=ts_date(ts[-1])
    if first > start + timedelta(days=7):
        gaps.append(("leading",start,first))
    if last < end - timedelta(days=3):
        gaps.append(("trailing",last+timedelta(days=1),end))
    threshold=7*86400000
    for a,b in zip(ts,ts[1:]):
        if b-a > threshold:
            gaps.append(("internal",ts_date(a)+timedelta(days=1),ts_date(b)))
    return gaps

ap=argparse.ArgumentParser()
ap.add_argument("--start",default=CFG["history_start"])
ap.add_argument("--end",default=(date.today()+timedelta(days=1)).isoformat())
a=ap.parse_args()
start=date.fromisoformat(a.start); end=date.fromisoformat(a.end)
tasks=[]
for pair in CFG["pairs"]:
    for tf in CFG["timeframes"]:
        for kind,s,e in find_gaps(pair,tf,start,end):
            if s < e: tasks.append((pair,tf,kind,s,e))
print(f"[INFO] repair tasks={len(tasks)}",flush=True)
for t in tasks: print(f"[GAP] {t[0]} {t[1]} {t[2]} {t[3]} -> {t[4]}",flush=True)
failed=[]
for i,(pair,tf,kind,s,e) in enumerate(tasks,1):
    if not download_range(pair,tf,s,e,i): failed.append((pair,tf,kind,s,e))
if failed:
    print("[FAIL] unresolved repairs:",failed)
    sys.exit(2)
print("[OK] gap repair completed",flush=True)
