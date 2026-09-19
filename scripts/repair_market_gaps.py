#!/usr/bin/env python3
import argparse,csv,json,shutil,subprocess,sys,time
from datetime import date,timedelta,datetime,timezone
from pathlib import Path
from scripts.native_candles import download_native_candles

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
                if r.get("timestamp"):
                    rows[r["timestamp"]]=r
    for fp in files:
        with fp.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("timestamp"):
                    rows[r["timestamp"]]=r
    cols=["timestamp","open","high","low","close","volume"]
    ordered=sorted(rows.values(), key=lambda r:int(r["timestamp"]))
    with final.with_suffix(".tmp").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader()
        for r in ordered:
            o,h,l,c=[float(r[x]) for x in ("open","high","low","close")]
            r["high"]=format(max(o,h,l,c),".10f").rstrip("0").rstrip(".")
            r["low"]=format(min(o,h,l,c),".10f").rstrip("0").rstrip(".")
            w.writerow({c:r.get(c,"") for c in cols})
    final.with_suffix(".tmp").replace(final)

def chunk_has_data(path,start,end):
    if not path.exists() or path.stat().st_size == 0:
        return False,0,None,None
    ts=[]
    try:
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("timestamp"):
                    ts.append(int(r["timestamp"]))
    except Exception:
        return False,0,None,None
    if not ts:
        return False,0,None,None
    first=min(ts); last=max(ts)
    in_range=sum(1 for x in ts if start <= ts_date(x) < end)
    return in_range > 0,in_range,ts_date(first),ts_date(last)

def download_range(pair,tf,start,end,idx):
    # Use small, independently verified chunks. dukascopy-node can return a
    # successful process exit with an empty dataset, so file existence alone
    # is not sufficient evidence that a gap was repaired.
    out=OUT/tf
    out.mkdir(parents=True,exist_ok=True)
    final=out/f"{pair}.csv"
    # Prefer native H4/D1 candle files. dukascopy-node currently has
    # intermittent datafeed failures on many historical chunks.
    if tf in ("h4","d1"):
        try:
            native_chunk=download_native_candles(pair,tf,start,end,OUT,TMP)
            if native_chunk:
                merge(final,[native_chunk])
                print(
                    f"[NATIVE-REPAIRED] {pair} {tf} {start} -> {end}",
                    flush=True,
                )
                return True
        except Exception as e:
            print(f"[NATIVE-WARN] {pair} {tf}: {e}",flush=True)

    cur=start
    chunk_no=0

    while cur < end:
        chunk_end=min(cur+timedelta(days=31),end)
        chunk_no += 1
        tmp=TMP/f"{pair}_{tf}_{idx}_{chunk_no}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True,exist_ok=True)
        chunk_file=tmp/"chunk.csv"

        cmd=[
            "npx","--yes","dukascopy-node@1.46.4",
            "-i",pair,
            "-from",cur.isoformat(),
            "-to",chunk_end.isoformat(),
            "-t",tf,
            "-f","csv",
            "-p","bid",
            "-v",
            "-s",
            "-r","3",
            "-re",
            "-dir",str(tmp),
            "-fn","chunk.csv",
            "-bs","5",
            "-bp","5000",
            "-rp","10000",
            "-ch",
            "-chpath",str(ROOT/".dukascopy-cache-repair")
        ]

        ok=False
        for attempt in range(1,4):
            if chunk_file.exists():
                chunk_file.unlink()
            try:
                subprocess.run(cmd,cwd=ROOT,check=True,timeout=900)
                has_data,count,first,last=chunk_has_data(chunk_file,cur,chunk_end)
                if not has_data:
                    raise RuntimeError(
                        f"empty/out-of-range CSV for {cur}->{chunk_end} "
                        f"(rows_in_range={count}, first={first}, last={last})"
                    )
                merge(final,[chunk_file])
                print(
                    f"[CHUNK] {pair} {tf} {cur}->{chunk_end} "
                    f"rows_in_range={count} source={first}->{last}",
                    flush=True
                )
                ok=True
                break
            except Exception as e:
                print(
                    f"[WARN] {pair} {tf} {cur}->{chunk_end} "
                    f"attempt {attempt}/3: {e}",
                    flush=True
                )
                time.sleep(30*attempt)

        if not ok:
            return False

        cur=chunk_end

    print(f"[REPAIRED] {pair} {tf} {start} -> {end} in {chunk_no} chunks",flush=True)
    return True

def find_gaps(pair,tf,start,end):
    path=OUT/tf/f"{pair}.csv"
    if not path.exists():
        return [("missing",start,end)]
    ts=sorted(load_ts(path))
    if not ts:
        return [("missing",start,end)]
    gaps=[]
    first=ts_date(ts[0])
    last=ts_date(ts[-1])
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
ap.add_argument("--pair",default=None,choices=CFG["pairs"])
ap.add_argument("--timeframe",default=None,choices=CFG["timeframes"])
a=ap.parse_args()
start=date.fromisoformat(a.start)
end=date.fromisoformat(a.end)

tasks=[]
pairs=[a.pair] if a.pair else CFG["pairs"]
timeframes=[a.timeframe] if a.timeframe else CFG["timeframes"]

for pair in pairs:
    for tf in timeframes:
        for kind,s,e in find_gaps(pair,tf,start,end):
            if s < e:
                tasks.append((pair,tf,kind,s,e))

print(f"[INFO] repair tasks={len(tasks)}",flush=True)
for t in tasks:
    print(f"[GAP] {t[0]} {t[1]} {t[2]} {t[3]} -> {t[4]}",flush=True)

failed=[]
for i,(pair,tf,kind,s,e) in enumerate(tasks,1):
    if i > 1:
        time.sleep(8)
    if not download_range(pair,tf,s,e,i):
        failed.append((pair,tf,kind,s,e))

if failed:
    print("[FAIL] download failures:",failed)
    sys.exit(2)

# Re-scan after repair. Never report success merely because the downloader
# exited successfully.
remaining=[]
for pair in pairs:
    for tf in timeframes:
        gaps=find_gaps(pair,tf,start,end)
        for kind,s,e in gaps:
            remaining.append((pair,tf,kind,s,e))

if remaining:
    print("[FAIL] unresolved repairs after verification:")
    for item in remaining:
        print(item)
    sys.exit(2)

print("[OK] gap repair completed and verified",flush=True)
