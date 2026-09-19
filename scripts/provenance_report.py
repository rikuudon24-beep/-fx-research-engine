#!/usr/bin/env python3
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1]; out=R/"reports"; out.mkdir(exist_ok=True); items=[]
for p in sorted((R/"data/market").glob("*/*.csv")):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    rows=0; first=last=None
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            rows+=1; first=first or r.get("timestamp"); last=r.get("timestamp")
    items.append({"file":str(p.relative_to(R)),"rows":rows,"first_timestamp":first,"last_timestamp":last,"sha256":h.hexdigest()})
(out/"data_provenance.json").write_text(json.dumps({"generated_at_utc":datetime.now(timezone.utc).isoformat(),"items":items},indent=2,ensure_ascii=False))
