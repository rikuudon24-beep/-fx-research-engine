#!/usr/bin/env python3
import csv, json, os, urllib.request, urllib.error

path="reports/current_notification_alerts.csv"
if not os.path.exists(path) or os.path.getsize(path)==0:
    print("No entry alerts.")
    raise SystemExit(0)

token=os.environ["GITHUB_TOKEN"]
repo=os.environ["GITHUB_REPOSITORY"]
base=f"https://api.github.com/repos/{repo}"
headers={"Authorization":f"Bearer {token}","Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"}

def api(url, method="GET", body=None):
    req=urllib.request.Request(url,headers=headers,method=method)
    if body is not None:
        req.data=json.dumps(body).encode()
        req.add_header("Content-Type","application/json")
    with urllib.request.urlopen(req,timeout=30) as r:
        return json.load(r)

issues=api(base+"/issues?state=open&per_page=100")
existing=[i.get("title","") for i in issues if "pull_request" not in i]

with open(path,newline="") as f:
    for r in csv.DictReader(f):
        pair=r["pair"]; direction=r["direction"]; signal=r["signal_candle"]
        title=f"FX ALERT {pair} {direction} {signal}"
        if title in existing:
            print(f"Already published: {title}")
            continue
        body=(
            "## FX entry alert\n\n"
            f"- Pair: {pair}\n"
            f"- Direction: {direction}\n"
            f"- Signal candle (completed H4): {signal}\n"
            f"- Next H4 entry candidate: {r['entry_candidate_h4']}\n"
            "- Rule: frozen H4 state-machine notification rule\n"
            "- Source: Dukascopy bid H4 data refreshed by GitHub Actions\n\n"
            "This issue is a durable notification event. Do not treat it as an execution order."
        )
        api(base+"/issues","POST",{"title":title,"body":body})
        print(f"Published: {title}")
