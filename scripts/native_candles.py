#!/usr/bin/env python3
import csv
import lzma
import struct
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone

def native_periods(tf, start, end):
    """Return native Dukascopy candle-file periods covering [start, end)."""
    if tf == "d1":
        last_year = end.year if (end.month, end.day) != (1, 1) else end.year - 1
        for year in range(start.year, last_year + 1):
            base = datetime(year, 1, 1, tzinfo=timezone.utc)
            if base.date() >= end:
                break
            period_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            yield base, period_end, (
                f"https://datafeed.dukascopy.com/datafeed/{{pair}}/"
                f"{year}/BID_candles_day_1.bi5"
            )
    elif tf in ("h1", "h4"):
        cur = date(start.year, start.month, 1)
        while cur < end:
            if cur.month == 12:
                nxt = date(cur.year + 1, 1, 1)
            else:
                nxt = date(cur.year, cur.month + 1, 1)
            base = datetime(cur.year, cur.month, 1, tzinfo=timezone.utc)
            period_end = datetime(nxt.year, nxt.month, 1, tzinfo=timezone.utc)
            month0 = cur.month - 1
            suffix = "BID_candles_hour_1.bi5" if tf == "h1" else "BID_candles_hour_4.bi5"
            yield base, period_end, (
                f"https://datafeed.dukascopy.com/datafeed/{{pair}}/"
                f"{cur.year}/{month0:02d}/{suffix}"
            )
            cur = nxt

def _get(url, attempts=5):
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 fx-research-engine/1.0",
                    "Accept": "*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
                if getattr(resp, "status", 200) == 200 and body:
                    return body
                last_err = f"HTTP {getattr(resp, 'status', '?')} / empty body"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, "HTTP 404"
            last_err = f"HTTP {e.code}: {e.reason}"
        except Exception as e:
            last_err = str(e)
        if attempt < attempts:
            time.sleep(min(30, 5 * attempt))

    alt = url.replace(
        "https://datafeed.dukascopy.com/datafeed",
        "https://www.dukascopy.com/datafeed",
    )
    if alt != url:
        for attempt in range(1, attempts + 1):
            try:
                req = urllib.request.Request(
                    alt,
                    headers={
                        "User-Agent": "Mozilla/5.0 fx-research-engine/1.0",
                        "Accept": "*/*",
                    },
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read()
                    if getattr(resp, "status", 200) == 200 and body:
                        print(f"[NATIVE] alternate host succeeded: {alt}", flush=True)
                        return body
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
            except Exception:
                pass
            if attempt < attempts:
                time.sleep(min(30, 5 * attempt))

    raise RuntimeError(last_err or "native download failed")

def download_native_candles(pair, tf, start, end, out, tmp):
    """Download native H4/D1 .bi5 candles and return a CSV chunk path."""
    if tf not in ("h4", "d1"):
        return False

    final = out / tf / f"{pair}.csv"
    (out / tf).mkdir(parents=True, exist_ok=True)
    scale = 1000 if pair.endswith("jpy") else 100000
    rows = []
    period_failures = 0

    for base, _period_end, url_tpl in native_periods(tf, start, end):
        url = url_tpl.format(pair=pair.upper())
        try:
            result = _get(url)
            if isinstance(result, tuple):
                raw, err = result
                if raw is None:
                    raise RuntimeError(err)
            else:
                raw = result
        except Exception as e:
            print(
                f"[NATIVE-WARN] {pair} {tf} {base.date()} unavailable: {e}",
                flush=True,
            )
            period_failures += 1
            continue

        try:
            data = lzma.decompress(raw)
        except Exception as e:
            print(
                f"[NATIVE-WARN] {pair} {tf} {base.date()} invalid LZMA: {e}",
                flush=True,
            )
            period_failures += 1
            continue

        if len(data) < 24 or len(data) % 24:
            print(
                f"[NATIVE-WARN] {pair} {tf} {base.date()} invalid record length={len(data)}",
                flush=True,
            )
            period_failures += 1
            continue

        base_epoch = int(base.timestamp())
        count = 0
        for off in range(0, len(data), 24):
            sec, o, c, low, high, vol = struct.unpack_from(">IIIIIf", data, off)
            if not (o and c and high and low):
                continue
            epoch = base_epoch + sec
            dt = datetime.fromtimestamp(epoch, timezone.utc)
            if start <= dt.date() < end:
                rows.append({
                    "timestamp": str(epoch * 1000),
                    "open": str(o / scale),
                    "high": str(high / scale),
                    "low": str(low / scale),
                    "close": str(c / scale),
                    "volume": str(vol),
                })
                count += 1

        print(f"[NATIVE] {pair} {tf} {base.date()} rows={count}", flush=True)
        if count == 0:
            period_failures += 1

    if rows and not period_failures:
        tmpdir = tmp / f"native_{pair}_{tf}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        chunk = tmpdir / "native.csv"
        with chunk.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
            )
            w.writeheader()
            w.writerows(rows)
        return chunk

    if tf == "h4":
        h1_rows = []
        h1_failures = 0
        for base, _period_end, url_tpl in native_periods("h1", start, end):
            url = url_tpl.format(pair=pair.upper())
            try:
                result = _get(url)
                if isinstance(result, tuple):
                    raw, err = result
                    if raw is None:
                        raise RuntimeError(err)
                else:
                    raw = result
                data = lzma.decompress(raw)
                if len(data) < 24 or len(data) % 24:
                    raise RuntimeError(f"invalid record length={len(data)}")
            except Exception as e:
                print(f"[H1-WARN] {pair} h1 {base.date()} unavailable: {e}", flush=True)
                h1_failures += 1
                continue

            base_epoch = int(base.timestamp())
            count = 0
            for off in range(0, len(data), 24):
                sec, o, c, low, high, vol = struct.unpack_from(">IIIIIf", data, off)
                if not (o and c and high and low):
                    continue
                epoch = base_epoch + sec
                dt = datetime.fromtimestamp(epoch, timezone.utc)
                if start <= dt.date() < end:
                    h1_rows.append({
                        "timestamp": str(epoch * 1000),
                        "open": str(o / scale),
                        "high": str(high / scale),
                        "low": str(low / scale),
                        "close": str(c / scale),
                        "volume": str(vol),
                    })
                    count += 1
            print(f"[H1] {pair} h1 {base.date()} rows={count}", flush=True)
            if count == 0:
                h1_failures += 1

        if h1_rows and not h1_failures:
            buckets = {}
            for r in h1_rows:
                epoch_ms = int(r["timestamp"])
                bucket = (epoch_ms // (4 * 3600 * 1000)) * (4 * 3600 * 1000)
                buckets.setdefault(bucket, []).append(r)
            h4_rows = []
            for bucket in sorted(buckets):
                rs = sorted(buckets[bucket], key=lambda r: int(r["timestamp"]))
                h4_rows.append({
                    "timestamp": str(bucket),
                    "open": rs[0]["open"],
                    "high": format(max(float(r["high"]) for r in rs), ".10f").rstrip("0").rstrip("."),
                    "low": format(min(float(r["low"]) for r in rs), ".10f").rstrip("0").rstrip("."),
                    "close": rs[-1]["close"],
                    "volume": str(sum(float(r["volume"]) for r in rs)),
                })
            tmpdir = tmp / f"native_{pair}_{tf}"
            tmpdir.mkdir(parents=True, exist_ok=True)
            chunk = tmpdir / "native.csv"
            with chunk.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["timestamp","open","high","low","close","volume"])
                w.writeheader()
                w.writerows(h4_rows)
            return chunk

    return False
