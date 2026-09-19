from __future__ import annotations

import argparse
import http.client
import json
import math
from pathlib import Path
import statistics
import time
from urllib.parse import urlsplit


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    i = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[i]


def make_connection(parts, timeout: float):
    host = parts.hostname
    if host is None:
        raise ValueError("URL must include a hostname")
    port = parts.port
    if parts.scheme == "https":
        return http.client.HTTPSConnection(host, port=port, timeout=timeout)
    if parts.scheme == "http":
        return http.client.HTTPConnection(host, port=port, timeout=timeout)
    raise ValueError("Only http:// and https:// URLs are supported")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent-connection HTTP latency benchmark using only the Python standard library."
    )
    parser.add_argument("url")
    parser.add_argument("--method", choices=("GET", "POST"), default="GET")
    parser.add_argument("--json-file", type=Path, default=None)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--new-connection-each-request", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.requests <= 0:
        raise SystemExit("--requests must be > 0")

    parts = urlsplit(args.url)
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query

    body = None
    headers = {"Connection": "keep-alive", "User-Agent": "survival-latency-diagnostic/1"}
    if args.method == "POST":
        if args.json_file is None:
            raise SystemExit("POST requires --json-file")
        parsed = json.loads(args.json_file.read_text(encoding="utf-8"))
        body = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))

    conn = None
    latencies_ms: list[float] = []
    statuses: dict[int, int] = {}
    failures = 0
    total = args.warmup + args.requests

    for i in range(total):
        if conn is None or args.new_connection_each_request:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            conn = make_connection(parts, args.timeout)

        t0 = time.perf_counter_ns()
        try:
            conn.request(args.method, target, body=body, headers=headers)
            response = conn.getresponse()
            response.read()  # Include complete response transfer in measured RTT.
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            statuses[response.status] = statuses.get(response.status, 0) + 1
            if i >= args.warmup:
                latencies_ms.append(elapsed_ms)
            if response.status >= 500:
                failures += 1
        except Exception as exc:
            failures += 1
            try:
                conn.close()
            except Exception:
                pass
            conn = None
            if i >= args.warmup:
                print(f"request {i - args.warmup}: {type(exc).__name__}: {exc}")

    if conn is not None:
        conn.close()

    if not latencies_ms:
        raise SystemExit("No successful measured requests")

    result = {
        "url": args.url,
        "method": args.method,
        "requests_measured": len(latencies_ms),
        "warmup": args.warmup,
        "keep_alive": not args.new_connection_each_request,
        "failures": failures,
        "statuses": statuses,
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 0.50),
        "p90_ms": percentile(latencies_ms, 0.90),
        "p95_ms": percentile(latencies_ms, 0.95),
        "p99_ms": percentile(latencies_ms, 0.99),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
