#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-0}"
OUTDIR="${2:-diagnostics/host_monitor}"
mkdir -p "$OUTDIR"

PID="${SURVIVAL_AGENT_PID:-}"
if [[ -z "$PID" ]]; then
  PID="$(pgrep -n -f 'uvicorn.*diagnostic_agent_server:app|uvicorn.*agent_server:app' || true)"
fi
if [[ -z "$PID" ]]; then
  echo "Could not find the Uvicorn PID. Set SURVIVAL_AGENT_PID explicitly." >&2
  exit 1
fi

echo "Monitoring PID $PID; logs -> $OUTDIR"
date -Is > "$OUTDIR/start_time.txt"
ps -p "$PID" -o pid,ppid,psr,pcpu,pmem,rss,vsz,etime,cmd > "$OUTDIR/process_start.txt"

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null || true
  date -Is > "$OUTDIR/end_time.txt"
}
trap cleanup EXIT INT TERM

if command -v pidstat >/dev/null 2>&1; then
  pidstat -h -r -u -w -p "$PID" 1 > "$OUTDIR/pidstat.log" &
else
  echo "pidstat not installed (sudo apt install sysstat)" > "$OUTDIR/pidstat.log"
fi

if command -v vmstat >/dev/null 2>&1; then
  vmstat -w 1 > "$OUTDIR/vmstat.log" &
fi

if command -v mpstat >/dev/null 2>&1; then
  mpstat -P ALL 1 > "$OUTDIR/mpstat.log" &
fi

if [[ "$DURATION" -gt 0 ]]; then
  sleep "$DURATION"
else
  echo "Press Ctrl-C after the validation finishes."
  while true; do sleep 3600; done
fi
