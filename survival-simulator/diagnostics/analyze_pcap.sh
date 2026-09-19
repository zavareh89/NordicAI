#!/usr/bin/env bash
set -euo pipefail

PCAP="${1:-diagnostics/validation_9052.pcap}"
PORT="${SURVIVAL_AGENT_PORT:-9052}"

if ! command -v tshark >/dev/null 2>&1; then
  echo "tshark is required: sudo apt install tshark" >&2
  exit 1
fi
if [[ ! -f "$PCAP" ]]; then
  echo "PCAP not found: $PCAP" >&2
  exit 1
fi

count_filter() {
  local filter="$1"
  tshark -r "$PCAP" -Y "$filter" -T fields -e frame.number 2>/dev/null | wc -l | tr -d ' '
}

SYNS="$(count_filter "tcp.dstport == $PORT && tcp.flags.syn == 1 && tcp.flags.ack == 0")"
RETRANS="$(count_filter "tcp.port == $PORT && tcp.analysis.retransmission")"
REQS="$(count_filter "tcp.dstport == $PORT && http.request.uri == \"/predict\"")"
RESP="$(count_filter "tcp.srcport == $PORT && http.response")"

cat <<EOF
=== PCAP quick summary ===
file:                 $PCAP
port:                 $PORT
incoming TCP SYNs:    $SYNS
HTTP /predict reqs:   $REQS
HTTP responses:       $RESP
TCP retransmissions:  $RETRANS

Interpretation:
- If incoming SYNs are close to /predict requests, the remote client is probably opening
  a fresh TCP connection for most ticks; connection setup can be a major cumulative cost.
- If SYNs are tiny relative to /predict requests, keep-alive is being reused.
- Non-trivial retransmissions point to packet loss / network-path problems.
EOF
