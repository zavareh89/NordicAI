from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value, digits=2):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpret /diag/perf against the competition accumulated-wait threshold."
    )
    parser.add_argument("perf_json", type=Path)
    parser.add_argument(
        "--platform-wait-seconds",
        type=float,
        default=600.0,
        help=(
            "Use the exact platform accumulated wait if known. If the message only says "
            "'exceeded 600 seconds', leave this at 600; outside-process estimates are then lower bounds."
        ),
    )
    args = parser.parse_args()

    data = json.loads(args.perf_json.read_text(encoding="utf-8"))
    app = data["app_predict"]
    ctrl = data["controller"]
    derived = data.get("derived", {})
    n = int(data.get("predict_requests", app.get("count", 0)) or 0)
    if n <= 0:
        raise SystemExit("No /predict requests recorded. Reset immediately before a validation and retry.")

    platform_s = max(0.0, args.platform_wait_seconds)
    app_s = float(app.get("wall_total_s", 0.0) or 0.0)
    ctrl_s = float(ctrl.get("wall_total_s", 0.0) or 0.0)
    outside_lower_s = max(0.0, platform_s - app_s)
    app_fraction = app_s / platform_s if platform_s > 0 else 0.0
    outside_per_request_ms = 1000.0 * outside_lower_s / n

    print("=== Survival latency diagnosis ===")
    print(f"predict requests recorded : {n}")
    print(f"platform wait used        : {platform_s:.2f} s")
    print(f"in-process /predict total : {app_s:.2f} s ({100*app_fraction:.1f}% of platform wait)")
    print(f"controller total          : {ctrl_s:.2f} s")
    print(f"outside-process lower bound: {outside_lower_s:.2f} s")
    print(f"outside lower bound/request: {outside_per_request_ms:.2f} ms")
    print()
    print("Per-request in-process:")
    print(f"  app mean/p95/p99/max ms : {fmt(app.get('wall_mean_all_ms'))} / "
          f"{fmt(app.get('rolling_wall', {}).get('p95_ms'))} / "
          f"{fmt(app.get('rolling_wall', {}).get('p99_ms'))} / "
          f"{fmt(app.get('wall_max_all_ms'))}")
    print(f"  controller mean/p95 ms  : {fmt(ctrl.get('wall_mean_all_ms'))} / "
          f"{fmt(ctrl.get('rolling_wall', {}).get('p95_ms'))}")
    print(f"  non-controller mean ms  : {fmt(derived.get('app_minus_controller_mean_ms'))}")
    print()

    if app_fraction < 0.20:
        verdict = (
            "STRONG evidence that code/FastAPI is not the main 600 s bottleneck. "
            "At least ~80% of the threshold lies outside measured in-process handling "
            "(network, remote platform scheduling, TCP behavior, or other external delay)."
        )
    elif app_fraction > 0.70:
        verdict = (
            "STRONG evidence that your host/application contributes most of the waiting budget. "
            "Profile controller vs app-minus-controller next."
        )
    else:
        verdict = (
            "MIXED result. Both in-process handling and external/network delay are material; "
            "use controller/app split, external benchmark, host monitoring, and tcpdump to separate them."
        )
    print("Verdict:")
    print(verdict)
    print()
    print(
        "Important: if the platform only reports 'exceeded 600 seconds', 600 s is a threshold, "
        "not the exact accumulated value. The outside-process number above is therefore a LOWER BOUND."
    )


if __name__ == "__main__":
    main()
