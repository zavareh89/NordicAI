# Diagnostic package verification

The diagnostic wrapper was smoke-tested against a synthetic FastAPI `agent_server.py` and synthetic `SurvivalController`.

Verified:

- diagnostic wrapper imports the existing app rather than replacing it;
- `/diag/ping` returns successfully;
- `/diag/reset` clears counters;
- `/predict` requests are counted;
- full in-process request wall/CPU times are recorded;
- controller-only wall/CPU times are recorded;
- `/diag/perf` reports cumulative and rolling statistics;
- `benchmark_http.py` successfully uses persistent HTTP connections;
- `interpret_perf.py` computes an outside-process lower bound and bottleneck verdict;
- all Python files compile successfully.

The synthetic smoke test intentionally did not exercise the Nordic AI Cup simulator itself because this package changes no controller/simulator logic.
