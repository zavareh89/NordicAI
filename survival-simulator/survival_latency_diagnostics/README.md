# Survival Simulator latency diagnostics

This is an overlay for an existing Survival Simulator solution. It does **not** replace or modify `agent_server.py` or controller logic.

Copy `diagnostic_agent_server.py` and the `diagnostics/` directory into the root of your current `survival-simulator/` checkout, then follow `DIAGNOSIS_GUIDE.md`.

Primary command:

```bash
./diagnostics/run_diagnostic_server.sh
```

The wrapper exposes the same production FastAPI app plus:

- `GET /diag/ping`
- `GET /diag/perf`
- `POST /diag/reset`
- `POST /diag/save`

Use one Uvicorn worker because the controller is stateful.
