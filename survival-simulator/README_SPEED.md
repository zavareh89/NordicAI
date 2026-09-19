# Survival Simulator latency patch

This patch intentionally does **not** change `SurvivalController`, planner logic,
parameters, memory, target selection, or action scoring. It only removes avoidable
HTTP/JSON/Pydantic overhead from the `/predict` serving path.

## Install

```bash
python -m pip install -r requirements-speed.txt
```

## Prove semantic equivalence first

From the official `survival-simulator/` directory after copying these files in:

```bash
python test_semantic_equivalence.py \
  --config config/C167_v2_hpo_base.json \
  --ticks 5000 \
  --agents 20
```

For even stronger verification, replay real captured payloads as NDJSON:

```bash
python test_semantic_equivalence.py \
  --config config/C167_v2_hpo_base.json \
  --payloads predict_payloads.ndjson
```

The script creates two independent controllers. The reference receives an official
`StepResponse` Pydantic DTO; the fast path receives the original JSON dict. It requires
identical action objects at every sequential tick.

## Benchmark parsing/controller path

```bash
python benchmark_controller_paths.py \
  --config config/C167_v2_hpo_base.json \
  --ticks 5000 --agents 20
```

## Start low-latency service

```bash
./run_fast_server.sh
```

Do not use multiple Uvicorn workers: the controller is stateful.

## Benchmark local HTTP latency

```bash
python benchmark_http.py \
  --url http://127.0.0.1:9052/predict \
  --requests 5000 --agents 20
```

Run this while the machine is otherwise idle. For competition serving, stop HPO jobs
and reserve at least one physical CPU core for the server.
