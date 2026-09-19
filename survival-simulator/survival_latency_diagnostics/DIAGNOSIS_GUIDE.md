# Survival Simulator latency diagnosis guide

This package measures latency without changing the C167-v2 policy logic.

The central idea is to separate four layers:

```text
competition platform waiting time
        = remote/network delay
        + in-process FastAPI handling

in-process FastAPI handling
        = request parsing / scheduling / locking
        + SurvivalController.decide_step()
        + response validation / serialization
```

The diagnostic server records both the full in-process `/predict` time and the controller-only time. It also counts the exact number of `/predict` requests, so after a validation you can compare your host's cumulative processing time against the platform's 600-second accumulated-wait threshold.

## Files

```text
diagnostic_agent_server.py          diagnostic wrapper around your unchanged agent_server.py

diagnostics/
├── perf_monitor.py                 cumulative + rolling timing statistics
├── benchmark_http.py               localhost/external HTTP RTT benchmark
├── interpret_perf.py               converts measurements into a bottleneck diagnosis
├── run_diagnostic_server.sh        starts Uvicorn correctly for the stateful controller
├── monitor_host.sh                 pidstat/vmstat/mpstat resource logging
└── capture_traffic.sh              optional tcpdump packet capture
```

## Important deployment rule

Run **one Uvicorn worker only**. Your policy keeps persistent per-agent memory. Multiple independent Uvicorn workers would own different controller states and requests could be routed to different memories.

Do not run the HPO/evaluation workers on the same machine during a competition validation. They can steal CPU time from the serving process.

---

# Level 0 — install the diagnostic wrapper

Copy this package into the root of your existing `survival-simulator` checkout so that the layout is:

```text
survival-simulator/
├── agent_server.py                 your existing production server — unchanged
├── diagnostic_agent_server.py      new
├── survival_policy/                your existing controller
├── diagnostics/                    new
└── ...
```

The wrapper imports your existing `agent_server.py`, uses its exact FastAPI app/config/controller initialization, and only adds timers and diagnostic endpoints.

Optionally protect the diagnostic statistics endpoint with a token:

```bash
export SURVIVAL_DIAG_TOKEN="$(openssl rand -hex 16)"
```

Keep that token private and do not commit it.

Set the same production config you normally use, for example:

```bash
export SURVIVAL_POLICY_CONFIG=config/C167_v2_hpo_base.json
```

If your production config is elsewhere, use that path instead.

Start the server:

```bash
./diagnostics/run_diagnostic_server.sh
```

Equivalent explicit command:

```bash
python -m uvicorn diagnostic_agent_server:app \
  --host 0.0.0.0 \
  --port 9052 \
  --workers 1 \
  --no-access-log \
  --log-level warning
```

### Verify the server

In another terminal on the same machine:

```bash
curl -s http://127.0.0.1:9052/diag/ping
```

Expected:

```json
{"ok":true}
```

Read performance data:

```bash
curl -s \
  -H "x-diagnostic-token: $SURVIVAL_DIAG_TOKEN" \
  http://127.0.0.1:9052/diag/perf | python -m json.tool
```

Before any validation, `predict_requests` should be zero or close to zero.

---

# Level 1 — measure the local HTTP stack

This determines whether Uvicorn/FastAPI/OS loopback itself is unexpectedly slow.

Run from the **same Ubuntu server**:

```bash
python diagnostics/benchmark_http.py \
  http://127.0.0.1:9052/diag/ping \
  --requests 2000 \
  --warmup 50 \
  --output diagnostics/localhost_keepalive.json
```

This uses one persistent HTTP connection.

Then test the cost of opening a new TCP connection for every request:

```bash
python diagnostics/benchmark_http.py \
  http://127.0.0.1:9052/diag/ping \
  --requests 500 \
  --warmup 20 \
  --new-connection-each-request \
  --output diagnostics/localhost_new_connection.json
```

### Interpretation

The exact absolute number depends on the machine, but localhost should normally be tiny relative to tens of milliseconds. The important comparison is later:

```text
external HTTP RTT  >> localhost HTTP RTT
```

If localhost is already tens of milliseconds, investigate the host before blaming the network.

---

# Level 2 — measure controller and full `/predict` time during a real run

Do **not** benchmark `/predict` repeatedly on your live stateful controller just to create synthetic load. The most useful measurement is the actual competition validation traffic.

Immediately before pressing validation on the Nordic AI Cup platform, reset counters:

```bash
curl -s -X POST \
  -H "x-diagnostic-token: $SURVIVAL_DIAG_TOKEN" \
  http://127.0.0.1:9052/diag/reset | python -m json.tool
```

Confirm:

```bash
curl -s \
  -H "x-diagnostic-token: $SURVIVAL_DIAG_TOKEN" \
  http://127.0.0.1:9052/diag/perf | python -m json.tool
```

You want:

```text
predict_requests = 0
```

Now run **one platform validation**.

After the validation finishes or fails, immediately save the measurements:

```bash
curl -s \
  -H "x-diagnostic-token: $SURVIVAL_DIAG_TOKEN" \
  http://127.0.0.1:9052/diag/perf \
  > diagnostics/perf_after_validation.json
```

Also force the on-server snapshot:

```bash
curl -s -X POST \
  -H "x-diagnostic-token: $SURVIVAL_DIAG_TOKEN" \
  http://127.0.0.1:9052/diag/save
```

The wrapper automatically writes a snapshot every 250 `/predict` requests by default, so partial data survives even if you forget the final command.

Inspect:

```bash
cat diagnostics/perf_after_validation.json | python -m json.tool
```

Pay attention to:

```text
predict_requests
app_predict.wall_total_s
app_predict.wall_mean_all_ms
app_predict.rolling_wall.p95_ms
app_predict.rolling_wall.p99_ms
controller.wall_total_s
controller.wall_mean_all_ms
controller.rolling_wall.p95_ms
derived.app_minus_controller_mean_ms
```

### What each field means

`app_predict.wall_total_s`
: Total time spent on your server from entry into `/predict` middleware until FastAPI returned the response object. It includes parsing, scheduling, lock wait, controller computation, response validation, and serialization. It excludes Internet travel before the request reaches your machine and after the response leaves it.

`controller.wall_total_s`
: Time spent specifically inside `SurvivalController.decide_step()`.

`derived.app_minus_controller_mean_ms`
: Approximate non-controller application overhead per request. This includes FastAPI/Pydantic, lock wait outside `decide_step`, serialization, and scheduling.

`controller.rolling_cpu`
: CPU time used by the process while the controller call ran. With a single mostly-idle serving process this helps distinguish actual computation from scheduling stalls.

---

# Level 3 — compare against the 600-second platform threshold

If the platform reports only:

```text
Accumulated wait time for agent exceeded 600 seconds
```

use 600 seconds as a **lower bound**, because the exact accumulated total may be slightly larger.

Run:

```bash
python diagnostics/interpret_perf.py \
  diagnostics/perf_after_validation.json \
  --platform-wait-seconds 600
```

Example A:

```text
predict requests recorded : 11200
platform wait used        : 600.00 s
in-process /predict total : 32.00 s
controller total          : 21.00 s
outside-process lower bound: 568.00 s
outside lower bound/request: 50.71 ms
```

This would be very strong evidence that your controller is **not** the main bottleneck. Your whole process used only ~32 seconds while the platform accumulated at least 600 seconds.

Example B:

```text
in-process /predict total : 510 s
controller total          : 470 s
```

That would indicate the policy itself is the main problem and should be profiled/optimized.

Example C:

```text
in-process /predict total : 480 s
controller total          : 80 s
```

That means the controller is not especially slow, but something else **inside your server process** is: lock waiting, request parsing, serialization, event-loop/thread scheduling, logging, or host contention.

---

# Level 4 — measure the real external network path

Run these tests from a **different machine on a different network**. A cloud VM or another university machine is ideal. Do not rely only on calling the public IP from the server itself; hairpin routing can be misleading.

Copy just `diagnostics/benchmark_http.py` to that machine, then run:

```bash
python benchmark_http.py \
  http://130.225.37.30:9052/diag/ping \
  --requests 1000 \
  --warmup 25 \
  --output external_keepalive.json
```

Then:

```bash
python benchmark_http.py \
  http://130.225.37.30:9052/diag/ping \
  --requests 300 \
  --warmup 10 \
  --new-connection-each-request \
  --output external_new_connection.json
```

Also run:

```bash
ping -c 50 130.225.37.30
```

If available:

```bash
mtr -rwzc 100 130.225.37.30
```

or:

```bash
traceroute 130.225.37.30
```

### Why both keep-alive and new-connection tests matter

If persistent external requests average 15 ms but a fresh TCP connection averages 45 ms, connection establishment is expensive. If the competition client does not reuse its TCP connection, that alone can consume a large fraction of the cumulative waiting budget.

Compare:

```text
external_keepalive.mean_ms - localhost_keepalive.mean_ms
```

This is a useful approximation of the network path contribution.

Do the same for the new-connection measurements.

---

# Level 5 — check whether the Ubuntu host is delaying the process

Install host-monitoring tools if necessary:

```bash
sudo apt update
sudo apt install -y sysstat
```

Before the platform validation, in a separate terminal:

```bash
./diagnostics/monitor_host.sh
```

Leave it running until validation finishes, then press Ctrl-C.

It writes:

```text
diagnostics/host_monitor/
├── pidstat.log
├── vmstat.log
├── mpstat.log
├── process_start.txt
├── start_time.txt
└── end_time.txt
```

### Look for

In `pidstat.log`:

- consistently high `%CPU` for the serving process;
- high voluntary/non-voluntary context switching;
- increasing RSS;
- CPU migration/scheduling anomalies.

In `vmstat.log`:

- `si`/`so` non-zero: swapping, bad;
- `r` much larger than available CPUs: CPU contention;
- high `wa`: I/O wait;
- low idle CPU while validation runs.

In `mpstat.log`:

- one serving core pinned near 100%;
- `%steal` on a VM: the hypervisor is depriving your VM of CPU;
- system-wide saturation.

### Very useful controller CPU-vs-wall diagnostic

If `/diag/perf` says:

```text
controller wall mean = 25 ms
controller CPU mean  = 4 ms
```

then the controller is mostly **waiting to be scheduled**, not computing for 25 ms.

If instead:

```text
controller wall mean = 25 ms
controller CPU mean  = 24 ms
```

then the controller itself is computationally expensive.

---

# Level 6 — packet capture: strongest server-side evidence

This is optional but valuable if you need to prove whether the delay is outside your machine.

Before validation:

```bash
./diagnostics/capture_traffic.sh diagnostics/validation_9052.pcap
```

Run the validation, then stop capture with Ctrl-C.

The PCAP contains timestamps for packets arriving at and leaving your host. It therefore establishes the time interval visible **at your NIC**, independent of what the competition UI reports.

Useful commands after installing Wireshark/tshark:

```bash
sudo apt install -y tshark
```

List HTTP requests if the traffic is plain HTTP:

```bash
tshark -r diagnostics/validation_9052.pcap \
  -Y 'http.request' \
  -T fields \
  -e frame.time_epoch \
  -e ip.src \
  -e tcp.stream \
  -e http.request.method \
  -e http.request.uri \
  | head -50
```

List HTTP responses:

```bash
tshark -r diagnostics/validation_9052.pcap \
  -Y 'http.response' \
  -T fields \
  -e frame.time_epoch \
  -e ip.dst \
  -e tcp.stream \
  -e http.response.code \
  | head -50
```

Run the included quick connection analysis:

```bash
./diagnostics/analyze_pcap.sh diagnostics/validation_9052.pcap
```

This reports incoming TCP SYNs, `/predict` requests, HTTP responses, and retransmissions. A particularly important result is:

```text
incoming TCP SYNs ≈ /predict requests
```

which indicates that the remote caller is probably creating a new TCP connection for most ticks. If there are only a few SYNs but thousands of `/predict` calls, keep-alive is being reused correctly.

If the request reaches your NIC and your response leaves a few milliseconds later, but the platform counts tens of milliseconds, the missing time is not controller computation.

---

# Level 7 — decision tree

Use this order; do not optimize code before you know the layer.

```text
1. Is app_predict mean/p95 small?
   |
   +-- YES --> Is external RTT much larger than localhost RTT?
   |              |
   |              +-- YES --> network / remote path / connection setup dominates
   |              |
   |              +-- NO  --> platform-side scheduling or protocol behavior is suspect
   |
   +-- NO --> Is controller time most of app_predict time?
                  |
                  +-- YES --> controller is the optimization target
                  |
                  +-- NO  --> FastAPI/Pydantic/lock/host scheduling is the target
```

The most decisive comparison after a failed platform validation is:

```text
platform threshold:                 >= 600 s
measured total inside your process: app_predict.wall_total_s
```

If the second number is dramatically smaller than 600 seconds, speeding up the policy cannot by itself solve the cumulative-wait failure.

---

# Recommended exact validation procedure

Use this sequence for one clean diagnostic validation.

### Terminal 1 — diagnostic server

```bash
cd /path/to/Nordic-AI-Cup-2026/survival-simulator
export SURVIVAL_POLICY_CONFIG=config/C167_v2_hpo_base.json
export SURVIVAL_DIAG_TOKEN="your-private-random-token"
export SURVIVAL_DIAG_SNAPSHOT_EVERY=250
./diagnostics/run_diagnostic_server.sh
```

### Terminal 2 — local baseline and reset

```bash
python diagnostics/benchmark_http.py \
  http://127.0.0.1:9052/diag/ping \
  --requests 2000 \
  --warmup 50 \
  --output diagnostics/localhost_keepalive.json

curl -s -X POST \
  -H "x-diagnostic-token: $SURVIVAL_DIAG_TOKEN" \
  http://127.0.0.1:9052/diag/reset
```

### Terminal 3 — host monitoring

```bash
./diagnostics/monitor_host.sh
```

### Terminal 4 — optional packet capture

```bash
./diagnostics/capture_traffic.sh diagnostics/validation_9052.pcap
```

### Then

1. Submit/run exactly one platform validation.
2. Do not run HPO or local simulations simultaneously.
3. After validation stops, save `/diag/perf`.

```bash
curl -s \
  -H "x-diagnostic-token: $SURVIVAL_DIAG_TOKEN" \
  http://127.0.0.1:9052/diag/perf \
  > diagnostics/perf_after_validation.json
```

4. Interpret it:

```bash
python diagnostics/interpret_perf.py \
  diagnostics/perf_after_validation.json \
  --platform-wait-seconds 600
```

5. From a separate external machine, run the keep-alive and new-connection benchmarks against `http://130.225.37.30:9052/diag/ping`.

With these four data sources — in-process timings, localhost RTT, external RTT, and host resource logs — the bottleneck should be identifiable without changing the C167-v2 policy.

---

# After diagnosis

Do not leave the diagnostic wrapper exposed indefinitely. Once the bottleneck is identified, return to the normal production command:

```bash
python -m uvicorn agent_server:app \
  --host 0.0.0.0 \
  --port 9052 \
  --workers 1 \
  --no-access-log \
  --log-level warning
```

If the diagnostic results show that the controller is the bottleneck, optimize the controller next while preserving behavior. If the results show that most waiting time lies outside `app_predict.wall_total_s`, focus on network placement/routing/connection behavior instead.
