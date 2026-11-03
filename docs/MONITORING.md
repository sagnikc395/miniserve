Yes. The standard way is the same one vLLM uses: the engine records per-request timestamps, exposes them as **Prometheus** metrics on `/metrics`, Prometheus scrapes that endpoint every few seconds, and **Grafana** reads from Prometheus. Nothing in the engine talks to Grafana directly, so the dashboard stack stays outside your package.

## 1. Where it lives in the layout

```
miniserve/
  metrics/
    __init__.py
    timing.py          # RequestTimes dataclass + derived metrics (pure Python)
    prometheus.py      # metric definitions and a Recorder that updates them
  api/routers/metrics.py   # GET /metrics in Prometheus text format
deploy/
  docker-compose.yml       # Prometheus + Grafana
  prometheus.yml
  grafana/
    provisioning/datasources/prometheus.yml
    provisioning/dashboards/miniserve.yml
    dashboards/miniserve.json
benchmarks/
  load.py                  # async OpenAI-API load generator, client-side timings
tests/unit/test_timing.py
```

Keep `metrics/timing.py` free of `prometheus_client` so it is unit-testable with a fake clock, and keep `prometheus.py` as the only module that imports the client library.

## 2. Definitions

Get these precise first, since they decide where the timestamps go:

| Metric | Definition | Recorded |
|---|---|---|
| Queue time | first scheduled − arrival | Scheduler admits the sequence |
| TTFT | first output token − arrival | First committed token |
| ITL (inter-token latency) | gap between consecutive output tokens | Every committed token |
| TPOT | (e2e − TTFT) / (n_output − 1) | On finish |
| E2E latency | finish − arrival | On finish |
| Prefill time | first token − first scheduled | On first token |
| Throughput (TPS) | output tokens per second, across all requests | Counter, rate computed in Prometheus |
| Context length | prompt tokens, and total tokens at finish | On admit and on finish |

Two engine-level metrics are just as useful for a tinyserve-style engine: **step time** and **tokens per step** (how full each forward pass is), plus gauges for running/waiting sequences, KV cache usage, and prefix cache hit rate.

A note on timing accuracy on MPS: GPU work is asynchronous, so a `time.perf_counter()` right after the forward call measures enqueue time, not execution time. In your engine the sampler ends with `.tolist()`, which forces a sync, so timestamps taken in `Engine.step` after sampling reflect real completion. Keep them there.

## 3. Recording timings

```python
# metrics/timing.py
from dataclasses import dataclass, field
import time

Clock = time.perf_counter

@dataclass
class RequestTimes:
    arrival: float
    first_scheduled: float | None = None
    first_token: float | None = None
    last_token: float | None = None
    finished: float | None = None
    itl: list[float] = field(default_factory=list)

    def on_token(self, now: float) -> None:
        if self.first_token is None:
            self.first_token = now
        else:
            self.itl.append(now - self.last_token)
        self.last_token = now

    @property
    def ttft(self) -> float:  return self.first_token - self.arrival
    @property
    def queue(self) -> float: return self.first_scheduled - self.arrival
    @property
    def e2e(self) -> float:   return self.finished - self.arrival

    def tpot(self, n_out: int) -> float | None:
        return None if n_out < 2 else (self.finished - self.first_token) / (n_out - 1)
```

Add `times: RequestTimes` to `Sequence`. Set `first_scheduled` in the scheduler's admission phase only the first time (a preempted sequence that is readmitted keeps its original value, otherwise preemption hides queueing delay). Call `on_token` in `Engine.step` where tokens are committed. If you add speculative decoding later, several tokens commit in one step with the same timestamp, so ITL naturally shows a mix of zeros and larger gaps; that is correct behaviour, and TPOT is the cleaner metric to compare.

## 4. Prometheus metrics

```python
# metrics/prometheus.py
from prometheus_client import Counter, Gauge, Histogram, Info

LAT = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
TOK_LAT = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1)
LEN = (16, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
L = ["model_name"]

TTFT  = Histogram("miniserve_time_to_first_token_seconds", "TTFT", L, buckets=LAT)
TPOT  = Histogram("miniserve_time_per_output_token_seconds", "TPOT", L, buckets=TOK_LAT)
ITL   = Histogram("miniserve_inter_token_latency_seconds", "ITL", L, buckets=TOK_LAT)
E2E   = Histogram("miniserve_e2e_request_latency_seconds", "E2E", L, buckets=LAT)
QUEUE = Histogram("miniserve_request_queue_time_seconds", "Queue", L, buckets=LAT)
STEP  = Histogram("miniserve_step_duration_seconds", "Engine step", L, buckets=TOK_LAT)

PROMPT_LEN = Histogram("miniserve_request_prompt_tokens", "Prompt length", L, buckets=LEN)
GEN_LEN    = Histogram("miniserve_request_generation_tokens", "Output length", L, buckets=LEN)
STEP_TOK   = Histogram("miniserve_step_batched_tokens", "Tokens per step", L,
                       buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048))

PROMPT_TOK = Counter("miniserve_prompt_tokens", "Prompt tokens", L)
GEN_TOK    = Counter("miniserve_generation_tokens", "Generated tokens", L)
CACHED_TOK = Counter("miniserve_prefix_cache_hit_tokens", "Prompt tokens from cache", L)
FINISHED   = Counter("miniserve_requests_finished", "Finished", L + ["reason"])
PREEMPT    = Counter("miniserve_preemptions", "Preemptions", L)

RUNNING  = Gauge("miniserve_num_requests_running", "Running", L)
WAITING  = Gauge("miniserve_num_requests_waiting", "Waiting", L)
KV_USAGE = Gauge("miniserve_kv_cache_usage_ratio", "Used KV blocks / total", L)
MPS_MEM  = Gauge("miniserve_mps_driver_allocated_bytes", "Metal driver memory", L)

CONFIG = Info("miniserve_engine", "Engine configuration")
```

Counters get a `_total` suffix automatically when exported. A `Recorder` class wraps these, binds `model_name` once with `.labels(...)`, and exposes `on_step(batch, duration, scheduler, kv)` and `on_finish(seq)`. Call it from `Engine.step`; `prometheus_client` is thread-safe, so updating from the engine thread while FastAPI serves `/metrics` from the event loop is fine.

At startup, publish the config so dashboards can compare runs:

```python
CONFIG.info({"model": cfg.model, "dtype": str(cfg.dtype), "backend": cfg.attention_backend,
             "block_size": str(cfg.block_size), "max_num_seqs": str(cfg.max_num_seqs),
             "prefix_caching": str(cfg.enable_prefix_caching)})
```

Expose them with a plain route (mounting `make_asgi_app()` at `/metrics` works too, but redirects to `/metrics/`):

```python
# api/routers/metrics.py
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()

@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

Your current JSON `/metrics` endpoint moves to `/stats` so the two don't collide.

## 5. Prometheus and Grafana on the Mac

Run the server natively (it needs MPS) and only the monitoring stack in Docker. Containers reach the host via `host.docker.internal`.

```yaml
# deploy/docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports: ["9090:9090"]
  grafana:
    image: grafana/grafana
    environment:
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Admin
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    ports: ["3000:3000"]
    depends_on: [prometheus]
```

```yaml
# deploy/prometheus.yml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: miniserve
    static_configs:
      - targets: ["host.docker.internal:8000"]
```

```yaml
# deploy/grafana/provisioning/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
```

```yaml
# deploy/grafana/provisioning/dashboards/miniserve.yml
apiVersion: 1
providers:
  - name: miniserve
    type: file
    options:
      path: /var/lib/grafana/dashboards
```

Then `docker compose -f deploy/docker-compose.yml up -d`, start the server, and open `localhost:3000`. The easiest way to produce `miniserve.json` is to build the panels in the Grafana UI once, export the dashboard as JSON, and commit it; provisioning reloads it on every start.

## 6. Panels and queries

| Panel | Type | PromQL |
|---|---|---|
| TTFT p50 / p95 / p99 | Time series | `histogram_quantile(0.95, sum by (le) (rate(miniserve_time_to_first_token_seconds_bucket[1m])))` |
| TPOT p50 / p95 | Time series | same pattern on `miniserve_time_per_output_token_seconds_bucket` |
| E2E latency p95 | Time series | same pattern on `miniserve_e2e_request_latency_seconds_bucket` |
| Mean TTFT | Stat | `rate(miniserve_time_to_first_token_seconds_sum[1m]) / rate(miniserve_time_to_first_token_seconds_count[1m])` |
| Output TPS | Time series | `rate(miniserve_generation_tokens_total[1m])` |
| Prompt TPS | Time series | `rate(miniserve_prompt_tokens_total[1m])` |
| Request rate | Time series | `sum(rate(miniserve_requests_finished_total[1m]))` |
| Running / waiting | Time series | `miniserve_num_requests_running`, `miniserve_num_requests_waiting` |
| KV cache usage | Gauge | `miniserve_kv_cache_usage_ratio` |
| Prefix cache hit rate | Stat | `rate(miniserve_prefix_cache_hit_tokens_total[5m]) / rate(miniserve_prompt_tokens_total[5m])` |
| Prompt length distribution | Heatmap | `sum by (le) (rate(miniserve_request_prompt_tokens_bucket[1m]))` |
| TTFT vs context length | Heatmap | the TTFT buckets, split by a prompt-length label (see below) |
| Batch fill | Time series | p50 of `miniserve_step_batched_tokens_bucket` |
| Preemptions | Time series | `rate(miniserve_preemptions_total[1m])` |
| MPS memory | Time series | `miniserve_mps_driver_allocated_bytes` |

Add a Grafana variable `model_name` from `label_values(miniserve_num_requests_running, model_name)` and filter every query on it.

For "latency vs context length", a histogram cannot be correlated with another histogram after the fact. If you want TTFT broken down by prompt size, add a coarse `prompt_bucket` label (`short`, `medium`, `long`, using the same edges as your benchmark bins) to the TTFT histogram only. Keep label sets small; every label combination is a separate time series.

Keep the `rate(...[1m])` window at least 4x the scrape interval, otherwise short benchmark runs show gaps.

## 7. Client-side measurement and run comparisons

Server-side metrics exclude tokenization in the HTTP handler, network, and SSE framing, so also measure from the client. `benchmarks/load.py` should send streaming requests through the OpenAI API at a controlled rate (fixed QPS with Poisson arrivals, or a fixed concurrency), record TTFT and per-chunk arrival times for each request, and write one JSON line per request with the run's config.

Grafana is best for watching a live server. For comparing benchmark runs side by side (prefix cache on vs off, block size 16 vs 32, speculation on vs off), the JSONL files plus a small plotting script or notebook are usually clearer: latency CDFs per run, and throughput vs p95 TPOT curves as you sweep the request rate. Those curves are the standard way to compare serving configurations, since any engine can look good at one load level.

Finally, the unit test for `timing.py` is worth writing early: drive a `RequestTimes` with a fake clock through admit, preempt, readmit, three tokens, and finish, and assert queue time, TTFT, TPOT and ITL values exactly.
## 8. Learning resources

Roughly the order worth reading them in. The sections above are the plan; these explain why it is shaped that way.

### Prometheus data model

What trips people up is not Grafana, it is understanding why counters plus `rate()` behave the way they do.

- [Metric types](https://prometheus.io/docs/concepts/metric_types/) — counter vs gauge vs histogram vs summary. Short, and the vocabulary everything else assumes.
- [Histograms and summaries](https://prometheus.io/docs/practices/histograms/) — the most important page for this use case. Why `histogram_quantile` over bucket counters is an approximation, why bucket edges must be chosen up front, and why you cannot average quantiles across instances.
- [How does a Prometheus histogram work?](https://www.robustperception.io/how-does-a-prometheus-histogram-work/) — Brian Brazil's blog. The whole archive is the best Prometheus writing available; also read his posts on `rate()` extrapolation and on cardinality.
- [Metric and label naming](https://prometheus.io/docs/practices/naming/) — the `_seconds` and `_total` suffixes used in §4 come from here.
- [Native histograms](https://prometheus.io/docs/specs/native_histograms/) — exponential, auto-scaling buckets, an alternative to the fixed edges in §4. Worth knowing it exists before hand-tuning TTFT and ITL buckets, though the classic buckets are the safer default for now.

### PromQL

- [Querying basics](https://prometheus.io/docs/prometheus/latest/querying/basics/), then [functions](https://prometheus.io/docs/prometheus/latest/querying/functions/). Focus on `rate`, `irate`, `increase`, `sum by`, `histogram_quantile`.
- [PromLabs PromQL cheat sheet](https://promlabs.com/promql-cheat-sheet/) — they also have a free interactive "Must-Know PromQL" course.

The mental model to internalise: `rate()` needs a window of at least 4x the scrape interval, and `sum by (le) (rate(..._bucket[...]))` is the fixed incantation for percentiles. The queries in §6 are correct; knowing why they are shaped that way is what makes an empty panel quick to debug.

### prometheus_client

- [client_python docs](https://prometheus.github.io/client_python/) — `Counter`/`Gauge`/`Histogram`/`Info`, `generate_latest`, the ASGI app, and the multiprocess mode caveats. Those caveats matter if uvicorn ever runs with more than one worker, since gauges stop making sense.

### Grafana

- [Dashboards](https://grafana.com/docs/grafana/latest/dashboards/) — panels, variables, transformations.
- [Provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/) — the authority on the datasource and dashboard YAML format used in §5.
- [Heatmap panel](https://grafana.com/docs/grafana/latest/panels-visualizations/visualizations/heatmap/) — specifically the heatmap-from-histogram-buckets mode that the prompt length distribution panel needs. Easy to get wrong.
- [Variables](https://grafana.com/docs/grafana/latest/dashboards/variables/) — for the `model_name` template variable.
- [play.grafana.org](https://play.grafana.org) — live sandbox with real data. The fastest way to learn panel options is to edit an existing panel and read its JSON.

Hand-writing dashboard JSON is miserable and the schema is undocumented in practice, so build in the UI and export from Dashboard settings → JSON Model.

### LLM serving metrics

Where the generic observability material stops helping.

- [vLLM metrics design doc](https://docs.vllm.ai/en/latest/design/metrics.html) — the design rationale for the metric set §4 replicates, including what they chose not to measure and which metrics were dropped between v0 and v1.
- vLLM's Grafana example, in the repo under `examples/online_serving/prometheus_grafana/` — the path has moved between releases, so search the repo for `grafana.json` if it 404s. Ships a working compose file, Prometheus config and dashboard JSON. Cloning that JSON and rewriting `vllm:` to `miniserve_` in the queries is a much faster start than building panels from scratch.
- vLLM's `benchmarks/benchmark_serving.py` — the reference implementation of the load generator described in §7: Poisson arrivals, per-request TTFT/ITL/TPOT, percentile reporting. Read it before writing `benchmarks/load.py`; the arrival process and warmup details matter.
- Gil Tene, [How NOT to Measure Latency](https://www.youtube.com/watch?v=lJ8ydIuPFeU) — 45 minutes, and the best time available to spend on this. Coordinated omission is why naive load generators report latencies that are too good, and it bites hard when driving load at a fixed concurrency rather than a fixed arrival rate.
- [Brendan Gregg, the USE method](https://www.brendangregg.com/usemethod.html) — utilisation, saturation, errors as a checklist for what to instrument. The running and waiting gauges and KV usage are the saturation signals; this frames why.

### Longer form

- *Prometheus: Up & Running*, Brian Brazil, O'Reilly, 2nd ed. 2023 — one coherent pass instead of scattered pages.
- Google SRE book, [ch. 6, Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/) — free online. Signals vs causes, and why distributions beat averages.

### A path through it

Histogram practices, then the PromQL cheat sheet, then stand up the compose stack in §5 against a hardcoded `/metrics`, then clone vLLM's dashboard JSON and adapt it. Read the vLLM design doc and Tene's talk before trusting any number the stack produces.

The constraint that shapes the schema, repeating §6 because it is the expensive one to get wrong: Prometheus histograms cannot be correlated with each other after the fact. Any breakdown of TTFT by prompt size needs the label added at record time, and deciding label sets early is cheaper than migrating them later.
