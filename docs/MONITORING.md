# Monitoring with Prometheus and Grafana

Three boxes, one direction of data flow:

```
miniserve (native, needs MPS)          Docker
┌──────────────────────────┐    ┌────────────┐    ┌─────────┐
│ Engine ── counters,      │    │ Prometheus │    │ Grafana │
│          gauges,         │◄───┤  scrapes   │◄───┤ queries │
│          histograms      │ 5s │  /metrics  │    │ PromQL  │
│ GET /metrics (text)      │    │  stores    │    │  draws  │
│ GET /stats   (JSON)      │    └────────────┘    └─────────┘
└──────────────────────────┘
```

Prometheus pulls; the engine never pushes and never talks to Grafana. So the
whole dashboard stack lives in `deploy/` and nothing in `miniserve/` imports it.
If Prometheus is down, serving is unaffected.

## 1. What already works

`miniserve/api/routers/metrics.py` serves both endpoints today:

- `GET /stats` — `Engine.stats()` as JSON, which the TUI reads.
- `GET /metrics` — the same numbers rendered as Prometheus text, by hand. There is no `prometheus_client` dependency, and there doesn't need to be: the engine already keeps running totals in `Engine.stats()`, so the exporter is a loop over two dicts.

Exported now:

| Counters | Gauges |
|---|---|
| `miniserve_requests_total` | `miniserve_num_requests_running` |
| `miniserve_requests_finished_total` | `miniserve_num_requests_waiting` |
| `miniserve_prompt_tokens_total` | `miniserve_kv_cache_usage_ratio` |
| `miniserve_generation_tokens_total` | `miniserve_kv_cache_used_blocks` |
| `miniserve_prefix_cache_hit_tokens_total` | `miniserve_kv_cache_total_blocks` |
| `miniserve_preemptions_total` | `miniserve_step_batched_tokens` |
| | `miniserve_uptime_seconds` |

All carry one label, `model_name`.

That is enough for throughput, saturation and cache panels. What's missing is
latency: nothing in `Request` records a timestamp, so there is no TTFT, no
TPOT, no end-to-end. Two things to add, in this order:

1. timestamps on `Request` plus six histograms in the exporter (§3, §4);
2. the `deploy/` stack that scrapes and draws it (§5).

## 2. Metric definitions

Pin these down before touching code, because they decide where the timestamps go.

| Metric | Definition | Recorded |
|---|---|---|
| Queue time | first scheduled − arrival | Scheduler admits the request |
| TTFT | first output token − arrival | First committed token |
| ITL | gap between consecutive output tokens | Every committed token after the first |
| TPOT | (finish − first token) / (n_output − 1) | On finish, needs ≥ 2 output tokens |
| E2E latency | finish − arrival | On finish |
| Step time | duration of one `Engine.step` | Every step |

Timing accuracy on MPS: GPU work is asynchronous, so a `perf_counter()` taken
right after the forward call measures enqueue, not execution. The sampler ends
in `.tolist()`, which forces a sync, so a timestamp taken in `Engine.step`
*after* sampling reflects real completion. Take them there and nowhere else.

## 3. Recording the timestamps

Four fields on `Request` (`miniserve/core/sequence.py`), all `perf_counter`
values, so only differences between them mean anything:

```
arrival          set in __init__
first_scheduled  set once, on first admission
first_token      set on the first committed token
last_token       moved forward on every committed token
```

Where each one gets written:

- **arrival** — `Request.__init__`, so it covers the scheduler queue even if
  `add_request` is slow. Tokenization in the HTTP handler happens before this,
  which is one reason to also measure client-side (§7).
- **first_scheduled** — `Scheduler.schedule`, in the waiting-admission branch
  right after `self.waiting.popleft()`. Guard it with `if ... is None`. A
  preempted request goes back onto `waiting` and is admitted again; overwriting
  the field would hide the original queueing delay and report near-zero queue
  times under load, which is exactly when you need the number.
- **first_token / last_token** — `Engine.step`, inside the per-request loop,
  after the `if not s.completes_prompt: continue` guard. Tokens sampled from a
  chunked prefill that stops mid-prompt are discarded, so counting them would
  report a TTFT that never happened. First token sets `first_token` and observes
  TTFT and queue time; later tokens observe ITL as `now - last_token`.
- **on finish** — where `req.finished` is already checked, observe E2E, and TPOT
  when `len(req.output_ids) >= 2`. Skip single-token outputs instead of
  recording a zero; a zero drags the low bucket down and the division is
  undefined anyway.

Take `now = time.perf_counter()` once per step and reuse it for every request in
the batch. They did all finish in the same forward pass, so per-request clock
reads only add noise.

If speculative decoding lands later (see `docs/SPECULATIVE_DECODING.md`),
several tokens commit in one step sharing a timestamp, so ITL shows a mix of
zeros and larger gaps. That's correct, not a bug — TPOT stays the comparable
metric.

## 4. Exporting histograms

Keep the hand-rolled exporter. A Prometheus histogram is three series per
metric, and nothing more:

```
<name>_bucket{le="0.025"}   cumulative count of observations <= 0.025
<name>_bucket{le="+Inf"}    total count, must equal _count
<name>_sum                  sum of observed values
<name>_count                number of observations
```

Cumulative is the part people get wrong: an observation increments *every*
bucket whose edge it is at or below, not just the first matching one. Get that
backwards and `histogram_quantile` returns nonsense rather than an error.

A small `miniserve/metrics.py` carries its weight here: a `Histogram` class with
`observe()`, two bucket tuples, and the render helpers. Suggested shape:

```
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
TOKEN_BUCKETS   = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1)
```

| Series | Buckets |
|---|---|
| `miniserve_request_queue_time_seconds` | latency |
| `miniserve_time_to_first_token_seconds` | latency |
| `miniserve_e2e_request_latency_seconds` | latency |
| `miniserve_time_per_output_token_seconds` | token |
| `miniserve_inter_token_latency_seconds` | token |
| `miniserve_step_duration_seconds` | token |

Hold them on the engine (`self.metrics = EngineMetrics()`) so the exporter can
reach them through `aeng.engine`, the way it already reaches `stats()`. Bucket
edges are fixed at import and cannot be changed retroactively, so err wide:
a value past the last finite edge still lands in `+Inf` and in `_sum`, and the
mean stays right even when the p99 reads as "off the top".

The engine thread updates these while FastAPI renders them from the event loop.
No lock needed — a scrape landing mid-update reports one bucket count behind,
which a `rate()` over a 1m window cannot even see.

Two schema notes while you are in there:

- Add a `reason` label to `miniserve_requests_finished_total` (`stop`, `length`,
  `abort`, `error`). Distinguishing a healthy finish from a truncation is worth
  one extra label, and it needs a per-reason counter in `Engine.stats()`.
- `miniserve_step_batched_tokens` is a gauge of the *last* step. Fine for a live
  "is the batch full" readout, useless for a distribution, since a scrape every
  5s sees one step in several hundred. Make it a histogram too if batch fill
  matters; that is the one metric where the gauge actively misleads.

Resist adding labels beyond `model_name` and `reason`. Every combination is a
separate time series, and a per-request-id label would turn this into a
cardinality bomb within one benchmark run.

## 5. The deploy stack

The server runs natively because it needs MPS; only Prometheus and Grafana go in
Docker. They reach the host through `host.docker.internal`.

```
deploy/
  docker-compose.yml
  prometheus.yml
  grafana/
    provisioning/datasources/prometheus.yml
    provisioning/dashboards/miniserve.yml
    dashboards/miniserve.json
```

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

Anonymous admin is deliberate: this is a laptop dev stack, so skipping the login
screen is worth more than the access control. Don't copy that compose file
anywhere a network can reach it.

`miniserve.json` is painful to hand-write and the schema is effectively
undocumented. Build the panels once in the Grafana UI, export from
**Dashboard settings → JSON Model**, and commit the result; provisioning
reloads it on every start. vLLM's dashboard JSON is a faster starting point than
an empty canvas — clone it and rewrite the `vllm:` prefixes to `miniserve_`.

## 6. Panels

With no label filter yet, add a Grafana variable `model_name` from
`label_values(miniserve_num_requests_running, model_name)` and filter every
query on it.

| Panel | Type | PromQL |
|---|---|---|
| TTFT p50/p95/p99 | Time series | `histogram_quantile(0.95, sum by (le) (rate(miniserve_time_to_first_token_seconds_bucket[1m])))` |
| TPOT p50/p95 | Time series | same pattern on `miniserve_time_per_output_token_seconds_bucket` |
| E2E p95 | Time series | same pattern on `miniserve_e2e_request_latency_seconds_bucket` |
| Queue time p95 | Time series | same pattern on `miniserve_request_queue_time_seconds_bucket` |
| Mean TTFT | Stat | `rate(miniserve_time_to_first_token_seconds_sum[1m]) / rate(miniserve_time_to_first_token_seconds_count[1m])` |
| Output TPS | Time series | `rate(miniserve_generation_tokens_total[1m])` |
| Prompt TPS | Time series | `rate(miniserve_prompt_tokens_total[1m])` |
| Request rate | Time series | `rate(miniserve_requests_finished_total[1m])` |
| Finish reasons | Time series | `sum by (reason) (rate(miniserve_requests_finished_total[1m]))` |
| Running / waiting | Time series | `miniserve_num_requests_running`, `miniserve_num_requests_waiting` |
| KV cache usage | Gauge | `miniserve_kv_cache_usage_ratio` |
| Prefix cache hit rate | Stat | `rate(miniserve_prefix_cache_hit_tokens_total[5m]) / rate(miniserve_prompt_tokens_total[5m])` |
| Step time p95 | Time series | same pattern on `miniserve_step_duration_seconds_bucket` |
| Batch fill | Time series | `miniserve_step_batched_tokens` |
| Preemptions | Time series | `rate(miniserve_preemptions_total[1m])` |

Two things worth remembering:

`sum by (le) (rate(..._bucket[1m]))` is the fixed incantation for a percentile.
`rate` first, then `sum by (le)`, then `histogram_quantile`. Any other order is
wrong, and the wrong order still returns numbers.

Keep the `rate()` window at 1m or more against a 5s scrape — a window under
about 4× the interval leaves gaps in panels during short runs.

Histograms cannot be correlated after the fact, so "TTFT by prompt length" is
not a query you can write over the metrics above. It needs a coarse
`prompt_bucket` label (`short`/`medium`/`long`) added to the TTFT histogram at
record time. Decide that early; migrating label sets later is the expensive kind
of change.

## 7. Checking it actually works

In order, so a failure points at one thing:

1. `curl -s localhost:8000/metrics | head -40` — the text format, served.
2. `docker run --rm -v "$PWD/deploy/prometheus.yml:/c.yml" --entrypoint promtool prom/prometheus check config /c.yml` — catches a bad scrape config before the container silently starts with it.
3. `docker compose -f deploy/docker-compose.yml up -d`, then `localhost:9090/targets` — the `miniserve` target must read **UP**. Down here is nearly always `host.docker.internal` (server bound to `127.0.0.1` instead of `0.0.0.0`) or the port.
4. `localhost:9090` → query `miniserve_num_requests_running`. A result means scraping and storage both work.
5. Send load (`miniserve/benchmarks/run.py`), then query
   `histogram_quantile(0.95, sum by (le) (rate(miniserve_time_to_first_token_seconds_bucket[1m])))`.
   `NaN` with data present means the `_bucket` series are not cumulative.
6. `localhost:3000` — panels drawn from the same queries.

An empty Grafana panel is almost always one of: the target is down, the `rate()`
window is shorter than the scrape interval, no traffic in the window, or the
`model_name` variable filtering on a value that isn't in the data.

A unit test for the timestamp arithmetic pays for itself early: drive a request
through admit → preempt → readmit → three tokens → finish with a fake clock, and
assert queue time, TTFT, ITL and TPOT exactly. The readmit case is the one that
breaks, and it breaks silently.

## 8. Client-side numbers

Server-side metrics exclude tokenization in the HTTP handler, network and SSE
framing, so they always look better than what a client sees. Grafana is for
watching a live server; comparing runs (prefix cache on/off, block size 16 vs
32, speculation on/off) is better served by a load generator that records
per-request TTFT and chunk arrival times to JSONL, plus a plotting script —
latency CDFs per run, and throughput vs p95 TPOT as you sweep request rate.
A single load level makes any engine look good.

Drive load at a fixed arrival rate (Poisson), not fixed concurrency. Fixed
concurrency waits for the server before sending the next request, so queueing
delay never gets measured — coordinated omission, and it flatters the numbers
by a lot.

## 9. Reading

Shortest path: histogram practices → PromQL cheat sheet → stand up §5 against
the current `/metrics` → clone vLLM's dashboard JSON.

- [Histograms and summaries](https://prometheus.io/docs/practices/histograms/) — the one page to read before §4. Why quantiles over buckets are approximate and why edges must be chosen up front.
- [Metric types](https://prometheus.io/docs/concepts/metric_types/) and [naming](https://prometheus.io/docs/practices/naming/) — where the `_seconds` / `_total` conventions come from.
- [How does a Prometheus histogram work?](https://www.robustperception.io/how-does-a-prometheus-histogram-work/) — Brian Brazil; also read his posts on `rate()` extrapolation and cardinality.
- [PromQL cheat sheet](https://promlabs.com/promql-cheat-sheet/), then [functions](https://prometheus.io/docs/prometheus/latest/querying/functions/).
- [Grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/) — the authority on the YAML in §5. Plus [variables](https://grafana.com/docs/grafana/latest/dashboards/variables/) for `model_name`.
- [vLLM metrics design doc](https://docs.vllm.ai/en/latest/design/metrics.html) — rationale for this metric set, including what they chose *not* to measure.
- vLLM's `examples/online_serving/prometheus_grafana/` and `benchmarks/benchmark_serving.py` — working dashboard JSON, and the reference load generator for §8. Paths move between releases; search the repo.
- Gil Tene, [How NOT to Measure Latency](https://www.youtube.com/watch?v=lJ8ydIuPFeU) — 45 minutes on why §8's arrival process matters.
- [Native histograms](https://prometheus.io/docs/specs/native_histograms/) — auto-scaling buckets, worth knowing about before hand-tuning edges, though fixed buckets stay the safer default here.
