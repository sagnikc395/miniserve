"""Hand-rolled Prometheus metrics. See docs/MONITORING.md §2-4.

Histograms keep cumulative bucket counts plus `_sum` / `_count`, exactly the
three series Prometheus expects from the text format:
`<name>_bucket{le="..."}` (cumulative!), `<name>_sum`, `<name>_count`.
"""

import time
from bisect import bisect_left

# Bucket edges are fixed at import and cannot be changed retroactively, so err
# wide: values past the last finite edge still land in +Inf and _sum.
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
TOKEN_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1)


class Histogram:
    __slots__ = ("buckets", "counts", "count", "sum")

    def __init__(self, buckets) -> None:
        self.buckets = buckets
        # cumulative: counts[i] is the number of observations <= buckets[i]
        self.counts = [0] * len(buckets)
        self.count = 0
        self.sum = 0.0

    def observe(self, value: float) -> None:
        # every bucket whose edge is >= the value includes it (not just the
        # first matching one); bisect_left so a value exactly on an edge is in
        # the bucket whose le equals the value
        i = bisect_left(self.buckets, value)
        for j in range(i, len(self.counts)):
            self.counts[j] += 1
        self.count += 1
        self.sum += value


class EngineMetrics:
    """Latency histograms and per-reason finish counters, held on the Engine."""

    def __init__(self) -> None:
        self.queue_time = Histogram(LATENCY_BUCKETS)
        self.time_to_first_token = Histogram(LATENCY_BUCKETS)
        self.e2e_latency = Histogram(LATENCY_BUCKETS)
        self.time_per_output_token = Histogram(TOKEN_BUCKETS)
        self.inter_token_latency = Histogram(TOKEN_BUCKETS)
        self.step_duration = Histogram(TOKEN_BUCKETS)

        # finish_reason -> count, exported as label values
        self.finished: dict[str, int] = {}

    def observe_first_token(self, req, now: float) -> None:
        req.first_token = req.last_token = now
        if req.first_scheduled is not None:
            # queue time is first_scheduled - arrival (§2); it is observed
            # here because the histogram lives on the engine, not the scheduler
            self.queue_time.observe(req.first_scheduled - req.arrival)
        self.time_to_first_token.observe(now - req.arrival)

    def observe_next_token(self, req, now: float) -> None:
        self.inter_token_latency.observe(now - req.last_token)
        req.last_token = now

    def observe_finish(self, req, now: float) -> None:
        self.e2e_latency.observe(now - req.arrival)
        if req.first_token is not None and len(req.output_ids) >= 2:
            self.time_per_output_token.observe(
                (now - req.first_token) / (len(req.output_ids) - 1)
            )
        reason = req.finish_reason or "error"
        self.finished[reason] = self.finished.get(reason, 0) + 1

    def render(self, label: str) -> list[str]:
        lines = []
        for name, hist in (
            ("miniserve_request_queue_time_seconds", self.queue_time),
            ("miniserve_time_to_first_token_seconds", self.time_to_first_token),
            ("miniserve_e2e_request_latency_seconds", self.e2e_latency),
            ("miniserve_time_per_output_token_seconds", self.time_per_output_token),
            ("miniserve_inter_token_latency_seconds", self.inter_token_latency),
            ("miniserve_step_duration_seconds", self.step_duration),
        ):
            lines.extend(render_histogram(name, hist, label))
        return lines


def render_histogram(name: str, hist: Histogram, label: str) -> list[str]:
    """Render one histogram family; `label` is `{model_name="..."}`."""
    head = label[:-1] + ",le="  # '{model_name="...",le='
    lines = [f"# TYPE {name} histogram"]
    for edge, cnt in zip(hist.buckets, hist.counts):
        lines.append(f'{name}_bucket{head}"{edge}"}} {cnt}')
    lines.append(f'{name}_bucket{head}"+Inf"}} {hist.count}')
    lines.append(f"{name}_sum{label} {hist.sum!r}")
    lines.append(f"{name}_count{label} {hist.count}")
    return lines
