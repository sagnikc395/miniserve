"""Timestamp arithmetic and histogram rendering, docs/MONITORING.md §3-4, §7.

The readmit case (admit -> preempt -> readmit) is the one that breaks silently:
overwriting `first_scheduled` would report near-zero queue times under load.
"""

import time

import pytest

from miniserve.config import EngineConfig
from miniserve.core.kv_cache import KVCacheManager
from miniserve.core.scheduler import Scheduler
from miniserve.core.sequence import Request
from miniserve.metrics import EngineMetrics, Histogram, render_histogram
from miniserve.sampling import SamplingParams


@pytest.fixture
def clock(monkeypatch):
    """Fake clock: every perf_counter() call advances 0.5s."""
    t = {"now": 1000.0, "tick": 0.5}

    def perf():
        t["now"] += t["tick"]
        return t["now"]

    monkeypatch.setattr(time, "perf_counter", perf)
    return t


def make_req():
    return Request("r1", [1, 2, 3, 4], SamplingParams(max_tokens=3))


def make_scheduler():
    cfg = EngineConfig(model="t", device="cpu", max_num_seqs=4, max_num_batched_tokens=512)
    kv = KVCacheManager(64, 16, True)
    return Scheduler(cfg, kv)


def test_histogram_buckets_are_cumulative():
    h = Histogram((0.1, 0.5, 1.0))
    for v in (0.05, 0.4, 0.9, 2.0):
        h.observe(v)
    # an observation increments every bucket it is <=, not just the first
    assert h.counts == [1, 2, 3]  # +Inf lands only in count/sum
    assert h.count == 4
    assert h.sum == pytest.approx(3.35)

    edge = Histogram((0.1, 0.5, 1.0))
    edge.observe(0.5)  # a value exactly on an edge belongs to that bucket
    assert edge.counts == [0, 1, 1]


def test_histogram_render_ends_in_inf_count():
    h = Histogram((0.1, 0.5, 1.0))
    for v in (0.05, 2.0):
        h.observe(v)
    text = "\n".join(render_histogram("x_seconds", h, '{model_name="m"}'))
    assert '_bucket{model_name="m",le="+Inf"} 2' in text
    assert 'x_seconds_count{model_name="m"} 2' in text
    assert 'le="0.1"} 1' in text  # 0.05 <= 0.1


def test_lifecycle_admit_preempt_readmit_tokens_finish(clock):
    sch = make_scheduler()
    req = make_req()

    sch.add(req)
    batch = sch.schedule()
    assert len(batch) == 1
    admit = req.first_scheduled
    arrival = req.arrival
    assert admit is not None
    queue = admit - arrival

    # preempt and readmit: first_scheduled must survive untouched
    sch.preempt(0)
    sch.schedule()
    assert req.first_scheduled == admit

    m = EngineMetrics()
    t1 = time.perf_counter()
    req.token_ids.append(7)  # the engine appends before it observes
    m.observe_first_token(req, t1)
    assert req.first_token == t1 == admit + clock["tick"]
    assert m.queue_time.sum == pytest.approx(queue)
    assert m.time_to_first_token.count == 1

    # three committed tokens total -> two ITL gaps
    for _ in range(2):
        req.token_ids.append(8)
        m.observe_next_token(req, time.perf_counter())
    assert m.inter_token_latency.sum == pytest.approx(2 * clock["tick"])

    # finish: e2e covers load -> last token, tpot spans (n-1) gaps
    req.finished, req.finish_reason = True, "length"
    t_fin = time.perf_counter()
    m.observe_finish(req, t_fin)
    assert m.e2e_latency.count == 1
    assert m.time_per_output_token.sum == pytest.approx((t_fin - req.first_token) / 2)
    assert m.finished == {"length": 1}
