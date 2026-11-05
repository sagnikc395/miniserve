from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from miniserve.api.deps import get_engine
from miniserve.engine.engine import AsyncEngine

router = APIRouter()


@router.get("/stats")
def stats(aeng: AsyncEngine = Depends(get_engine)):
    return aeng.engine.stats()


@router.get("/metrics", include_in_schema=False)
def metrics(aeng: AsyncEngine = Depends(get_engine)) -> PlainTextResponse:
    engine = aeng.engine
    s = engine.stats()
    model = s["model"].replace("\\", "\\\\").replace('"', '\\"')
    label = f'{{model_name="{model}"}}'

    counters = {
        "miniserve_requests_total": s["requests"],
        "miniserve_prompt_tokens_total": s["prompt_tokens"],
        "miniserve_generation_tokens_total": s["generated_tokens"],
        "miniserve_prefix_cache_hit_tokens_total": s["cached_tokens"],
        "miniserve_preemptions_total": s["preemptions"],
    }

    lines = []
    for name, value in counters.items():
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name}{label} {value}")

    # one series per finish reason: stop vs length vs abort distinguishes a
    # healthy finish from a truncation, without a per-request label (§4)
    lines.append("# TYPE miniserve_requests_finished_total counter")
    for reason, count in sorted((s.get("finished_reasons") or {}).items()):
        lines.append(
            f'miniserve_requests_finished_total{{model_name="{model}",reason="{reason}"}} {count}'
        )

    gauges = {
        "miniserve_num_requests_running": s["running"],
        "miniserve_num_requests_waiting": s["waiting"],
        "miniserve_kv_cache_usage_ratio": s["kv_usage_ratio"],
        "miniserve_kv_cache_used_blocks": s["kv_used_blocks"],
        "miniserve_kv_cache_total_blocks": s["kv_total_blocks"],
        "miniserve_step_batched_tokens": s["last_step_tokens"],
        "miniserve_uptime_seconds": s["uptime_s"],
    }
    for name, value in gauges.items():
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{label} {value}")

    # FakeEngine (tests) carries no EngineMetrics
    m = getattr(engine, "m", None)
    if m is not None:
        lines.extend(m.render(label))
    return PlainTextResponse(
        "\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
