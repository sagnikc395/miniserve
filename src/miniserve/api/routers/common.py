import json
import time
import uuid

from miniserve.engine.engine import Output


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def created() -> int:
    return int(time.time())


def usage(out: Output) -> dict:
    prompt = out.num_prompt_tokens
    completion = len(out.token_ids)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": out.num_cached_tokens},
    }


def sse(payload) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def finish_reason(out: Output) -> str:
    reason = out.finish_reason or "stop"
    return "stop" if reason == "abort" else reason
