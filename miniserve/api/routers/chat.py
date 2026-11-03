from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from miniserve.api.deps import get_engine
from miniserve.api.errors import APIError, error_body
from miniserve.api.routers.common import (
    created,
    finish_reason,
    new_id,
    sse,
    usage,
)
from miniserve.api.schemas import (
    ChatCompletionRequest,
    build_sampling_params,
    chat_messages_to_dicts,
)
from miniserve.engine.engine import AsyncEngine

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    aeng: AsyncEngine = Depends(get_engine),
):
    engine = aeng.engine
    cfg = engine.cfg

    if body.n != 1:
        raise APIError(400, "only n=1 is supported", param="n")

    messages = chat_messages_to_dicts(body.messages)
    if not messages:
        raise APIError(400, "messages must not be empty", param="messages")

    try:
        prompt = engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as exc:
        raise APIError(400, f"chat template failed: {exc}") from exc

    ids = engine.tokenizer.encode(prompt)
    if not ids:
        raise APIError(400, "the prompt is empty", param="messages")
    if len(ids) >= cfg.max_model_len:
        raise APIError(
            400,
            f"prompt is too long: {len(ids)} tokens > {cfg.max_model_len - 1} maximum",
            param="messages",
            code="context_length_exceeded",
        )

    params = build_sampling_params(
        body,
        engine.generation_defaults,
        prompt_len=len(ids),
        max_model_len=cfg.max_model_len,
    )
    rid = new_id("chatcmpl")
    model = body.model or cfg.model

    if body.stream:
        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        return StreamingResponse(
            _stream(aeng, ids, params, rid, model, include_usage),
            media_type="text/event-stream",
        )

    final = None
    async for out in aeng.generate(ids, params, request_id=rid):
        final = out

    if final is None:
        raise APIError(500, "the engine produced no output", type="server_error")

    return {
        "id": rid,
        "object": "chat.completion",
        "created": created(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final.text},
                "finish_reason": finish_reason(final),
            }
        ],
        "usage": usage(final),
    }


async def _stream(aeng, ids, params, rid, model, include_usage):
    stamp = created()
    role_sent = False
    final = None

    try:
        async for out in aeng.generate(ids, params, request_id=rid):
            final = out
            delta = out.delta
            if not role_sent:
                delta = {"role": "assistant", "content": delta}
                role_sent = True
            else:
                if not delta:
                    continue
                delta = {"content": delta}

            yield sse(
                {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": stamp,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None,
                        }
                    ],
                }
            )

        yield sse(
            {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": stamp,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason(final)
                        if final is not None
                        else "stop",
                    }
                ],
            }
        )
        if include_usage and final is not None:
            yield sse(
                {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": stamp,
                    "model": model,
                    "choices": [],
                    "usage": usage(final),
                }
            )
    except Exception as exc:
        yield sse(
            error_body(
                f"generation failed: {exc}",
                type="server_error",
            )
        )

    yield sse("[DONE]")
