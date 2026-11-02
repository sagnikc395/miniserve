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
from miniserve.api.schemas import CompletionRequest, build_sampling_params
from miniserve.engine.engine import AsyncEngine

router = APIRouter()


def _prompts(engine, body: CompletionRequest):
    prompt = body.prompt
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        if not prompt:
            raise APIError(400, "prompt must not be empty", param="prompt")
        if all(isinstance(x, int) for x in prompt):
            return [list(prompt)]
        if all(isinstance(x, str) for x in prompt):
            return list(prompt)
    raise APIError(
        400, "prompt must be a string or a list of strings", param="prompt"
    )


@router.post("/v1/completions")
async def completions(
    body: CompletionRequest,
    aeng: AsyncEngine = Depends(get_engine),
):
    engine = aeng.engine
    cfg = engine.cfg

    if body.n != 1:
        raise APIError(400, "only n=1 is supported", param="n")

    prompts = _prompts(engine, body)
    if body.stream and len(prompts) != 1:
        raise APIError(
            400, "streaming requires exactly one prompt", param="prompt"
        )

    if body.stream:
        include_usage = bool(
            body.stream_options and body.stream_options.include_usage
        )
        return StreamingResponse(
            _stream(aeng, prompts[0], body, include_usage),
            media_type="text/event-stream",
        )

    choices = []
    final_for_usage = None
    for index, prompt in enumerate(prompts):
        ids = engine.encode(prompt)
        if len(ids) >= cfg.max_model_len:
            raise APIError(
                400,
                f"prompt is too long: {len(ids)} tokens > "
                f"{cfg.max_model_len - 1} maximum",
                param="prompt",
                code="context_length_exceeded",
            )
        params = build_sampling_params(
            body,
            engine.generation_defaults,
            prompt_len=len(ids),
            max_model_len=cfg.max_model_len,
        )
        rid = new_id("cmpl")
        final = None
        async for out in aeng.generate(ids, params, request_id=rid):
            final = out
        if final is None:
            raise APIError(
                500, "the engine produced no output", type="server_error"
            )
        final_for_usage = final
        text = final.text
        if body.echo:
            text = (
                prompt
                if isinstance(prompt, str)
                else engine.tokenizer.decode(ids)
            ) + text
        choices.append(
            {
                "index": index,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason(final),
            }
        )

    return {
        "id": new_id("cmpl"),
        "object": "text_completion",
        "created": created(),
        "model": body.model or cfg.model,
        "choices": choices,
        "usage": usage(final_for_usage),
    }


async def _stream(aeng, prompt, body: CompletionRequest, include_usage):
    stamp = created()
    engine = aeng.engine
    ids = engine.encode(prompt)
    if len(ids) >= engine.cfg.max_model_len:
        yield sse(
            error_body(
                f"prompt is too long: {len(ids)} tokens > "
                f"{engine.cfg.max_model_len - 1} maximum",
                code="context_length_exceeded",
            )
        )
        yield sse("[DONE]")
        return

    params = build_sampling_params(
        body,
        engine.generation_defaults,
        prompt_len=len(ids),
        max_model_len=engine.cfg.max_model_len,
    )
    rid = new_id("cmpl")
    model = body.model or engine.cfg.model
    final = None

    try:
        async for out in aeng.generate(ids, params, request_id=rid):
            final = out
            if not out.delta:
                continue
            yield sse(
                {
                    "id": rid,
                    "object": "text_completion",
                    "created": stamp,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "text": out.delta,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )

        yield sse(
            {
                "id": rid,
                "object": "text_completion",
                "created": stamp,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "logprobs": None,
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
                    "object": "text_completion",
                    "created": stamp,
                    "model": model,
                    "choices": [],
                    "usage": usage(final),
                }
            )
    except Exception as exc:
        yield sse(error_body(f"generation failed: {exc}", type="server_error"))

    yield sse("[DONE]")
