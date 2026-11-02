"""Request/response schemas for the OpenAI-compatible HTTP API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from miniserve.sampling import SamplingParams


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Any = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    n: int = 1
    seed: int | None = None
    user: str | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    prompt: str | list[str] | list[int]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    echo: bool = False
    n: int = 1
    seed: int | None = None
    user: str | None = None


def content_to_text(content: Any) -> str:
    """Flatten OpenAI message content (str or a list of parts) to text.

    Image/audio parts are dropped: this is a text-only server.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in (None, "text", "input_text"):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)
    return str(content)


def chat_messages_to_dicts(messages: list[ChatMessage]) -> list[dict]:
    out = []
    for message in messages:
        item = {"role": message.role, "content": content_to_text(message.content)}
        if message.name:
            item["name"] = message.name
        if message.tool_calls:
            item["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        out.append(item)
    return out


def build_sampling_params(
    request: ChatCompletionRequest | CompletionRequest,
    defaults: dict,
    prompt_len: int,
    max_model_len: int,
) -> SamplingParams:
    temperature = (
        request.temperature
        if request.temperature is not None
        else defaults.get("temperature", 1.0)
    )
    top_p = (
        request.top_p if request.top_p is not None else defaults.get("top_p", 1.0)
    )
    top_k = (
        request.top_k if request.top_k is not None else defaults.get("top_k", -1)
    )

    remaining = max(1, max_model_len - prompt_len)
    requested = getattr(request, "max_completion_tokens", None) or request.max_tokens
    max_tokens = remaining if requested is None else requested
    max_tokens = min(max(1, max_tokens), remaining)

    return SamplingParams(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_tokens=int(max_tokens),
        ignore_eos=bool(request.ignore_eos),
    )
