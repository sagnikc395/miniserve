from dataclasses import dataclass


@dataclass(slots=True)
class Output:
    request_id: str
    text: str # fully generated text so far
    delta: str # new text since the previous output
    token_ids: list[int] # all generated token ids so far
    finished: bool
    finish_reason: str | None # "stop" | "length" | None
    num_prompt_tokens: int
    num_cached_tokens: int # prefix-cache hits, reported as usage.cached_tokens
