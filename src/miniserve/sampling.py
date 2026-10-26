from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0  # 0 -> greedy decode
    top_p: float = 1.0
    top_k: int = -1  # -1 means off
    max_tokens: int = 64  # max tokens to generate
    ignore_eos: bool = False  # as we normally stop generating on eos token

    # testing
    forced_token_ids: list[int] | None = None  # teacher forcing to compare log probs
    return_logprobs: bool = False
