from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Output:
    request_id: str
    text: str  # fully generated text so far
    delta: str  # new text since the previous output
    token_ids: list[int]  # all generated token ids so far
    finished: bool
    finish_reason: str | None  # "stop" | "length" | None
    num_prompt_tokens: int
    num_cached_tokens: int  # prefix-cache hits, reported as usage.cached_tokens
    logprobs: list[torch.Tensor] | None = None


# detokenize and return back the generated token IDs into text
# also avoid emitting a trailing escpae sequence because it can mean that
# a multi byte UTF-8 character is incomplete and needs another tokens
#
class Detokenizer:
    def __init__(self, tok) -> None:
        self.tok = tok
        self.text = ""
        self.prefix = 0
        self.read = 0
        self.released = 0

    def update(self, ids, final) -> str:
        old = self.tok.decode(
            ids[self.prefix : self.read],
            skip_special_tokens=True,
        )

        new = self.tok.decode(
            ids[self.prefix :],
            skip_special_tokens=True,
        )

        if len(new) <= len(old) or (new.endswith("�") and not final):
            return ""

        delta = new[len(old) :]
        self.text += delta
        self.prefix, self.read = self.read, len(ids)
        return delta
