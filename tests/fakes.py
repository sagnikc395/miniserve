"""Shared test doubles: a tiny fake engine and paged-cache builders."""

import torch

from miniserve.config import EngineConfig
from miniserve.engine.engine import Output


class FakeTokenizer:
    eos_token_id = 2

    def encode(self, text):
        return [ord(c) % 50 + 1 for c in str(text)][:16] or [1]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(96 + (int(i) % 26) + 1) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(str(m["content"]) for m in messages)


class FakeEngine:
    def __init__(self):
        self.cfg = EngineConfig(
            model="fake-model",
            device="cpu",
            dtype="float32",
            max_model_len=64,
            max_num_batched_tokens=16,
        )
        self.tokenizer = FakeTokenizer()
        self.generation_defaults = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
        }
        self.requests = {}

    def stats(self):
        return {
            "model": self.cfg.model,
            "started_at": 0.0,
            "uptime_s": 0.0,
            "requests": 1,
            "finished": 1,
            "running": 0,
            "waiting": 0,
            "prompt_tokens": 4,
            "generated_tokens": 3,
            "cached_tokens": 0,
            "steps": 3,
            "last_step_tokens": 1,
            "preemptions": 0,
            "kv_used_blocks": 0,
            "kv_total_blocks": 8,
            "kv_usage_ratio": 0.0,
        }


class FakeAsyncEngine:
    def __init__(self):
        self.engine = FakeEngine()

    async def generate(self, prompt, params, request_id=None):
        pieces = ["Hel", "lo", "!"]
        for i, piece in enumerate(pieces):
            yield Output(
                request_id=request_id or "req",
                text="".join(pieces[: i + 1]),
                delta=piece,
                token_ids=[7, 8, 9][: i + 1],
                finished=i == len(pieces) - 1,
                finish_reason="stop" if i == len(pieces) - 1 else None,
                num_prompt_tokens=4,
                num_cached_tokens=1,
            )


class FakeRequest:
    def __init__(self, prompt_ids):
        self.prompt_ids = prompt_ids
