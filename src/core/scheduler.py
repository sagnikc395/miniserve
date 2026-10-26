import torch
from src.core.kv_cache import KVCacheManager
from src.miniserve.config import SamplingParams, Config
from dataclasses import dataclass
from collections import deque


class Request:
    # metadata needed by the scheduler about a single request
    #
    # num_computed_tokens is the most important metric -> how many token_ids already have K/V in the cache
    # each step the scheduler picks how many new tokens to process for this request.

    # eg: given a budget of 512 tokens , but prompt size of 1000 tokens, we first do 512 tokens and then the rest 488
    #
    def __init__(
        self, request_id: str, prompt_ids: list[int], params: SamplingParams
    ) -> None:
        self.request_id = request_id
        self.prompt_len = len(prompt_ids)

        # prompt + generated so far
        self.token_ids = list(prompt_ids)
        self.params = params

        # computed till now ie written in prefill
        self.num_computed_tokens = 0

        # request the page table (what block contain the KV cache for this specific request?)
        self.block_ids: list[int] = []

        self.block_hashes: list[int] = []
        self.num_cached_tokens = 0  # prefix-cache hits at admission
        self.logprobs: list[torch.Tensor] = []
        self.finished = False
        self.finish_reason: str | None = None

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def output_ids(self):
        # return everything after the prompt of what we generated
        return self.token_ids[self.prompt_len :]


@dataclass
class ScheduledRequest:
    req: Request  # live request
    num_computed: int  # kv present before the forward pass
    num_new: int  # how many tokens this forward pass will process

    @property
    def completes_prompt(self):
        # true -> this step reaches the end of the current sequence (finish prefill or decode). Smapled next token will be real and we append it
        # false -> chunked prefill stops mid prompt , we still run these tokens to write KV, but the last positions sampled token will be junk and just throw it away, next step continues from the prompt
        return self.num_computed + self.num_new >= self.req.num_tokens


class Scheduler:
    # called once per step, which requests run, and how many tokens each
    # 1. Running requests fo first.
    # 2. Then admit waitiing requests while there is token budget
    def __init__(self, cfg: Config, kv: KVCacheManager):
        self.cfg = cfg
        self.kv = kv
        self.waiting: deque[Request] = deque()  # FIFO of not yet admitted requests
        self.running: list[Request] = []  # currently holding KV pages

    def add(self, req: Request):
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> list[ScheduledRequest]:
        pass

    def finish(self, req: Request, reason: str) -> None:
        req.finished, req.finish_reason = True, reason
        if req in self.running:
            self.running.remove(req)
        elif req in self.waiting:
            self.waiting.remove(req)
        self.kv.free_request(req)
