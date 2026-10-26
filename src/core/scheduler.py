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

        budget = self.cfg.max_num_batched_tokens

        # for each request id, how many tokens will we run this step ?
        plan = {}

        # if we evict anyone, dont admit anyone new work
        preempted = False

        # running requests
        i = 0

        while i < len(self.running) and budget > 0:
            # what request
            req = self.running[i]

            # actively decoding , new tokens to generate
            n = min(req.num_tokens - req.num_computed_tokens, budget)

            while not self.kv.allocate(req, n):
                # as long as it doesnt fit in our kv cache, we have an issue and we need to evict things
                # KV RUNNING OUT OF SPACE WHEN a comes, so c gets evicted
                victim = self.running.pop()
                self.kv.free_request(victim)
                victim.num_computed_tokens = 0
                # first thing on the waiting list
                self.waiting.appendleft(victim)
                # all the tokens that we have gone back
                budget += plan.pop(victim.request_id, 0)
                # something has been skipped
                preempted = True

                if victim is req:
                    # grown past the boundaries and kv cache
                    break

            if req not in self.running:
                # the request is done
                break

            # this request is responsible for these many tokens
            plan[req.request_id] = n

            budget -= n
            i += 1

        # currently waiting request
        while (not preempted and self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs):
            # for something in the waiting list and some budget remaining
            req = self.waiting[0]
            cached = self.kv.find_cached_prefix(req)
            # how many tokens that I have - how many can i borrow from precomputed
            n = min(req.num_tokens - len(cached) * self.cfg.block_size, budget)

            if not self.kv.allocate(req,n,cached):
                # couldnt fit this particular request, this budget based on our total token count
                break

            self.waiting.popleft()

            req.num_computed_tokens = len(cached) * self.cfg.block_size
            # if some request can get evicted we can recompute and keep track of what the best cache hit we had
            req.num_cached_tokens = max(req.num_cached_tokens, req.num_computed_tokens)

            self.running.append(req)

            plan[req.request_id] = n
            # decrease budget by n tokens
            budget -= n

        # emit the batch in running order
        batch = []
        for req in self.running:
            if req.request_id in plan:
                batch.append(
                    ScheduledRequest(
                        req,req.num_computed_tokens,plan[req.request_id]
                    )
                )

                # advance optimistically
                req.num_computed_tokens += plan[req.request_id]

        return batch

    def finish(self, req: Request, reason: str) -> None:
        req.finished, req.finish_reason = True, reason
        if req in self.running:
            self.running.remove(req)
        elif req in self.waiting:
            self.waiting.remove(req)
        self.kv.free_request(req)
