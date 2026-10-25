import torch
from config import SamplingParams


class Request:
    # metadata needed by the scheduler about a single request
    #
    # num_computed_tokens is the most important metric -> how many token_ids already have K/V in the cache
    # each step the scheduler picks how many new tokens to process for this request.

    # eg: given a budget of 512 tokens , but prompt size of 1000 tokens, we first do 512 tokens and then the rest 488
    #
    def __init__(self,request_id: str,prompt_ids:list[int],params:SamplingParams) -> None:
        self.request_id = request_id
        self.prompt_len = len(prompt_ids)

        # prompt + generated so far
        self.token_ids = list(prompt_ids)
        self.params = params

        # computed till now ie written in prefill
        self.num_computed_tokens = 0

        # request the page table (what block contain the KV cache for this specific request?)
        self.block_ids: list[int] = []

        self.block_hashes :list[int] = []
        self.num_cached_tokens =  0 # prefix-cache hits at admission
        self.logprobs: list[torch.Tensor] = []
        self.finished = False
        self.finish_reason : str | None = None

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def output_ids(self):
        # return everything after the prompt of what we generated ,
        return self.token_ids[self.prompt_len:]
