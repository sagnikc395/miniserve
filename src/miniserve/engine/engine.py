from dataclasses import dataclass
from miniserve.core.kv_cache import KVCacheManager
import torch
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from miniserve.runner.model_runner import ModelRunner
from miniserve.core.scheduler import Scheduler
from miniserve.core.sequence import Request


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

class Engine:
    def __init__(self,cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.stop_ids = {self.tokenizer.eos_token_id}

        # some model could have multiple stop ids so just incase we make a list
        try:
            eos = GenerationConfig.from_pretrained(cfg.model).eos_token_id
            self.stop_ids |= set(eos if isinstance(eos,list) else [eos])
        except Exception:
            pass

        # if none is in here so just remove it
        self.stop_ids.discard(None)

        # set up runner
        self.runner = ModelRunner(cfg)
        num_blocks = self.runner.allocate_kv_cache()
        if cfg.use_cuda_graphs:
            self.runner.capture_cuda_graphs()

        # setup the kv cache and the scheduler
        self.kv = KVCacheManager(num_blocks,cfg.block_size,cfg.enable_prefix_caching)
        self.scheduler = Scheduler(cfg,self.kv)

        # setup storage for the rest
        self.detok: dict[str,Detokenizer] = {}
        self.requests: dict[str,Request] = {}
