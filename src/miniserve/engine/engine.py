import asyncio
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass

import torch
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer

from miniserve.core.kv_cache import KVCacheManager
from miniserve.core.scheduler import Scheduler
from miniserve.core.sequence import Request
from miniserve.runner.model_runner import ModelRunner
from miniserve.sampling import SamplingParams

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Output:
    request_id: str
    text: str  # fully generated text so far
    delta: str  # new text since the previous output
    token_ids: list[int]  # all generated token ids so far
    finished: bool
    finish_reason: str | None  # "stop" | "length" | "abort" | "error" | None
    num_prompt_tokens: int
    num_cached_tokens: int  # prefix-cache hits, reported as usage.cached_tokens
    logprobs: list[torch.Tensor] | None = None
    error: str | None = None


# detokenize and return back the generated token IDs into text
# also avoid emitting a trailing escape sequence because it can mean that
# a multi byte UTF-8 character is incomplete and needs another token
class Detokenizer:
    def __init__(self, tok) -> None:
        self.tok = tok
        self.text = ""
        self.prefix = 0
        self.read = 0

    def update(self, ids, final) -> str:
        old = self.tok.decode(ids[self.prefix : self.read], skip_special_tokens=True)
        new = self.tok.decode(ids[self.prefix :], skip_special_tokens=True)

        if len(new) <= len(old) or (new.endswith("\ufffd") and not final):
            return ""

        delta = new[len(old) :]
        self.text += delta
        self.prefix, self.read = self.read, len(ids)
        return delta


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.stop_ids = {self.tokenizer.eos_token_id}

        # sampling defaults shipped with the model (vLLM does the same)
        self.generation_defaults = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
        }
        try:
            gen = GenerationConfig.from_pretrained(cfg.model)
            # some model could have multiple stop ids, so keep them all
            eos = gen.eos_token_id
            self.stop_ids |= set(eos if isinstance(eos, list) else [eos])
            for name in ("temperature", "top_p", "top_k"):
                value = getattr(gen, name, None)
                if value is not None:
                    self.generation_defaults[name] = value
        except Exception:
            pass

        # if none is in here then just remove it
        self.stop_ids.discard(None)

        # set up runner
        self.runner = ModelRunner(cfg)
        num_blocks = self.runner.allocate_kv_cache()
        if cfg.use_cuda_graphs:
            self.runner.capture_cuda_graphs()

        # setup the kv cache and the scheduler
        self.kv = KVCacheManager(
            num_blocks, cfg.block_size, cfg.enable_prefix_caching
        )
        self.scheduler = Scheduler(cfg, self.kv)

        # setup storage for the rest
        self.detok: dict[str, Detokenizer] = {}
        self.requests: dict[str, Request] = {}

        self.started_at = time.time()
        self.num_requests = 0
        self.num_finished = 0
        self.num_prompt_tokens = 0
        self.num_generated_tokens = 0
        self.num_cached_tokens = 0
        self.num_steps = 0
        self.last_step_tokens = 0

    @property
    def max_model_len(self) -> int:
        return self.cfg.max_model_len

    def encode(self, prompt) -> list[int]:
        ids = (
            self.tokenizer.encode(prompt)
            if isinstance(prompt, str)
            else list(prompt)
        )
        if not 0 < len(ids) < self.max_model_len:
            raise ValueError(
                f"prompt has {len(ids)} tokens; must be between 1 and "
                f"{self.max_model_len - 1}"
            )
        return ids

    def add_request(self, prompt, params: SamplingParams, request_id=None):
        ids = self.encode(prompt)
        req_id = request_id or uuid.uuid4().hex

        self.requests[req_id] = req = Request(req_id, ids, params)
        self.scheduler.add(req)
        self.detok[req_id] = Detokenizer(self.tokenizer)

        self.num_requests += 1
        self.num_prompt_tokens += len(ids)
        return req_id

    def abort(self, request_id: str):
        # client has disappeared, just stop and give its KV blocks back.
        req = self.requests.pop(request_id, None)
        if req is not None and not req.finished:
            self.scheduler.finish(req, "abort")
        self.detok.pop(request_id, None)

    def has_work(self):
        return self.scheduler.has_work()

    @torch.inference_mode()
    def step(self):
        batch = self.scheduler.schedule()
        if not batch:
            return []

        tokens, logits = self.runner.run(batch)

        outputs = []
        for i, (s, tok) in enumerate(zip(batch, tokens)):
            req = s.req

            # chunked prefill that stops mid-prompt: we wrote k/v for those
            # tokens, but the last position is not the next output token, so
            # discard the sample
            if not s.completes_prompt:
                continue

            if req.params.return_logprobs:
                lp = torch.log_softmax(logits[i].float(), dim=-1)
                req.logprobs.append(lp.cpu())

            req.token_ids.append(tok)
            n_out = len(req.output_ids)

            # stop conditions: EOS (unless ignored) or no token space left
            if tok in self.stop_ids and not req.params.ignore_eos:
                self.scheduler.finish(req, "stop")
            elif n_out >= req.params.max_tokens or req.num_tokens >= self.max_model_len:
                self.scheduler.finish(req, "length")

            delta = self.detok[req.request_id].update(
                req.output_ids, req.finished
            )
            self.num_generated_tokens += 1

            out = Output(
                request_id=req.request_id,
                text=self.detok[req.request_id].text,
                delta=delta,
                token_ids=list(req.output_ids),
                finished=req.finished,
                finish_reason=req.finish_reason,
                num_prompt_tokens=req.prompt_len,
                num_cached_tokens=req.num_cached_tokens,
                logprobs=req.logprobs if req.params.return_logprobs else None,
            )
            outputs.append(out)

            if req.finished:
                self.num_finished += 1
                self.num_cached_tokens += req.num_cached_tokens
                self.requests.pop(req.request_id, None)
                self.detok.pop(req.request_id, None)

        self.num_steps += 1
        self.last_step_tokens = sum(s.num_new for s in batch)
        return outputs

    def stats(self) -> dict:
        total_blocks = len(self.kv.ref_cnt) - 1  # page 0 is reserved
        used_blocks = total_blocks - self.kv.num_free
        return {
            "model": self.cfg.model,
            "started_at": self.started_at,
            "uptime_s": time.time() - self.started_at,
            "requests": self.num_requests,
            "finished": self.num_finished,
            "running": len(self.scheduler.running),
            "waiting": len(self.scheduler.waiting),
            "prompt_tokens": self.num_prompt_tokens,
            "generated_tokens": self.num_generated_tokens,
            "cached_tokens": self.num_cached_tokens,
            "steps": self.num_steps,
            "last_step_tokens": self.last_step_tokens,
            "preemptions": getattr(self.scheduler, "preemptions", 0),
            "kv_used_blocks": used_blocks,
            "kv_total_blocks": total_blocks,
            "kv_usage_ratio": used_blocks / total_blocks if total_blocks else 0.0,
        }


class AsyncEngine:
    """Runs the synchronous engine on a dedicated thread.

    The FastAPI event loop submits `add` / `abort` messages through `inbox` and
    receives `Output` objects through a per-request asyncio queue.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.inbox: queue.Queue = queue.Queue()
        self.streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._thread: threading.Thread | None = None
        self._closed = False

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._loop, name="miniserve-engine", daemon=True
        )
        self._thread.start()
        return self

    def stop(self):
        if self._closed:
            return
        self._closed = True
        self.inbox.put(None)
        if (
            self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=5)

    close = stop

    # ---- engine thread -------------------------------------------------
    def _handle(self, message):
        kind, rid, payload = message
        if kind == "add":
            prompt, params = payload
            try:
                self.engine.add_request(prompt, params, request_id=rid)
            except Exception as exc:
                logger.exception("failed to add request %s", rid)
                self._deliver_error(rid, exc)
        elif kind == "abort":
            self.engine.abort(rid)

    def _deliver_error(self, rid, exc):
        stream = self.streams.get(rid)
        if stream is None:
            return
        loop, q = stream
        out = Output(
            request_id=rid,
            text="",
            delta="",
            token_ids=[],
            finished=True,
            finish_reason="error",
            num_prompt_tokens=0,
            num_cached_tokens=0,
            logprobs=None,
            error=str(exc),
        )
        loop.call_soon_threadsafe(q.put_nowait, out)

    def _dispatch(self, outputs):
        for out in outputs:
            stream = self.streams.get(out.request_id)
            if stream is None:
                continue
            loop, q = stream
            loop.call_soon_threadsafe(q.put_nowait, out)

    def _loop(self):
        while not self._closed:
            try:
                while True:
                    message = self.inbox.get_nowait()
                    if message is None:
                        return
                    self._handle(message)
            except queue.Empty:
                pass

            if self.engine.has_work():
                try:
                    outputs = self.engine.step()
                except Exception:
                    logger.exception("engine step failed")
                    continue
                self._dispatch(outputs)
                continue

            # idle: sleep until a request arrives
            message = self.inbox.get()
            if message is None:
                return
            self._handle(message)

    # ---- event loop ----------------------------------------------------
    async def generate(
        self, prompt, params: SamplingParams, request_id: str | None = None
    ):
        # validate here as well so a bad prompt cannot kill the engine thread
        if isinstance(prompt, str):
            self.engine.encode(prompt)

        rid = request_id or uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        self.streams[rid] = (loop, q)
        self.inbox.put(("add", rid, (prompt, params)))

        finished = False
        try:
            while True:
                out = await q.get()
                if out.error is not None:
                    raise RuntimeError(out.error)
                yield out
                if out.finished:
                    finished = True
                    break
        finally:
            self.streams.pop(rid, None)
            if not finished:
                self.inbox.put(("abort", rid, None))
