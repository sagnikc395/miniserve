from miniserve.attention.backends import ATTENTION_BACKENDS
from miniserve.attention.forward_context import set_forward_context
from miniserve.model.loader import load_model
from miniserve.platform import device_memory
import torch

class ModelRunner:
    # turns the schedulers decisions into GPU tensors
    # and then runs the model

    def __init__(self,cfg):
        self.cfg = cfg
        # the device the whole runner lives on: cuda, mps or cpu
        self.device = cfg.device
        torch.manual_seed(cfg.seed)

        #load the model
        self.model = load_model(cfg.model,cfg.dtype,cfg.max_model_len)

        # get the model config
        mc = self.model.cfg

        # get the metadata
        self.H ,self.Hkv = mc.num_attention_heads, mc.num_key_value_heads
        self.p = getattr(mc,"head_dim",None) or mc.hidden_size // self.H
        self.num_layers = mc.num_hidden_layers

        # TODO: set attention backends

        # set the max blocks per sequence (if our model supports lens upto 1024, and our block size
        # is 256, we can have at ost 4 blocks)
        self.max_blocks_per_seq = -(-cfg.max_model_len // cfg.block_size)

        # build persistent buffers for the decode path
        N, S  = cfg.max_num_batched_tokens, cfg.max_num_seqs

        # helper method to create an input with a given shape
        z = lambda *shape, dt=torch.int32: torch.zeros(*shape, dtype=dt, device=self.device)

        # create some dummy inuts to act as our buffer
        # when we are decoding , leverage the CUDA graph
        self.input_ids , self.positions = z(N,dt=torch.int64), z(N,dt=torch.int64)
        self.slot_mapping = z(N,dt=torch.int64)
        # when we are decoding, we do it on the CPU
        self.query_start_loc, self.cu_seqlens_k = z(S+1), z(S+1)
        self.block_table = z(S, self.max_blocks_per_seq)

        # a place to store the graphs : ("decode", batch_size) or ("mixed",token count)
        # acts like a replay graph
        # instead of capture every cuda graph at some granular level and then pad our data when we are decoding
        self.graphs: dict[tuple[str,int],tuple[torch.cuda.CUDAGraph,torch.Tensor]] = {}

        # last step's sampled token ids.
        # stay on th GPU (next step's inut ids copied from here) and also go to pinned host memory
        # in the background, where Engine.step will pick hem up once they arrived.
        #
        self.prev_tokens = z(S,dt=torch.int64)
        # pinned host memory is a CUDA transfer optimisation; MPS unified memory
        # does not need it (and `pin_memory` is a no-op there).
        self.tokens_host = torch.zeros(
            S, dtype=torch.int64, pin_memory=self.device.type == "cuda"
        )

    @torch.inference_mode()
    def allocate_kv_cache(self):
        # we first have to allocate the KV Cache
        # run the largest possible batch without a cache, measure peak memory
        # created by the activations and intermediate computations and give the rest to KV
        # (upto the set gpu_memory_utilization)
        device, cfg = self.device, self.cfg

        #1. ensure that ATTN_META is just None for now
        with set_forward_context(None):
            # reset the status (CUDA keeps a persistent high-water mark)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            elif device.type == "mps":
                torch.mps.synchronize()

            # measure the high water mark of weights + activations + workspace
            # for the largest packed step we will ever run
            n = cfg.max_num_batched_tokens
            h = self.model(self.input_ids[:n], self.positions[:n])

            # logits path sized for a full decode batch (one row per seq slot)
            logits = self.model.logits(h[: cfg.max_num_seqs]).softmax(-1)

            # read the high-water mark, then drop the profiling tensors below.
            # CUDA keeps an explicit peak counter; on MPS the driver allocation
            # never shrinks, so it acts as one.
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device)
                free, total = torch.cuda.mem_get_info(device)
                # total -> whole GPU
                # free -> unused right now
                # reserved -> torch's caching allocator arena
                # other -> bytes held by someone else (or untracked)
                other = total - free - torch.cuda.memory_reserved(device)
            else:
                total = device_memory(device)[0]
                # MPS uses unified memory: recommended_max_memory() is Metal's
                # working-set budget, and there is no separate "someone else's
                # memory" pool to subtract. There is also no peak counter, but
                # the driver allocation never shrinks, so it is the high-water
                # mark of weights + activations + the allocator's cached blocks.
                if device.type == "mps":
                    peak = torch.mps.driver_allocated_memory()
                else:
                    # CPU is a debugging path; just account for the weights.
                    peak = sum(
                        p.numel() * p.element_size() for p in self.model.parameters()
                    )
                other = 0

        # drop the tensors; the peak has already been recorded
        del logits, h

        # bytes per page: K and V, every layer, block_size tokens, one KV head
        # per group, head_dim each, in the configured dtype
        element_size = torch.finfo(cfg.dtype).bits // 8
        bytes_per_block = (
            2 * self.num_layers * cfg.block_size * self.Hkv * self.p * element_size
        )

        # never allocate more pages than the scheduler could ever hand out;
        # the +1 is the reserved null page 0.
        max_blocks = cfg.max_num_seqs * self.max_blocks_per_seq + 1
        budget = int(total * cfg.gpu_memory_utilization) - other - peak
        num_blocks = min(budget // bytes_per_block, max_blocks)

        if num_blocks < 2:
            raise RuntimeError(
                "not enough memory for a KV cache: "
                f"budget={budget} bytes, bytes_per_block={bytes_per_block}, "
                f"device={device}. Lower max_model_len, max_num_seqs or "
                "gpu_memory_utilization, or use a smaller model/dtype."
            )

        # every layer gets one tensor shaped [2, num_blocks, block_size, Hkv, D].
        # All layers share the same page ids, so one block table serves them all.
        shape = (2, num_blocks, cfg.block_size, self.Hkv, self.p)
        for layer in self.model.layers:
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is not None:
                attn.kv_cache = torch.zeros(shape, dtype=cfg.dtype, device=device)

        self.num_blocks = num_blocks
        return num_blocks
