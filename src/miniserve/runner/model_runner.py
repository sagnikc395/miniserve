from miniserve.attention.backends import ATTENTION_BACKENDS
from miniserve.model.loader import load_model
import torch

class ModelRunner:
    # turns the schedulers decisions into GPU tensors
    # and then runs the model

    def __init__(self,cfg):
        self.cfg = cfg
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

        # build persistent buffers for CUDA graphs
        N, S  = cfg.max_num_batched_tokens, cfg.max_run_seqs

        # helper method to craete an input with a given shape
        z = lambda *shape, dt=torch.int32: torch.zeros(*shape, dtype=dt, device="cuda")

        # create some dummy inuts to act as our buffer
        self.input_ids , self.positions = z(N,dt=torch.int64), z(N,dt=torch.int64)
        self.slot_mapping = z(N,dt=torch.int64)
        self.query_start_loc, self.cu_seqlens_k = z(S+1), z(S+1)
        self.block_table = z(S, self.max_blocks_per_seq)

        # a place to store the graphs : ("decode", batch_size) or ("mixed",token count)
        # acts like a replay graph
        self.graphs: dict[tuple[str,int],tuple[torch.cuda.CUDAGraph,torch.Tensor]] = {}

        # last step's sampled token ids.
        # stay on th GPU (next step's inut ids copied from here) and also go to pinned host memory
        # in the background, where Engine.step will pick hem up once they arrived.
        #
        self.prev_tokens = z(S,dt=torch.int64)
        self.tokens_host = torch.zeros(s,dtype=torch.int64,pin_memory=True)
