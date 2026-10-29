import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig

def load_model(name: str, dtype, max_position: int, tp_rank:int=0,
    tp_size: int = 1, expert_parallel: bool= False):
        # a utility function to load weights into our model (this rank's shard of them)
        path = name if os.path.isdir(name) else snapshot_download(
            name, allow_patterns=["*.json","*.safetensors","tokenizer*","*.txt"]
        )

        cfg = AutoConfig.from_pretrained(path)

        pass


def to_gpu(data, dtype=None) -> torch.Tensor:
    ## host data (list or numpy) -> GPU tensor
    t = torch.as_tensor(data,dtype=dtype)
    return t.pin_memory().to("mps",non_blocking=True)
