import torch

def load_model(name: str, dtype, max_position: int, tp_rank:int=0,
    tp_size: int = 1, expert_parallel: bool= False):
        pass

def to_gpu(data, dtype=None) -> torch.Tensor:
    ## host data (list or numpy) -> GPU tensor
    t = torch.as_tensor(data,dtype=dtype)
    return t.pin_memory().to("mps",non_blocking=True)
