import torch
from miniserve.sampling import SamplingParams


def sample(logits, params: list[SamplingParams], token_ids):
    # a batched call for mixed sampling settings
    # logits : B * V
    # greedy decoding -> logits.argmax(axis=-1)
    # params : [sampling_1, sampling_2, sampling_3,...]
    greedy = logits.argmax(axis=-1)

    if all(p.temperature == 0 for p in params):
        return greedy.tolist()

    dev = logits.device

    temp = torch.tensor([p.temperature or 1.0 for p in params], device=dev)
    logits = logits / temp[:, None] # B* V / B * 1

    V = logits.shape[-1]

    top_k = torch.tensor([p.top_k if p.tok_k > 0 else v for p in params],device=dev)
    top_p = torch.tensor([p.top_p for p in params],device=dev)

    # sort logits by most likely to least likely
    sorted_logits, idx = logits.sort(-1,descending=True)

    ranks = torch.arange(V,device=dev)[None,:]
    probs = sorted_logits.softmax(axis=1)

    # drop a sorted position if it fails a top p or top k
    drop = (ranks >= top_k[:,None]) | (probs.cumsum[-1] - probs > top_p[:,None])
    # the specific positions will be zeroed out
    sorted_logits = sorted_logits.masked_fill(drop,float("-inf"))

    probs = torch.empty_like(logits)
