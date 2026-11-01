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

    top_k = torch.tensor([p.top_k if p.top_k > 0 else v for p in params],device=dev)
    top_p = torch.tensor([p.top_p for p in params],device=dev)

    # sort logits by most likely to least likely
    sorted_logits, idx = logits.sort(-1,descending=True)

    ranks = torch.arange(V,device=dev)[None,:]
    probs = sorted_logits.softmax(axis=1)

    # drop a sorted position if it fails a top p or top k
    drop = (ranks >= top_k[:,None]) | (probs.cumsum[-1] - probs > top_p[:,None])
    # the specific positions will be zeroed out
    sorted_logits = sorted_logits.masked_fill(drop,float("-inf"))

    # idx -> how we sorted it and then unsorted it
    drop_unsorted = torch.zeros_like(drop).scatter_(-1, idx, drop)
    probs = logits.masked_fill(drop_unsorted, float("-inf")).softmax(-1)

    # Gumbel-Max : multimodial sampling from a distribution
    gumbel = -torch.empty_like(probs).exponential_().log()
    sampled = (probs.log() + gumbel).argmax(-1)

    # if we wanted some samples to be greedy decode, we replace that here
    is_greedy = torch.tensor([p.temperature == 0 for p in params],device=dev)

    return torch.where(is_greedy,greedy, sampled).tolist()
