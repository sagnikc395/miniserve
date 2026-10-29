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
    top_v = torch.tensor([p.top_p for p in params],device=dev)
