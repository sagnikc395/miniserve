import torch

from miniserve.sampling import SamplingParams


def sample(logits, params: list[SamplingParams], token_ids=None):
    """Sample one token per row of `logits` ([B, V]).

    token_ids, when given, is a per-row list of forced token ids (teacher
    forcing for equivalence tests). A None entry samples normally.
    """
    if logits.shape[0] == 0:
        return []

    # a batched call for mixed sampling settings
    greedy = logits.argmax(axis=-1)

    if all(p.temperature == 0 for p in params):
        tokens = greedy.tolist()
    else:
        dev = logits.device

        temp = torch.tensor(
            [p.temperature or 1.0 for p in params], device=dev
        )
        scaled = logits / temp[:, None]  # B * V / B * 1

        V = scaled.shape[-1]

        top_k = torch.tensor(
            [p.top_k if p.top_k > 0 else V for p in params], device=dev
        )
        top_p = torch.tensor([p.top_p for p in params], device=dev)

        # sort logits by most likely to least likely
        sorted_logits, idx = scaled.sort(-1, descending=True)

        ranks = torch.arange(V, device=dev)[None, :]
        probs = sorted_logits.softmax(axis=1)

        # drop a sorted position if it fails top-p or top-k
        drop = (ranks >= top_k[:, None]) | (
            probs.cumsum(-1) - probs > top_p[:, None]
        )
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))

        # idx -> how we sorted it and then unsorted it
        drop_unsorted = torch.zeros_like(drop).scatter_(-1, idx, drop)
        probs = scaled.masked_fill(drop_unsorted, float("-inf")).softmax(-1)

        # Gumbel-Max: multimodal sampling from a distribution
        gumbel = -torch.empty_like(probs).exponential_().log()
        sampled = (probs.log() + gumbel).argmax(-1)

        # rows with temperature 0 are replaced by their argmax
        is_greedy = torch.tensor(
            [p.temperature == 0 for p in params], device=dev
        )
        tokens = torch.where(is_greedy, greedy, sampled).tolist()

    if token_ids is not None:
        tokens = [
            int(forced) if forced is not None else tok
            for tok, forced in zip(tokens, token_ids)
        ]
    return tokens
