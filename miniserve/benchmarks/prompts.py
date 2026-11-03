"""Prompt sets used by `main.py bench`."""

SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. Answer the question directly and "
    "do not repeat the prompt."
)

INSTRUCTIONS = [
    "List three uses of a paged KV cache.",
    "Explain continuous batching in two sentences.",
    "What is the difference between prefill and decode?",
    "Write a short Python function that reverses a string.",
    "Summarise the idea behind prefix caching.",
    "Why is decode bandwidth-bound rather than compute-bound?",
    (
        "A serving engine receives requests of very different lengths. Explain "
        "how a token budget and chunked prefill keep a long prompt from "
        "stalling requests that are already decoding."
    ),
    (
        "Compare static batching with continuous batching. Cover padding "
        "waste, tail latency, and how a finished request frees its slot for a "
        "waiting request on the next engine step."
    ),
]


def build_prompts(num_prompts: int, tokenizer=None) -> list[str]:
    """Return a deterministic mix of short and long instructions.

    When a tokenizer is supplied the prompt is rendered through the model's
    chat template, so the benchmark measures the same token stream a real
    chat request would produce.
    """
    prompts = []
    for i in range(num_prompts):
        instruction = INSTRUCTIONS[i % len(INSTRUCTIONS)]
        # stretch some prompts so the benchmark covers both short and long
        # prefills without pulling in an external dataset
        if i % 4 == 3:
            instruction = (
                instruction
                + "\n\nContext:\n"
                + ("The KV cache stores keys and values for every layer. " * 40)
            )

        if tokenizer is None:
            prompts.append(f"{SYSTEM_PROMPT}\n\nUser: {instruction}\nAssistant:")
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ]
            try:
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
            except Exception:
                prompts.append(f"{SYSTEM_PROMPT}\n\nUser: {instruction}\nAssistant:")
    return prompts
