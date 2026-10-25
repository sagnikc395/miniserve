## miniserve 

a tiny vLLM implementation, to serve models at scale.

parts of the implementation:
- scheduler (kv cache manager before this)
- forward pass
- engine
- serving 

### references

- [Paged Attention](https://arxiv.org/abs/2309.06180)
- [Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM Scheduler High Level Sketch](https://audreywongkg.medium.com/understanding-vllm-scheduling-token-budgets-chunked-prefill-and-policies-2c879e3980e3)
-
