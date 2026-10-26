## miniserve 

A tiny vLLM implementation, to serve models at scale.
This uses a simplified implementation of 
- Paged KV Cached with prefix caching
- Continuous batching with chunked prefill, 
- SDPA(Flash Attention is CUDA dependent) , 
- static shapes + preallocation + Metal fusion , 
- and using FastAPI to setup a OpenAI-compatible HTTP Server.(to query and setup the whole flow of chatting with a LLM).


### Architecture
```mermaid
flowchart LR
    client["OpenAI SDK / curl"]

    subgraph server["miniserve (FastAPI, single process)"]
        subgraph apilayer["api"]
            app["create_app() + lifespan<br/>engine on app.state"]
            routers["Routers<br/>/v1/chat/completions, /v1/completions,<br/>/v1/models, /health, /metrics"]
            schemas["Pydantic schemas<br/>request validation"]
            deps["get_engine dependency"]
        end

        subgraph enginelayer["engine"]
            aeng["AsyncEngine<br/>inbox queue + engine thread"]
            eng["Engine.step()"]
            detok["Detokenizer<br/>incremental UTF-8 safe"]
        end

        subgraph corelayer["core"]
            sched["Scheduler<br/>continuous batching, chunked prefill,<br/>preemption"]
            kv["KVCacheManager<br/>paged blocks, prefix cache, LRU"]
        end

        subgraph runtime["runner + model (MPS)"]
            runner["ModelRunner<br/>batch packing, slot mapping,<br/>KV cache sizing"]
            model["Transformer<br/>fused qkv, RoPE, RMSNorm"]
            attn["Attention backends<br/>paged_sdpa, torch_ref"]
            samp["Sampler<br/>greedy, top-k, top-p"]
            plat["platform<br/>MPS / CPU helpers"]
        end
    end

    hf[("Hugging Face Hub<br/>safetensors, tokenizer")]

    client -->|"HTTP / SSE"| routers
    routers --> schemas
    routers --> deps --> aeng
    app --> aeng
    aeng --> eng
    eng --> sched --> kv
    eng --> runner --> model --> attn
    runner --> samp
    runner --> plat
    eng --> detok
    detok -->|"stream deltas"| aeng
    hf -.->|"weights at startup"| runner
```

### Streaming Request LifeCycle

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant R as FastAPI router
    participant A as AsyncEngine thread
    participant S as Scheduler + KV cache
    participant M as ModelRunner (MPS)

    C->>R: POST /v1/chat/completions (stream=true)
    R->>R: validate schema, apply chat template, tokenize
    alt prompt too long
        R-->>C: 400 error
    else valid
        R->>A: inbox.put(add, request)
        loop every engine step
            A->>S: schedule()
            S->>S: prefix cache lookup, allocate blocks, chunk prompt
            A->>M: run(batch)
            M->>M: write K/V, paged attention, sample
            M-->>A: next tokens
            A-->>R: Output via call_soon_threadsafe
            R-->>C: SSE chunk
        end
        A->>S: finish(), blocks return to LRU free list
        R-->>C: final chunk with usage, then DONE event
    end
```

### Engine

```mermaid
flowchart TB
    start(["Engine.step()"]) --> p1

    subgraph schedule["Scheduler.schedule()"]
        p1["Phase 1: running sequences<br/>reserve 1 decode token or next prefill chunk"]
        fit1{"KV blocks available?"}
        pre["Preempt newest running sequence<br/>free blocks, requeue at front"]
        p2["Phase 2: admit waiting (FIFO)<br/>only if nothing was preempted"]
        hit["Prefix cache lookup<br/>chained block hashes"]
        fit2{"allocate() ok?"}
        p1 --> fit1
        fit1 -->|"no"| pre --> fit1
        fit1 -->|"yes"| p2
        p2 --> hit --> fit2
        fit2 -->|"yes, try next"| p2
    end

    fit2 -->|"no, stop admitting"| batch
    p2 -->|"queue empty or budget spent"| batch

    batch["Batch of ScheduledSequence<br/>num_computed, num_new"] --> prep
    prep["ModelRunner.prepare()<br/>input_ids, positions, slot_mapping,<br/>query_start_loc, block_table"] --> fwd

    subgraph forward["Model forward, per layer"]
        fwd["qkv projection + RoPE"] --> write["index_copy_ new K/V into paged cache"]
        write --> kind{"all q_len == 1?"}
        kind -->|"decode"| dec["Batched paged SDPA<br/>single gather, padding mask"]
        kind -->|"prefill / mixed"| pf["Per-sequence SDPA<br/>bottom-right causal mask"]
        dec --> mlp["o_proj + MLP"]
        pf --> mlp
    end

    mlp --> samp["Sampler<br/>greedy, top-k, top-p"]
    samp --> done{"completes prompt?"}
    done -->|"no, mid-prompt chunk"| drop["Discard sample"]
    done -->|"yes"| commit["Append token, detokenize"]
    commit --> stop{"EOS or max_tokens?"}
    stop -->|"yes"| fin["finish(): free blocks to LRU<br/>hashes stay reusable"]
    stop -->|"no"| nextstep(["next step"])
    drop --> nextstep
    fin --> nextstep
```

### Packages

```mermaid
flowchart LR
    cli["cli<br/>serve, generate, bench"]
    api["api<br/>app, routers, schemas, deps, settings"]
    engine["engine<br/>Engine, AsyncEngine, Detokenizer"]
    core["core<br/>Sequence, KVCacheManager, Scheduler"]
    runner["runner<br/>ModelRunner, memory, sampler"]
    model["model<br/>layers, transformer, loader"]
    attention["attention<br/>paged_sdpa, torch_ref, metadata,<br/>forward_context"]
    platform["platform<br/>MPS / CPU helpers"]
    tests["tests/ (outside src)<br/>unit, model, api"]

    cli --> api
    cli --> engine
    api --> engine
    engine --> core
    engine --> runner
    runner --> model
    runner --> attention
    runner --> platform
    model --> attention
    tests -.-> api
    tests -.-> engine
    tests -.-> core
```

### References

- [Paged Attention](https://arxiv.org/abs/2309.06180)
- [Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM Scheduler High Level Sketch](https://audreywongkg.medium.com/understanding-vllm-scheduling-token-budgets-chunked-prefill-and-policies-2c879e3980e3)
- [Continuous Batching](https://huggingface.co/blog/continuous_batching)
- [Chunked Prefill](https://medium.com/@shirleyzhai521/how-chunked-prefill-improves-llm-serving-in-vllm-b3705dea557f)
- [SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

### Future Work
- [Speculative Decoding](https://research.google/blog/looking-back-at-speculative-decoding/)
- [Implementing Speculative Decoding in PyTorch](https://github.com/romsto/Speculative-Decoding)
