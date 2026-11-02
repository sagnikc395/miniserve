"""Command line entry point for miniserve."""

import sys
from pathlib import Path

import click
import torch

from miniserve.api.settings import Settings
from miniserve.config import EngineConfig

DTYPES = {
    "auto": None,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _config(**kw) -> EngineConfig:
    return EngineConfig(
        model=kw["model"],
        device=kw["device"],
        dtype=DTYPES[kw["dtype"]],
        max_model_len=kw["max_model_len"],
        max_num_seqs=kw["max_num_seqs"],
        max_num_batched_tokens=kw["max_num_batched_tokens"],
        gpu_memory_utilization=kw["gpu_memory_utilization"],
        attention_backend=kw["attention_backend"],
        block_size=kw["block_size"],
        enable_prefix_caching=kw["prefix_caching"],
        use_cuda_graphs=kw["cuda_graphs"],
    )


def _common_options(fn):
    """Options shared by serve, generate and bench."""
    options = [
        click.option(
            "--model",
            envvar="MINISERVE_MODEL",
            default=Settings.model,
            show_default=True,
            help="Hugging Face model id or a local checkpoint directory.",
        ),
        click.option("--device", envvar="MINISERVE_DEVICE", default="auto"),
        click.option(
            "--dtype",
            type=click.Choice(list(DTYPES), case_sensitive=False),
            default="auto",
        ),
        click.option("--max-model-len", default=4096, show_default=True),
        click.option("--max-num-seqs", default=16, show_default=True),
        click.option("--max-num-batched-tokens", default=1024, show_default=True),
        click.option(
            "--gpu-memory-utilization", default=0.60, show_default=True
        ),
        click.option(
            "--attention-backend",
            type=click.Choice(["auto", "sdpa", "torch", "flash_attn"]),
            default="auto",
        ),
        click.option("--block-size", default=16, show_default=True),
        click.option(
            "--prefix-caching/--no-prefix-caching",
            default=True,
            show_default=True,
        ),
        click.option("--cuda-graphs/--no-cuda-graphs", default=False),
    ]
    for option in reversed(options):
        fn = option(fn)
    return fn


@click.group()
@click.version_option("0.1.0", prog_name="miniserve")
def main():
    """A tiny vLLM-style inference server."""


@main.command()
@_common_options
@click.option("--host", envvar="MINISERVE_HOST", default="127.0.0.1")
@click.option("--port", envvar="MINISERVE_PORT", default=8000, type=int)
@click.option("--log-level", envvar="MINISERVE_LOG_LEVEL", default="info")
def serve(**kwargs):
    """Start the OpenAI-compatible HTTP server."""
    cfg = _config(**kwargs)
    try:
        import uvicorn
    except ImportError as exc:
        raise click.ClickException(
            "uvicorn is required for serve; install it with `uv add uvicorn`"
        ) from exc

    from miniserve.api.app import create_app

    click.echo(
        f"[miniserve] serving {cfg.model} on http://{kwargs['host']}:"
        f"{kwargs['port']} ({cfg.device}, {cfg.attention_backend})"
    )
    uvicorn.run(
        create_app(cfg),
        host=kwargs["host"],
        port=kwargs["port"],
        log_level=kwargs["log_level"],
    )


@main.command()
@_common_options
@click.option("--prompt", "-p", required=True, help="Prompt text.")
@click.option("--max-tokens", default=128, show_default=True)
@click.option("--temperature", default=0.0, show_default=True)
@click.option("--top-p", default=1.0, show_default=True)
@click.option("--top-k", default=-1, show_default=True)
@click.option("--ignore-eos", is_flag=True)
@click.option("--chat", is_flag=True, help="Apply the model's chat template.")
@click.option("--stream/--no-stream", default=True, show_default=True)
def generate(**kwargs):
    """Generate text from a prompt on the command line."""
    from miniserve.engine.engine import Engine
    from miniserve.sampling import SamplingParams

    cfg = _config(**kwargs)
    prompt = kwargs["prompt"]

    engine = Engine(cfg)
    if kwargs["chat"]:
        messages = [{"role": "user", "content": prompt}]
        prompt = engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    params = SamplingParams(
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        top_k=kwargs["top_k"],
        max_tokens=kwargs["max_tokens"],
        ignore_eos=kwargs["ignore_eos"],
    )
    rid = engine.add_request(prompt, params)

    final = None
    while engine.has_work():
        for out in engine.step():
            if out.request_id != rid:
                continue
            final = out
            if kwargs["stream"] and out.delta:
                click.echo(out.delta, nl=False)
                sys.stdout.flush()

    if final is None:
        raise click.ClickException("generation produced no output")

    if not kwargs["stream"]:
        click.echo(final.text)
    else:
        click.echo()

    click.echo(
        f"[{len(final.token_ids)} output tokens, finish_reason="
        f"{final.finish_reason}, cached={final.num_cached_tokens}]",
        err=True,
    )


@main.command()
@_common_options
@click.option("--num-prompts", default=16, show_default=True)
@click.option("--max-tokens", default=64, show_default=True)
@click.option("--hf-baseline/--no-hf-baseline", default=False)
def bench(**kwargs):
    """Benchmark miniserve with and without prefix caching."""
    from miniserve.benchmarks.run import run_benchmark

    cfg = _config(**kwargs)
    run_benchmark(
        cfg,
        num_prompts=kwargs["num_prompts"],
        max_tokens=kwargs["max_tokens"],
        hf_baseline=kwargs["hf_baseline"],
    )


@main.command("attn-demo")
def attn_demo():
    """Compare the paged attention backends on a toy pool."""
    try:
        from demos import attention_demo
    except ImportError:
        # when installed as a console script, `demos/` is not on sys.path
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from demos import attention_demo

    attention_demo.demo()


@main.command("kv-demo")
def kv_demo_cmd():
    """Walk the KV cache manager by hand."""
    try:
        from demos import kv_demo
    except ImportError:
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from demos import kv_demo

    kv_demo.demo()
