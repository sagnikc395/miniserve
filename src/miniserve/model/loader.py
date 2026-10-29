import os
import glob
import torch
import warnings
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from src.miniserve.model.transformer import Model


def load_model(
    name: str,
    dtype,
    max_position: int,
    tp_rank: int = 0,
    tp_size: int = 1,
    expert_parallel: bool = False,
):
    # a utility function to load weights into our model (this rank's shard of them)
    path = (
        name
        if os.path.isdir(name)
        else snapshot_download(
            name, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"]
        )
    )

    cfg = AutoConfig.from_pretrained(path)

    if tp_size > 1:
        cfg = shard_config(cfg, tp_size, tp_rank, expert_parallel)
    torch.set_default_dtype(dtype)
    with torch.device("cuda"):
        model = Model(cfg, max_position)

    torch.set_default_dtype(torch.float32)
    params = dict(model.named_parameters())
    loaded, unused = set(), []
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, "pt", device=f"cuda:{torch.cuda.current_device()}") as fh:
                for key in fh.keys():
                    name_ = key.removeprefix("model.")

                    # MoE -> seperate tensors per expert go into the layer's stacked w13/ w2
                    if m := EXPERT_KEY.fullmatch(name_):
                        moe = model.layers[int(m[1])].mlp
                        if moe.load_expert(int(m[2]), m[3], fh, key, tp_rank, tp_size):
                            loaded.add(
                                f"layers.{m[1]}.mlp.{'w2' if m[3] == 'down_proj' else 'w13'}"
                            )

                    # q/k/v and gate/up : each goes into its row range of the fused projection.
                    # load shard still sees the original name, so under tensor parallelism it
                    # returns this GPU's slice of q (or k, ...) and each GPU's fused weights ends up
                    # [its q heads; its k heads; its v heads] , never a cu of the stack
                    elif m := FUSED_KEY.fullmatch(name_):
                        module = model.get_submodule(m[1])
                        fused, part = FUSED_PARTS[m[2]]
                        sizes = (
                            module.qkv_sizes
                            if fused == "qkv_proj"
                            else module.gate_up_sizes
                        )
                        start = sum(sizes[:part])
                        getattr(getatr(module, fused), m[3])[
                            start : start + sizes[part]
                        ].copy_(load_shard(fh, key, tp_rank, tp_size))
                        loaded.add(f"{m[1]}.{fused}.{m[3]}")
                    elif name_ in params:
                        params[name_].copy_(load_shard(fh, key, tp_rank, to_size))
                        loaded.add(name_)
                    else:
                        unused.append(key)

    # weigh the checkpoint never filled would silently stay random , so fail here
    # this is usually a new architeture whose names dont match ours
    missing = sorted(set(params) - loaded)
    assert not missing, f"{len(missing)} parameters not found in the checkpoints e.g. {missing[:3]}"

    # the other directions : checkpoint tensors we have no place for. harmless for a tied lm_head
    # but eg: Qwen1.5-MoE's shared_expert would be silently dropped
    if unused and tp_rank == 0:
        warnings.warn(f"ignored {len(unused)} checkpoint tensors, e.g {unused[:3]}")
    return model.eval()

def to_gpu(data, dtype=None) -> torch.Tensor:
    ## host data (list or numpy) -> GPU tensor
    t = torch.as_tensor(data, dtype=dtype)
    return t.pin_memory().to("mps", non_blocking=True)
