import torch

def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_device_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def device_memory(device: torch.device) -> tuple[int, int]:
    """Return (recommended_total_bytes, currently_allocated_bytes)."""
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        return total, allocated

    if device.type == "mps":
        return (
            torch.mps.recommended_max_memory(),
            torch.mps.current_allocated_memory(),
        )

    # CPU is useful for debugging, but choose a conservative explicit budget.
    return 8 * 1024**3, 0
