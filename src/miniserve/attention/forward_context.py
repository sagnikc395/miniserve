from contextlib import contextmanager
from contextvars import ContextVar

from miniserve.attention.metadata import AttnMetadata

# Per-step batch descriptor: the runner sets it once per forward pass and every
# attention layer reads it back. A plain module global will not do, because
# `from miniserve.config import ATTN_META` copies the binding at import time and
# the layer then never sees an update.
ATTN_METADATA: ContextVar[AttnMetadata | None] = ContextVar(
    "attn_metadata", default=None
)


@contextmanager
def set_forward_context(meta: AttnMetadata | None):
    """Bind meta for the duration of one forward pass."""
    token = ATTN_METADATA.set(meta)
    try:
        yield meta
    finally:
        ATTN_METADATA.reset(token)


def get_attn_metadata() -> AttnMetadata | None:
    """Metadata for the forward pass in flight, or None during warmup."""
    return ATTN_METADATA.get()
