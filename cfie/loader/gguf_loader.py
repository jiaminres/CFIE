"""GGUF loader（Phase 1 未启用）。"""

from __future__ import annotations


class GGUFLoaderNotImplementedError(NotImplementedError):
    """GGUF 在后续阶段接入。"""


def load_gguf_model(*args, **kwargs):
    del args, kwargs
    raise GGUFLoaderNotImplementedError("GGUF loader is not implemented in Phase 1")
