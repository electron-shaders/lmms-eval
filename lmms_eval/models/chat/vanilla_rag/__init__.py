"""Single-pass dense video retrieval with a lazily imported lmms-eval adapter."""

__all__ = ["VanillaRAGModel"]


def __getattr__(name):
    # Launcher preflight and retrieval helpers must not import the API client.
    if name == "VanillaRAGModel":
        from .model import VanillaRAGModel

        return VanillaRAGModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
