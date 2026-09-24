"""Explicit failures that must not be converted into model predictions."""


class FatalEvaluationError(RuntimeError):
    """Abort the evaluation even when ordinary request errors are nonfatal."""
