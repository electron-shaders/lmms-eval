"""Readable, payload-free diagnostics for incomplete Playback conversations."""

from __future__ import annotations

import errno
import json
import os
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

_MISSING = object()
_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_POLL_SECONDS = 0.05
_MEDIA_KEYS = {
    "audio",
    "audio_url",
    "input_audio",
    "image",
    "image_url",
    "video_url",
    "frames",
    "pixel_values",
    "binary",
    "bytes",
    "data",
    "payload",
    "base64",
    "b64_json",
    "safetensors",
    "tensor",
    "tensors",
}
_TOKEN_KEYS = {"input_ids", "output_ids", "prompt_ids", "completion_ids", "generated_ids", "raw_tokens", "logprobs"}
_DATA_URI = re.compile(r"data:[^\s,;]*[;][^\s,]*,[^\s\"'<>)]*|data:[^\s,]*,[^\s\"'<>)]*", re.IGNORECASE)
_URL = re.compile(r"(?:https?|s3|gs|hf|ftp)://[^\s<>\"']+", re.IGNORECASE)
_SECRET_NAME = r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|authorization|cookie|session(?:[_-]?id)?)"
_INLINE_SECRET = re.compile(rf"(?i)(\b{_SECRET_NAME}\b[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}}]+)")
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s,;\"'}]+")
_INLINE_TOKEN_IDS = re.compile(r"(?i)\b(?:[a-z]+_)*(?:token_ids|input_ids|output_ids|prompt_ids|completion_ids)\b[\"']?\s*[:=]\s*\[[^\]]*\]")


def _key(value: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value).lower().replace("-", "_")


def _sensitive(name: str) -> bool:
    return (
        any(
            part in name
            for part in (
                "session",
                "credential",
                "password",
                "passwd",
                "secret",
                "authorization",
                "cookie",
                "api_key",
                "apikey",
                "private_key",
            )
        )
        or name in {"token", "key", "auth"}
        or name.endswith("_token")
    )


def _clean_url(match: re.Match) -> str:
    try:
        parsed = urlsplit(match.group())
        return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
    except ValueError:
        return "[URL omitted]"


def _text(value: str) -> str:
    value = _DATA_URI.sub("[data URI omitted]", value)
    value = _URL.sub(_clean_url, value)
    value = _BEARER.sub("Bearer [redacted]", value)
    value = _INLINE_SECRET.sub(lambda match: match.group(1) + "[redacted]", value)
    return _INLINE_TOKEN_IDS.sub("[token IDs omitted]", value)


def _read(value: Any, name: str, default: Any = _MISSING) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _sanitize(value: Any, seen: set[int] | None = None) -> Any:
    """Copy safe values without asking tensors or opaque objects for their repr."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        # Tool arguments/results often carry structured data inside JSON strings.
        if value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except (ValueError, RecursionError):
                pass
            else:
                if isinstance(decoded, (dict, list)):
                    return _sanitize(decoded, seen)
        return _text(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[binary omitted]"
    if isinstance(value, os.PathLike):
        return _text(os.fspath(value))

    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return "[circular reference omitted]"
    seen.add(identity)
    try:
        if isinstance(value, (list, tuple)):
            return [_sanitize(item, seen) for item in value]
        if not isinstance(value, Mapping):
            # OpenAI Pydantic models and SimpleNamespace both expose fields here.
            # Do not serialize unknown objects (notably tensors and media objects).
            module = type(value).__module__
            if module.startswith(("torch", "numpy", "PIL", "safetensors")):
                return f"[{type(value).__name__} omitted]"
            attributes = getattr(value, "__dict__", None)
            if not isinstance(attributes, dict):
                return f"[{type(value).__name__} omitted]"
            value = {**attributes, **(getattr(value, "model_extra", None) or {})}
        media_type = str(value.get("type", "")).lower()
        is_media = any(kind in media_type for kind in ("image", "video", "audio"))
        result = {}
        for name, item in value.items():
            if not isinstance(name, str) or name.startswith("_") or callable(item):
                continue
            normalized = _key(name)
            if _sensitive(normalized):
                continue
            if normalized in _TOKEN_KEYS or "token_ids" in normalized or ("tokens" in normalized and isinstance(item, (list, tuple))):
                result[name] = "[token IDs omitted]"
            elif normalized in _MEDIA_KEYS or normalized.endswith(("_base64", "_safetensors")) or (is_media and normalized in {"url", "video", "source"}):
                result[name] = "[media payload omitted]"
            else:
                result[_text(name)] = _sanitize(item, seen)
        return result
    finally:
        seen.remove(identity)


def _fields(value: Any, names: tuple[str, ...]) -> dict:
    return {name: _sanitize(item) for name in names if (item := _read(value, name)) is not _MISSING}


def response_snapshot(response: Any) -> dict:
    """Immediately detach textual response data and usage from an SDK response.

    Supports mappings, OpenAI Pydantic models, and SimpleNamespace test doubles.
    The return value contains no raw response object or media/token payloads.
    """
    snapshot = _fields(
        response,
        (
            "id",
            "model",
            "created",
            "object",
            "service_tier",
            "system_fingerprint",
            "usage",
            "finish_reason",
            "stop_reason",
        ),
    )
    snapshot["choices"] = []
    for choice in _read(response, "choices", None) or []:
        saved = _fields(choice, ("index", "finish_reason", "stop_reason", "text"))
        message = _read(choice, "message", None)
        if message is not None:
            saved["message"] = _fields(message, ("role", "content", "reasoning", "reasoning_content", "refusal", "name", "tool_call_id"))
            calls = _read(message, "tool_calls")
            if calls is not _MISSING:
                saved["message"]["tool_calls"] = (
                    None
                    if calls is None
                    else [
                        {
                            **_fields(call, ("id", "type")),
                            "function": _fields(_read(call, "function", None), ("name", "arguments")),
                        }
                        for call in calls
                    ]
                )
        snapshot["choices"].append(saved)
    return snapshot


def _render(value: Any, indent: int = 2) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for name, item in value.items():
            if isinstance(item, (dict, list)) or isinstance(item, str) and "\n" in item:
                lines.append(f"{prefix}{name}:")
                lines.extend(_render(item, indent + 2))
            else:
                lines.append(f"{prefix}{name}: {item}")
        return lines or [prefix + "(empty)"]
    if isinstance(value, list):
        lines = []
        for index, item in enumerate(value, 1):
            lines.append(f"{prefix}[{index}]")
            lines.extend(_render(item, indent + 2))
        return lines or [prefix + "(empty)"]
    return [prefix + line for line in str(value).split("\n")]


@contextmanager
def _append_lock(destination: Path):
    """Use atomic mkdir across hosts, without invoking NFS's lock manager."""
    lock = destination.with_name(destination.name + ".lock")
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            lock.mkdir(mode=0o700)
            break
        except FileExistsError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    errno.ETIMEDOUT,
                    "Timed out waiting for the Playback error-log lock. Check for an active writer; " "if none is running, remove this stale lock directory before retrying",
                    str(lock),
                ) from None
            time.sleep(min(_LOCK_POLL_SECONDS, remaining))
    try:
        yield
    finally:
        # Only the successful mkdir owner reaches this cleanup. Never remove a
        # preexisting lock, even if its owner may have crashed.
        lock.rmdir()


def write_error_log(
    path: str | os.PathLike,
    *,
    context: dict,
    messages: list,
    responses: list[dict],
    tool_trace: list,
    error: BaseException | str,
) -> None:
    """Append one complete, readable record; propagate filesystem failures.

    Text is never truncated. Media, credentials, session fields, URL queries,
    and raw token arrays are redacted. Cooperating processes and hosts use an
    atomic directory lock through append, flush, fsync, and close. Lock waits
    time out after five seconds; stale locks require manual inspection/removal.
    """
    record_id = uuid4().hex
    error_text = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
    lines = [f"=== Playback error record {record_id} ===", f"Time (UTC): {datetime.now(timezone.utc).isoformat()}"]
    for heading, value in (
        ("Context", context),
        ("Error", error_text),
        (f"Messages ({len(messages)})", messages),
        (f"Responses ({len(responses)})", responses),
        (f"Tool trace ({len(tool_trace)})", tool_trace),
    ):
        lines.extend(["", heading + ":", *_render(_sanitize(value))])
    lines.extend([f"=== End Playback error record {record_id} ===", "", ""])
    record = "\n".join(lines)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _append_lock(destination):
        with destination.open("a", encoding="utf-8", errors="backslashreplace") as output:
            output.write(record)
            output.flush()
            os.fsync(output.fileno())
