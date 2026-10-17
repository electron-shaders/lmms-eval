"""Transport sampled clips through vLLM's native video processor.

Each payload contains JPEG frames and its own absolute timestamps, so clips in
the same request may have different intervals and sampling rates. The loader
never samples again. Qwen's native video processor subsequently groups temporal
patches (normally frame pairs), uses their midpoint timestamps, and formats
those timestamps to one decimal place. Exact original times remain in the
retrieval trace. A one-frame clip is duplicated only by the loader because the
native processor requires at least two frames.

Importing this module does not import vLLM. ``register_video_loader`` must run
in the serving process before using the ``vanilla_rag_clip`` video backend.
"""

from __future__ import annotations

import base64
import io
import json
import math
import zipfile
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

VIDEO_BACKEND = "vanilla_rag_clip"
VIDEO_MIME_TYPE = "video/x-lmms-rag"
VIDEO_DATA_URL_PREFIX = f"data:{VIDEO_MIME_TYPE};base64,"
_TIMEBASE = 1_000_000
_MANIFEST = "manifest.json"
_FORMAT_VERSION = 1


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _seconds_to_us(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Clip times must be finite non-negative numbers")
    scaled = float(value) * _TIMEBASE
    if not math.isfinite(scaled) or scaled < 0:
        raise ValueError("Clip times must be finite non-negative numbers")
    return round(scaled)


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or not _integer(manifest.get("version")) or manifest["version"] != _FORMAT_VERSION:
        raise ValueError("Unsupported vanilla RAG clip format version")
    shape = manifest.get("shape")
    if not isinstance(shape, list) or len(shape) != 4 or not all(_integer(n) and n > 0 for n in shape):
        raise ValueError("Clip shape must be positive TCHW dimensions")
    if shape[1] != 3:
        raise ValueError("Clip frames must have three RGB channels")
    timestamps = manifest.get("timestamps_us")
    if not isinstance(timestamps, list) or len(timestamps) != shape[0]:
        raise ValueError("Clip timestamp count must match frame count")
    if not all(_integer(t) and t >= 0 for t in timestamps):
        raise ValueError("Clip timestamps must be non-negative integer microseconds")
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Clip timestamps must be ordered")
    interval = manifest.get("interval_us")
    if not isinstance(interval, list) or len(interval) != 2 or not all(_integer(t) and t >= 0 for t in interval) or interval[1] <= interval[0]:
        raise ValueError("Clip interval must satisfy 0 <= start < end")
    if timestamps[0] < interval[0] or timestamps[-1] >= interval[1]:
        raise ValueError("Clip timestamps must lie inside its half-open interval")
    try:
        finite = math.isfinite(interval[1] / _TIMEBASE)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("Clip times must be finite")
    return shape, timestamps, interval


def _frame_name(index):
    return f"frames/{index:08d}.jpg"


def _write_entry(archive, name, data):
    # Fixed ZIP metadata makes equal clips byte-identical for vLLM media caches.
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED  # JPEG is already compressed.
    archive.writestr(info, data)


def encode_clip_video(frames: torch.Tensor, timestamps, interval) -> str:
    """Encode uint8 TCHW frames as one native ``video_url`` data URI.

    Preserve all sampled frames; timestamps and interval use source-video
    seconds. No source-video decoding or additional sampling occurs.
    Frames use JPEG quality 85.
    """
    import torch
    from PIL import Image

    if not isinstance(frames, torch.Tensor) or frames.dtype != torch.uint8 or frames.ndim != 4:
        raise ValueError("Clip frames must be a uint8 TCHW tensor")
    try:
        manifest = {
            "version": _FORMAT_VERSION,
            "shape": list(frames.shape),
            "timestamps_us": [_seconds_to_us(t) for t in timestamps],
            "interval_us": [_seconds_to_us(t) for t in interval],
        }
    except TypeError as exc:
        raise ValueError("Clip timestamps and interval must be sequences") from exc
    _validate_manifest(manifest)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        _write_entry(archive, _MANIFEST, json.dumps(manifest, separators=(",", ":")).encode("utf-8"))
        for index, frame in enumerate(frames.detach().cpu()):
            image = Image.fromarray(frame.permute(1, 2, 0).numpy())
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=85)
            _write_entry(archive, _frame_name(index), encoded.getvalue())
    return VIDEO_DATA_URL_PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")


class VanillaRAGClipVideoLoader:
    """vLLM video-loader contract, also usable directly for CPU validation."""

    @classmethod
    def load_bytes(cls, data: bytes, num_frames: int = -1, **kwargs):
        """Return native THWC RGB pixels and clip-specific video metadata.

        ``num_frames`` and other media-I/O sampling settings are deliberately
        ignored: vanilla RAG has already selected every supplied frame.
        """
        import numpy as np
        from PIL import Image

        if not isinstance(data, bytes):
            raise ValueError("vanilla RAG clip payload must contain bytes")
        try:
            with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
                if len(names) != len(set(names)) or _MANIFEST not in names:
                    raise ValueError("Clip archive must have one manifest and unique frame entries")
                if any(entry.compress_type != zipfile.ZIP_STORED or entry.flag_bits & 1 for entry in entries):
                    raise ValueError("Clip archive entries must be stored without ZIP compression or encryption")
                manifest = json.loads(archive.read(_MANIFEST))
                shape, timestamps, interval = _validate_manifest(manifest)
                count, _, height, width = shape
                if len(entries) != count + 1 or set(names) != {_MANIFEST, *(_frame_name(i) for i in range(count))}:
                    raise ValueError("Clip archive frame entries must match its declared shape")
                frames = []
                for index in range(count):
                    with Image.open(io.BytesIO(archive.read(_frame_name(index)))) as image:
                        if image.format != "JPEG" or image.size != (width, height):
                            raise ValueError("Clip frame encoding or dimensions do not match its manifest")
                        frames.append(np.asarray(image.convert("RGB")))
        except (
            zipfile.BadZipFile,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            OSError,
            Image.DecompressionBombError,
        ) as exc:
            raise ValueError("Malformed vanilla RAG clip payload") from exc

        if count == 1:
            frames.append(frames[0])
            timestamps = timestamps + timestamps
        total_num_frames = max(interval[1], timestamps[-1] + 1, len(frames))
        metadata = {
            "total_num_frames": total_num_frames,
            "fps": float(_TIMEBASE),
            "duration": total_num_frames / _TIMEBASE,
            "width": width,
            "height": height,
            "video_backend": VIDEO_BACKEND,
            "frames_indices": timestamps,
            "do_sample_frames": False,
        }
        return np.stack(frames), metadata


def register_video_loader():
    """Register the backend in the current vLLM process; safe to call again."""
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY

    VIDEO_LOADER_REGISTRY.register(VIDEO_BACKEND)(VanillaRAGClipVideoLoader)
