"""Question-independent video indexes and deterministic clip selection."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np


def positive(value, name, *, integer=False):
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0 or (integer and number != int(number)):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")
    return int(number) if integer else number


def read_window(*args, **kwargs):
    from playback.inference.media import read_window as read

    return read(*args, **kwargs)


def probe(path):
    from playback.inference.media import probe as inspect

    return inspect(path)


@dataclass(frozen=True)
class VideoSource:
    path: str
    start_us: int
    end_us: int

    @classmethod
    def resolve(cls, reference):
        start, end = 0, None
        if isinstance(reference, dict):
            start, end = reference.get("video_start", 0), reference.get("video_end")
            reference = reference.get("path") or reference.get("url")
        if not isinstance(reference, (str, Path)) or not str(reference).strip():
            raise ValueError("vanilla_rag requires a local video path or bounded video descriptor")
        parsed = urlparse(str(reference))
        if parsed.scheme not in ("", "file") or (parsed.scheme == "file" and parsed.netloc not in ("", "localhost")):
            raise ValueError("vanilla_rag requires finite local videos; download remote media first")
        path = Path(unquote(parsed.path) if parsed.scheme == "file" else str(reference)).expanduser().resolve(strict=True)
        duration = probe(path).duration_us
        end = duration / 1e6 if end is None else end
        if any(isinstance(t, bool) or not math.isfinite(float(t)) for t in (start, end)):
            raise ValueError("Video bounds must be finite seconds")
        start_us, end_us = round(float(start) * 1e6), round(float(end) * 1e6)
        if not 0 <= start_us < end_us <= duration:
            raise ValueError(f"Video bounds must satisfy 0 <= start < end <= {duration / 1e6:g} seconds")
        return cls(str(path), start_us, end_us)


def chunk_intervals(source, seconds):
    step = round(positive(seconds, "rag_chunk_seconds") * 1e6)
    if step < 1:
        raise ValueError("Chunks must be at least one microsecond")
    return np.asarray([(start, min(start + step, source.end_us)) for start in range(source.start_us, source.end_us, step)], dtype=np.int64)


def normalized(vectors, *, rows=None):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 768 or (rows is not None and len(vectors) != rows):
        raise ValueError("EmbeddingGemma must produce one 768-dimensional vector per input")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("EmbeddingGemma returned nonfinite or zero embeddings")
    return vectors / norms


def select_chunks(intervals, top_k, *, vectors=None, query=None):
    count = min(positive(top_k, "rag_top_k", integer=True), len(intervals))
    if not count:
        raise ValueError("The task supplied no video chunks")
    if vectors is None:
        indices = np.linspace(0, len(intervals) - 1, count).round().astype(int)
        scores = None
    else:
        scores = normalized(vectors, rows=len(intervals)) @ normalized(query, rows=1)[0]
        indices = np.lexsort((intervals[:, 0], -scores))[:count]
    ranks = {int(index): rank + 1 for rank, index in enumerate(indices)}
    return [{"chunk_id": int(i), "interval": [float(t) / 1e6 for t in intervals[i]], "rank": ranks[int(i)], "score": None if scores is None else float(scores[i])} for i in sorted(indices, key=lambda i: intervals[i, 0])]


def file_signature(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=128)
def _digest(path, signature):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if file_signature(path) != signature:
        raise RuntimeError("Source video changed while fingerprinting")
    return digest.hexdigest()


class VideoRetriever:
    def __init__(self, cache_dir, *, encoder=None, chunk_seconds=10, embed_fps=1, embed_max_frames=32, embed_batch_size=4):
        self.cache_dir = Path(cache_dir).expanduser()
        self.encoder = encoder
        self.cancelled = threading.Event()
        self.chunk_seconds = positive(chunk_seconds, "rag_chunk_seconds")
        self.embed_fps = positive(embed_fps, "rag_embed_fps")
        self.embed_max_frames = positive(embed_max_frames, "rag_embed_max_frames", integer=True)
        self.embed_batch_size = positive(embed_batch_size, "rag_embed_batch_size", integer=True)
        # 140 soft tokens per frame plus template overhead, within 8192 tokens.
        if self.embed_max_frames > 56:
            raise ValueError("rag_embed_max_frames must be <= 56 at the fixed 140-token frame budget")

    def _index(self, source, intervals):
        started = time.monotonic()
        signature = file_signature(source.path)
        profile = self.encoder.profile()
        manifest = {
            "version": 1,
            "source": source.path,
            "source_sha256": _digest(source.path, signature),
            "bounds_us": [source.start_us, source.end_us],
            "encoder": profile,
            "chunk_seconds": self.chunk_seconds,
            "fps": self.embed_fps,
            "max_frames": self.embed_max_frames,
            "longest_edge": 768,
            "max_soft_tokens": 140,
            "video_prompt": "",
            "add_timestamps": False,
            "sampler": "playback.read_window.v1",
        }
        serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(serialized.encode()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / f"{key}.npz"
        with (self.cache_dir / f"{key}.lock").open("a") as lock:
            while True:
                if self.cancelled.is_set():
                    raise RuntimeError("Video indexing cancelled")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.cancelled.wait(0.1)
            if target.exists():
                try:
                    with target.open("rb") as archive, np.load(archive, allow_pickle=False) as saved:
                        if str(saved["manifest"]) != serialized or not np.array_equal(saved["intervals"], intervals):
                            raise ValueError("Index metadata mismatch")
                        vectors = np.asarray(saved["vectors"], dtype=np.float32)
                        # Validate without another rounding step: cold and warm
                        # queries must score the same stored vector bytes.
                        normalized(vectors, rows=len(intervals))
                    if file_signature(source.path) != signature:
                        raise RuntimeError("Source video changed while reading its index")
                    return vectors, {"cache_key": key, "cache_hit": True, "index_build_s": 0.0, "index_lookup_s": time.monotonic() - started, "encoder": profile}
                except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                    # An invalid artifact is replaced atomically after a complete rebuild.
                    pass
            lookup_s = time.monotonic() - started
            building = time.monotonic()
            batches = []
            for offset in range(0, len(intervals), self.embed_batch_size):
                if self.cancelled.is_set():
                    raise RuntimeError("Video indexing cancelled")
                frames = [read_window(source.path, int(start), int(end), fps=self.embed_fps, max_frames=self.embed_max_frames, longest_edge=768)[0] for start, end in intervals[offset : offset + self.embed_batch_size]]
                batches.append(normalized(self.encoder.encode_videos(frames), rows=len(frames)))
            vectors = np.concatenate(batches)
            if file_signature(source.path) != signature:
                raise RuntimeError("Source video changed while indexing")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, prefix=f".{key}.", suffix=".tmp", delete=False) as output:
                    temporary = Path(output.name)
                    np.savez(output, manifest=serialized, intervals=intervals, vectors=vectors)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return vectors, {"cache_key": key, "cache_hit": False, "index_build_s": time.monotonic() - building, "index_lookup_s": lookup_s, "encoder": profile}

    def retrieve(self, source, question, top_k, selection="dense"):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The retrieval question must be nonempty text")
        if selection not in ("dense", "uniform"):
            raise ValueError("rag_selection must be dense or uniform")
        intervals = chunk_intervals(source, self.chunk_seconds)
        vectors, query = None, None
        trace = {"selection": selection, "num_chunks": len(intervals), "cache_hit": False, "index_build_s": 0.0, "index_lookup_s": 0.0}
        if selection == "dense":
            if self.encoder is None:
                raise ValueError("Dense retrieval requires an encoder")
            vectors, metadata = self._index(source, intervals)
            trace.update(metadata)
        search_start = time.monotonic()
        if selection == "dense":
            query = self.encoder.encode_query(question)
        selected = select_chunks(intervals, top_k, vectors=vectors, query=query)
        trace.update(search_s=time.monotonic() - search_start, selected=selected, source=source.path, bounds=[source.start_us / 1e6, source.end_us / 1e6])
        return selected, trace
