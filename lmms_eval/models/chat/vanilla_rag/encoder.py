"""Lazy, serialized EmbeddingGemma 2 inference using Sentence Transformers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import threading
from pathlib import Path

from .core import normalized


def check_dependencies():
    from packaging.version import Version

    for package, minimum in (("transformers", "5.19.0"), ("sentence-transformers", "6.1.0")):
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = "0"
        if Version(installed) < Version(minimum):
            raise RuntimeError(f"vanilla_rag needs {package}>={minimum}; found {installed}. Use the separate RAG_PYTHON environment described in the baseline README.")
    from transformers import (  # noqa: F401
        EmbeddingGemma2Model,
        EmbeddingGemma2Processor,
    )


class EmbeddingGemmaEncoder:
    def __init__(self, model="google/embeddinggemma-2", revision=None, device="cuda:0"):
        self.model_id, self.revision, self.device = model, revision or None, device
        self._model = None
        self._profile = None
        self._lock = threading.RLock()

    def _load(self):
        if self._model is not None:
            return
        check_dependencies()
        import torch
        from sentence_transformers import SentenceTransformer

        device = torch.device(self.device)
        if device.type == "cuda":
            with torch.cuda.device(device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            dtype = torch.float32
        resolved = None
        if not Path(self.model_id).is_dir():
            from huggingface_hub import hf_hub_download

            # Recent Transformers versions need not retain _commit_hash on the
            # model config. Resolve the cached snapshot first, then load all
            # modules at that exact revision, including in offline runs.
            config_path = hf_hub_download(self.model_id, "config.json", revision=self.revision)
            resolved = Path(config_path).parent.name
            if not re.fullmatch(r"[0-9a-f]{40}", resolved):
                raise RuntimeError("Cannot determine the embedding checkpoint's Hub snapshot")
        model = SentenceTransformer(self.model_id, revision=resolved or self.revision, device=self.device, config_kwargs={"audio_config": None}, model_kwargs={"dtype": dtype}, trust_remote_code=False)
        model.eval()
        model.requires_grad_(False)
        config = model[0].auto_model.config
        if config.model_type != "embedding_gemma2":
            raise ValueError("rag_embed_model must be an EmbeddingGemma 2 checkpoint")
        if resolved is None and Path(self.model_id).is_dir():
            # Local checkpoints do not have Hub commits; fingerprint their artifacts.
            digest = hashlib.sha256()
            for path in sorted(Path(self.model_id).rglob("*")):
                if path.is_file():
                    stat = path.stat()
                    digest.update(json.dumps([str(path.relative_to(self.model_id)), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]).encode())
            resolved = digest.hexdigest()
        if resolved is None:
            raise RuntimeError("Cannot determine the loaded embedding model revision")
        self._profile = {
            "model": self.model_id,
            "revision": resolved,
            "dtype": str(dtype),
            "dimension": 768,
            "query_prompt": model.prompts["query"],
            "transformers": importlib.metadata.version("transformers"),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
        }
        self._model = model

    def profile(self):
        with self._lock:
            self._load()
            return dict(self._profile)

    def encode_query(self, question):
        with self._lock:
            self._load()
            # Reject an oversized question instead of silently changing retrieval.
            if len(self._model.tokenizer(self._model.prompts["query"] + question)["input_ids"]) > 8192:
                raise ValueError("Retrieval question exceeds EmbeddingGemma's 8192-token input limit")
            return normalized(self._model.encode_query([question], convert_to_tensor=True, convert_to_numpy=False, show_progress_bar=False), rows=1, device=self.device)

    def encode_videos(self, frames):
        with self._lock:
            self._load()
            # The media reader has already sampled each bounded interval. Disable
            # processor sampling and timestamps for these predecoded THWC arrays.
            result = self._model.encode(
                [{"video": video} for video in frames],
                batch_size=len(frames),
                convert_to_tensor=True,
                convert_to_numpy=False,
                show_progress_bar=False,
                processing_kwargs={"video": {"do_sample_frames": False, "fps": None, "max_frames": None, "overflow_strategy": None, "add_timestamps": False, "max_soft_tokens": 140}},
            )
            return normalized(result, rows=len(frames), device=self.device)

    def close(self):
        with self._lock:
            self._model = None
            self._profile = None
