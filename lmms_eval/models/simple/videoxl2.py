"""Modern lmms-eval adapter for Video-XL-2 bi-level inference.

The upstream Video-XL-2 repository bundles an old evaluator and tightly
couples selection lookup to prompt formatting.  This adapter reuses only its
model implementation while keeping task loading, result tracking, and
selection policy in the current evaluator.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from loguru import logger as eval_logger

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.experiment_metrics import log_workload


class SelectionCoverageError(RuntimeError):
    """Raised when strict bi-level inference has no selection for a document."""


def _flatten_selection_file(path: Path) -> dict[str, tuple[Any, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    flattened: dict[str, tuple[Any, Any]] = {}
    for category in payload.values():
        if not category:
            continue
        config = category.get("config")
        for key, indices in (category.get("selected_unit_indices") or {}).items():
            raw_key = str(key)
            flattened[raw_key] = (indices, config)
            flattened[raw_key.strip()] = (indices, config)
    return flattened


def _as_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, dict):
        paths = value.get("paths", value.get("path", []))
        return [paths] if isinstance(paths, str) else list(paths)
    raise ValueError(f"Invalid selection manifest entry: {value!r}")


class SelectionResolver:
    """Resolve precomputed selections from stable task/document fields."""

    def __init__(self, manifest_path: str, policy: str = "required") -> None:
        if policy not in {"required", "fallback_full"}:
            raise ValueError("selection_policy must be 'required' or 'fallback_full'")
        self.policy = policy
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Selection manifest not found: {self.manifest_path}")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        task_entries = manifest.get("tasks", manifest)
        if not isinstance(task_entries, dict):
            raise ValueError("Selection manifest must contain a 'tasks' mapping")
        self.task_maps: dict[str, dict[str, tuple[Any, Any]]] = {}
        self.task_paths: dict[str, list[str]] = {}
        for task_name, entry in task_entries.items():
            merged: dict[str, tuple[Any, Any]] = {}
            resolved_paths: list[str] = []
            for raw_path in _as_paths(entry):
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = self.manifest_path.parent / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Selection file for {task_name!r} not found: {path}")
                merged.update(_flatten_selection_file(path))
                resolved_paths.append(str(path))
            self.task_maps[str(task_name)] = merged
            self.task_paths[str(task_name)] = resolved_paths
        self._counts: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _video_names(video_path: Any, doc: dict[str, Any]) -> list[str]:
        values: list[str] = []
        if isinstance(video_path, (list, tuple)):
            values.extend(str(item) for item in video_path if isinstance(item, (str, os.PathLike)))
        elif isinstance(video_path, (str, os.PathLike)):
            values.append(str(video_path))
        for field in ("video", "video_path", "video_name", "videoPath"):
            value = doc.get(field)
            if isinstance(value, (str, os.PathLike)):
                values.append(str(value))
        names: list[str] = []
        for value in values:
            basename = Path(value).name
            for candidate in (basename, Path(basename).stem):
                if candidate and candidate not in names:
                    names.append(candidate)
        return names

    @staticmethod
    def _document_texts(doc: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        for field in ("question", "question_text", "query", "Q"):
            value = doc.get(field)
            if isinstance(value, str) and value.strip():
                normalized = value.strip()
                variants = [normalized]
                # Some evaluator documents store the choices in the question
                # field. Selection metadata uses the stable question stem.
                for marker in ("\n(A)", "\nA.", "\nA)"):
                    if marker in normalized:
                        variants.append(normalized.split(marker, 1)[0].strip())
                for variant in variants:
                    if variant and variant not in texts:
                        texts.append(variant)
        return texts

    def candidate_keys(
        self,
        task_name: str,
        doc_id: Any,
        doc: dict[str, Any],
        video_path: Any,
    ) -> list[str]:
        names = self._video_names(video_path, doc)
        texts = self._document_texts(doc)
        doc_ids = [str(doc_id)]
        for field in ("id", "question_id", "qid", "uid", "unique_id"):
            value = doc.get(field)
            if value is not None and str(value) not in doc_ids:
                doc_ids.append(str(value))
        candidates: list[str] = []
        for value in [*doc_ids, *texts]:
            if value not in candidates:
                candidates.append(value)
        for name in names:
            for value in [*texts, *doc_ids]:
                key = f"{name}_{value}"
                if key not in candidates:
                    candidates.append(key)
        return candidates

    def resolve(
        self,
        task_name: str,
        doc_id: Any,
        doc: dict[str, Any],
        video_path: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        selection_map = self.task_maps.get(task_name)
        candidates = self.candidate_keys(task_name, doc_id, doc, video_path)
        match = next((key for key in candidates if selection_map and key in selection_map), None)
        with self._lock:
            counts = self._counts.setdefault(task_name, {"hits": 0, "misses": 0, "fallbacks": 0})
            if match is not None:
                counts["hits"] += 1
            else:
                counts["misses"] += 1
        if match is not None:
            indices, config = selection_map[match]
            return indices, config, {
                "selection_mode": "bi_level",
                "selection_hit": True,
                "selection_key": match,
                "selection_files": self.task_paths.get(task_name, []),
            }
        if self.policy == "required":
            reason = "task has no manifest entry" if selection_map is None else "document key was not found"
            preview = ", ".join(repr(key) for key in candidates[:5])
            raise SelectionCoverageError(
                f"Video-XL-2 selection required for task={task_name!r}, doc_id={doc_id!r}: "
                f"{reason}. Candidate keys: {preview}"
            )
        with self._lock:
            self._counts[task_name]["fallbacks"] += 1
        return None, None, {
            "selection_mode": "full_attention_fallback",
            "selection_hit": False,
            "selection_key": None,
            "selection_files": self.task_paths.get(task_name, []),
        }

    def coverage_summary(self) -> dict[str, dict[str, int | float]]:
        with self._lock:
            counts = {task: values.copy() for task, values in self._counts.items()}
        for values in counts.values():
            total = values["hits"] + values["misses"]
            values["coverage"] = values["hits"] / total if total else 0.0
        return counts


def _load_legacy_adapter(code_path: Path):
    code_path = code_path.expanduser().resolve()
    source = code_path / "lmms_eval" / "models" / "videoxl2.py"
    package_root = code_path / "videoxl2"
    if not source.is_file() or not package_root.is_dir():
        raise FileNotFoundError(
            "videoxl2_code_path must point to the w_chunk_bilevel directory "
            f"(missing {source} or {package_root})"
        )
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))
    module_name = "_lmms_eval_videoxl2_bilevel_legacy"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Video-XL-2 adapter from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    legacy_class = module.Videoxl2
    if "generate_until_multi_round" in getattr(legacy_class, "__abstractmethods__", ()):
        # The bundled Video-XL-2 evaluator predates the multi-round method on
        # lmms. Because it imports the already-loaded modern lmms base class,
        # Python otherwise considers the legacy adapter abstract and refuses
        # to instantiate it. Keep the compatibility behavior local to this
        # integration rather than modifying the external checkout.
        class CompatibleVideoxl2(legacy_class):
            def generate_until_multi_round(self, requests):
                raise NotImplementedError("Video-XL-2 does not support multi-round generation")

        CompatibleVideoxl2.__name__ = legacy_class.__name__
        CompatibleVideoxl2.__qualname__ = legacy_class.__qualname__
        legacy_class = CompatibleVideoxl2

    return legacy_class


def _count_selected_units(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return sum(_count_selected_units(item) for item in value) if value and isinstance(next(iter(value)), (list, tuple, set)) else len(value)
    return 1


@register_model("videoxl2")
class VideoXL2(lmms):
    """Video-XL-2 bi-level model hosted by the current evaluator."""

    is_simple = True

    def __init__(
        self,
        pretrained: str = "BAAI/Video-XL-2",
        videoxl2_code_path: str = "",
        selection_manifest: str = "",
        selection_policy: str = "required",
        block_size_chosed: int = 4,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()
        if not videoxl2_code_path:
            raise ValueError("videoxl2_code_path is required")
        if not selection_manifest:
            raise ValueError("selection_manifest is required")
        self.selection_resolver = SelectionResolver(selection_manifest, selection_policy)
        self.block_size_chosed = int(block_size_chosed)
        legacy_class = _load_legacy_adapter(Path(videoxl2_code_path))
        legacy_kwargs.pop("max_batch_size", None)

        selection_paths = [path for paths in self.selection_resolver.task_paths.values() for path in paths]
        self._bootstrap_selection_file: Optional[str] = None
        if selection_paths:
            bootstrap_selection = selection_paths[0]
        else:
            handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump({}, handle)
            handle.close()
            self._bootstrap_selection_file = handle.name
            bootstrap_selection = handle.name

        self._impl = legacy_class(
            pretrained=pretrained,
            selected_info_file_path=bootstrap_selection,
            block_size_chosed=self.block_size_chosed,
            **legacy_kwargs,
        )
        self.accelerator = self._impl.accelerator
        self._device = self._impl.device
        self._rank = self._impl.rank
        self._world_size = self._impl.world_size
        self.batch_size_per_gpu = self._impl.batch_size
        self.chat_template = None
        eval_logger.info(
            "Video-XL-2 bi-level adapter loaded with selection_policy={} manifest={}",
            selection_policy,
            selection_manifest,
        )

    @property
    def device(self):
        return self._impl.device

    @property
    def config(self):
        return self._impl.config

    @property
    def tokenizer(self):
        return self._impl.tokenizer

    @property
    def max_length(self):
        return self._impl.max_length

    @property
    def eot_token_id(self):
        return self._impl.eot_token_id

    @property
    def batch_size(self):
        return self._impl.batch_size

    def _document_and_video(self, request: Instance) -> tuple[str, Any, dict[str, Any], Any]:
        _, _, doc_to_visual, doc_id, request_task, split = request.args
        task = request.task_name or request_task
        doc = self.task_dict[task][split][doc_id]
        visuals = doc_to_visual(doc)
        video_path = visuals[0] if isinstance(visuals, (list, tuple)) and visuals else visuals
        return task, doc_id, doc, video_path

    def generate_until(self, requests: Iterable[Instance]) -> list[GenerationResult]:
        requests = list(requests)
        prepared: list[tuple[Instance, str, Any, Any, Any, dict[str, Any]]] = []
        # Resolve every document before the first model call. In strict mode this
        # guarantees incomplete coverage cannot produce a partial evaluation.
        for request in requests:
            task, doc_id, doc, video_path = self._document_and_video(request)
            try:
                selected_indices, selected_config, selection_meta = self.selection_resolver.resolve(
                    task, doc_id, doc, video_path
                )
            except SelectionCoverageError:
                log_workload(
                    task_name=task,
                    doc_id=doc_id,
                    workload={"selection_mode": "missing_required", "selection_hit": False},
                )
                raise
            prepared.append((request, task, doc_id, selected_indices, selected_config, selection_meta))

        results: list[GenerationResult] = []
        self._impl.task_dict = self.task_dict
        self._impl.cache_hook = self.cache_hook
        for request, task, doc_id, selected_indices, selected_config, selection_meta in prepared:
            decoded_frames = 0
            output_tokens: Optional[int] = None
            original_selector = self._impl.get_selected_unit_indices
            original_load_video = self._impl.load_video
            original_batch_decode = self._impl.tokenizer.batch_decode

            def fixed_selector(_prompt: str, _video_path: Any):
                return selected_indices, selected_config

            def measured_load_video(*args: Any, **kwargs: Any):
                nonlocal decoded_frames
                frames, timestamps = original_load_video(*args, **kwargs)
                decoded_frames = len(timestamps)
                return frames, timestamps

            def measured_batch_decode(token_ids: Any, *args: Any, **kwargs: Any):
                nonlocal output_tokens
                try:
                    first = token_ids[0]
                    output_tokens = int(first.numel() if hasattr(first, "numel") else len(first))
                except Exception:
                    output_tokens = None
                return original_batch_decode(token_ids, *args, **kwargs)

            self._impl.get_selected_unit_indices = fixed_selector
            self._impl.load_video = measured_load_video
            self._impl.tokenizer.batch_decode = measured_batch_decode
            started = time.perf_counter()
            try:
                text = self._impl.generate_until([request])[0]
            finally:
                self._impl.get_selected_unit_indices = original_selector
                self._impl.load_video = original_load_video
                self._impl.tokenizer.batch_decode = original_batch_decode
            latency = time.perf_counter() - started
            selected_units = _count_selected_units(selected_indices)
            selected_frames = min(decoded_frames, selected_units * self.block_size_chosed) if selected_indices is not None else decoded_frames
            workload = {
                **selection_meta,
                "latency_s": latency,
                "decoded_frames": decoded_frames,
                "selected_frames": selected_frames,
                "visual_tokens": ((decoded_frames + 3) // 4) * 144 if decoded_frames else None,
                "selected_units": selected_units,
                "selected_unit_indices": selected_indices,
                "selection_config": selected_config,
                "selection_coverage": self.selection_resolver.coverage_summary().get(task),
            }
            results.append(
                GenerationResult(
                    text=text,
                    token_counts=TokenCounts(output_tokens=output_tokens),
                    workload=workload,
                )
            )
        return results

    def loglikelihood(self, requests):
        self._impl.task_dict = self.task_dict
        return self._impl.loglikelihood(requests)

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Video-XL-2 does not support multi-round generation")

    def clean(self):
        try:
            self._impl.clean()
        finally:
            if self._bootstrap_selection_file:
                Path(self._bootstrap_selection_file).unlink(missing_ok=True)
