"""
VGent model backend for lmms-eval.

Wraps the ``async_openai`` chat backend (``lmms_eval/models/chat/async_openai.py``)
and augments every request with VGent's graph-based retrieval-reasoning context
before forwarding to the vLLM OpenAI-compatible server.

Two operating modes
-------------------

**Pre-built mode** (default)
    Graphs are built offline before running lmms-eval::

        conda activate vgent
        cd Vgent && torchrun --nproc_per_node=1 -m vgent_graph \\
            --model_name lmms_eval_async_openai \\
            --task mlvu_dev \\
            --data_path ./data/MLVU

    Then evaluate::

        python -m lmms_eval \\
            --model vgent \\
            --model_args "base_url=http://localhost:8000/v1,model=Qwen/Qwen3.5-4B,vgent_graph_dir=./graphs" \\
            --tasks mlvu_dev

    A ``FileNotFoundError`` is raised if a graph is missing.

**On-demand (JIT) mode** (``vgent_build_on_demand=True``)
    Graphs are built automatically the first time each video is encountered.

    The workflow per sample is::

        lmms-eval downloads video  →  VGent builds graph  →  query graph
                                  (optionally delete video)

    Enable with::

        --model_args "...,vgent_build_on_demand=True,vgent_delete_video_after_build=True"

VGent graph directory layout
-----------------------------
    <vgent_graph_dir>/
        <task_name>/
            <video_id>/
                graph.pkl

Environment variables
----------------------
    VGENT_PATH            sys.path prefix for ``vgent_adapter`` import.
    VGENT_GRAPH_DIR       Overrides ``vgent_graph_dir`` model arg.
    VGENT_MAX_CLIPS       Overrides ``vgent_max_clips`` model arg (default: 8).
"""

from __future__ import annotations

import asyncio
import copy
import os
import sys
import threading
import types
from typing import List, Optional, Tuple

from loguru import logger as eval_logger

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.async_openai import AsyncOpenAIChat
from lmms_eval.protocol import ChatMessages


# ---------------------------------------------------------------------------
# Lazy VGent import helper
# ---------------------------------------------------------------------------

_vgent_loaded: bool = False
_run_vgent_query = None        # callable(graph_dir, query, max_clips) -> str
_build_graph_for_video = None  # callable(video_path, output_dir, model_name, task) -> str
_init_vgent_instance = None    # callable(model_name, task, openai_client, model_version) -> None

# Per-video build lock: prevents concurrent coroutines from building the same
# graph twice when the same video appears in multiple requests.
_jit_build_locks: dict[str, asyncio.Lock] = {}
_jit_build_locks_meta = threading.Lock()


def _get_jit_lock(key: str) -> threading.Lock:
    with _jit_build_locks_meta:
        if key not in _jit_build_locks:
            _jit_build_locks[key] = threading.Lock()
        return _jit_build_locks[key]


def _load_vgent(vgent_path: Optional[str]) -> bool:
    """
    Try to import ``vgent_adapter.run_vgent_query`` and, if available,
    ``vgent_adapter.build_graph_for_video`` (needed for JIT mode).

    Returns True if the query function was imported successfully.
    """
    global _vgent_loaded, _run_vgent_query, _build_graph_for_video, _init_vgent_instance

    if _vgent_loaded:
        return _run_vgent_query is not None

    search_paths = []
    if vgent_path:
        search_paths.append(vgent_path)
    env_path = os.environ.get("VGENT_PATH", "")
    if env_path:
        search_paths.append(env_path)

    for p in search_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import vgent_adapter as _va  # type: ignore[import]

        _run_vgent_query = _va.run_vgent_query
        _build_graph_for_video = getattr(_va, "build_graph_for_video", None)
        _init_vgent_instance = getattr(_va, "init_vgent_instance", None)
        _vgent_loaded = True
        if _build_graph_for_video is not None:
            eval_logger.info(
                "[VGent] Imported run_vgent_query + build_graph_for_video "
                "(JIT mode available)"
            )
        else:
            eval_logger.info(
                "[VGent] Imported run_vgent_query "
                "(build_graph_for_video not found — JIT mode unavailable)"
            )
        return True
    except ImportError as exc:
        _vgent_loaded = True
        eval_logger.warning(
            f"[VGent] Could not import vgent_adapter ({exc}). "
            "Set VGENT_PATH or pass vgent_path=<dir> in --model_args."
        )
        return False


# ---------------------------------------------------------------------------
# Helper: extract fields from raw messages
# ---------------------------------------------------------------------------

def _extract_video_path(raw_messages: list[dict]) -> str:
    """Return the video path from the *first* user message, or empty string."""
    for msg in raw_messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for c in content:
                if c.get("type") == "video":
                    return c.get("url", "")
    return ""


def _extract_user_text(raw_messages: list[dict]) -> str:
    """Return the concatenated text content of the *last* user message."""
    for msg in reversed(raw_messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return " ".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Helper: resolve graph path for a document
# ---------------------------------------------------------------------------

def _resolve_graph_path(
    graph_dir: str,
    task: str,
    doc: dict,
) -> Optional[str]:
    """
    Try two layouts:
      1. <graph_dir>/<task>/<video_id>/
      2. <graph_dir>/<video_id>/
    Returns the first existing path, or None.
    """
    video_id = (
        doc.get("video_id")
        or doc.get("video")
        or doc.get("videoID")
        or doc.get("id")
        or ""
    )
    if not video_id:
        return None

    video_id = os.path.splitext(str(video_id))[0]

    candidates = [
        os.path.join(graph_dir, task, video_id),
        os.path.join(graph_dir, video_id),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


VGENT_MODEL_ID = {
    "lmms-lab/LLaVA-Video-7B-Qwen2": "llava_video",
    "Qwen/Qwen3.5-9B": "qwen35_9b",
    "Qwen/Qwen3.5-4B": "qwen35_4b",
    "Qwen/Qwen3.5-2B": "qwen35_2b",
    "Qwen/Qwen3-VL-8B-Instruct": "qwenvl3_8b",
    "Qwen/Qwen3-VL-4B-Instruct": "qwenvl3_4b",
    "Qwen/Qwen3-VL-2B-Instruct": "qwenvl3_2b",
    "Qwen/Qwen2.5-VL-7B-Instruct": "qwenvl25_7b",
    "Qwen/Qwen2.5-VL-3B-Instruct": "qwenvl25_3b",
    "Qwen/Qwen2-VL-7B-Instruct": "qwenvl2_7b",
    "Qwen/Qwen2-VL-2B-Instruct": "qwenvl2_2b",
    "OpenGVLab/InternVL2_5-2B": "internvl25_2b",
    "Vision-CAIR/LongVU_Qwen2_7B": "longvu",
}


# ---------------------------------------------------------------------------
# VGent model class
# ---------------------------------------------------------------------------

@register_model("vgent")
class VGentModel(AsyncOpenAIChat):
    """
    lmms-eval model backend that layers VGent graph-RAG on top of a vLLM
    OpenAI-compatible server (``vllm serve``).

    All parameters accepted by the base ``async_openai`` backend are forwarded
    transparently. VGent-specific parameters are consumed here.

    Parameters
    ----------
    vgent_graph_dir : str
        Root directory for VGent knowledge graphs (pre-built or JIT-built).
    vgent_path : str, optional
        Directory prepended to ``sys.path`` so ``vgent_adapter`` is importable.
        Can also be set via the ``VGENT_PATH`` env var.
    vgent_max_clips : int, optional
        Maximum clips retrieved per query. Default: 8.
    vgent_context_prefix : str, optional
        Label prepended to the retrieved context block.
        Default: \"[Retrieved Video Context]\".
    vgent_question_prefix : str, optional
        Label prepended to the original question.
        Default: \"[Question]\".
    vgent_build_on_demand : bool, optional
        When True, automatically build the VGent graph the first time a video
        is encountered instead of raising ``FileNotFoundError``.
        Default: False.
    vgent_model_name : str, optional
        Model identifier passed to VGent's graph builder. Should match the
        ``model`` arg. Default: \"\" (uses ``model``).
    vgent_delete_video_after_build : bool, optional
        When True and ``vgent_build_on_demand=True``, the raw video file is
        deleted from disk after the graph is successfully written.
        Default: False.
    """

    is_simple = False

    def __init__(
        self,
        # VGent-specific args
        vgent_graph_dir: str = "",
        vgent_path: str = "",
        vgent_max_clips: int = 8,
        vgent_context_prefix: str = "[Retrieved Video Context]",
        vgent_question_prefix: str = "[Question]",
        vgent_build_on_demand: bool = False,
        vgent_model_name: str = "",
        vgent_delete_video_after_build: bool = False,
        # All other kwargs forwarded to AsyncOpenAIChat
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vgent_graph_dir = vgent_graph_dir or os.environ.get("VGENT_GRAPH_DIR", "")
        self.vgent_build_on_demand = str(vgent_build_on_demand).lower() not in ("false", "0", "no", "")
        self.vgent_model_name = vgent_model_name or kwargs.get("model", self.model_version)
        self.vgent_model_id = VGENT_MODEL_ID.get(self.vgent_model_name, "lmms_eval_async_openai")
        self.vgent_delete_video_after_build = str(vgent_delete_video_after_build).lower() not in ("false", "0", "no", "")
        self.vgent_max_clips = int(os.environ.get("VGENT_MAX_CLIPS", str(vgent_max_clips)))
        self.vgent_context_prefix = vgent_context_prefix
        self.vgent_question_prefix = vgent_question_prefix

        # Attempt to import VGent now (non-fatal)
        vgent_available = _load_vgent(vgent_path or os.environ.get("VGENT_PATH", ""))

        if not vgent_available:
            raise ImportError(
                "[VGent] Could not import vgent_adapter.run_vgent_query.\n"
                "Make sure VGENT_PATH (or vgent_path=) points to the Vgent repo root\n"
                "and that vgent_adapter.py exists there."
            )

        if not self.vgent_graph_dir:
            raise ValueError(
                "[VGent] vgent_graph_dir is required but was not set.\n"
                "Pass it via --model_args: vgent_graph_dir=/path/to/graphs\n"
                "or export VGENT_GRAPH_DIR=/path/to/graphs before running."
            )

        if not self.vgent_build_on_demand and not os.path.isdir(self.vgent_graph_dir):
            raise FileNotFoundError(
                f"[VGent] vgent_graph_dir '{self.vgent_graph_dir}' does not exist.\n"
                "Create it by running VGent's offline graph construction:\n"
                "  conda activate vgent\n"
                f"  cd Vgent && torchrun --nproc_per_node=1 -m vgent_graph \\\n"
                "      --model_name lmms_eval_async_openai \\\n"
                "      --task <TASK> \\\n"
                "      --data_path <DATA_PATH>\n"
                "Or enable on-demand building with vgent_build_on_demand=True."
            )

        if self.vgent_build_on_demand:
            if _build_graph_for_video is None:
                raise ImportError(
                    "[VGent] vgent_build_on_demand=True requires "
                    "vgent_adapter.build_graph_for_video to be defined, "
                    "but it was not found in vgent_adapter.py.\n"
                    "Add the function or set vgent_build_on_demand=False "
                    "and pre-build graphs with torchrun -m vgent_graph."
                )

            # Pre-initialize VGent on the main thread before any coroutines run.
            # This avoids loguru thread-safety issues and ensures the singleton
            # Vgent instance is created before async workers start.
            if _init_vgent_instance is not None:
                eval_logger.info("[VGent] Pre-initializing VGent instance on main thread...")
                _init_vgent_instance(
                    self.vgent_model_id,
                    "custom",
                    openai_client=self.client,
                    openai_model_version=self.model_version,
                )

            os.makedirs(self.vgent_graph_dir, exist_ok=True)
            eval_logger.info(
                f"[VGent] On-demand mode — graphs will be built under "
                f"'{self.vgent_graph_dir}' as videos are encountered."
                + (" Video files will be deleted after build." if self.vgent_delete_video_after_build else "")
            )
        else:
            eval_logger.info(
                f"[VGent] Pre-built mode — graph_dir='{self.vgent_graph_dir}', "
                f"max_clips={self.vgent_max_clips}"
            )

    # ------------------------------------------------------------------
    # JIT graph builder (synchronous, runs in thread-pool worker)
    # ------------------------------------------------------------------

    def _build_graph_jit(self, task: str, doc: dict) -> str:
        """
        Build the VGent knowledge graph for a single video on-the-fly.
        Uses a per-video lock so that concurrent workers never build the
        same graph simultaneously.
        """
        video_path = (
            doc.get("video")
            or doc.get("video_path")
            or doc.get("videoPath")
            or ""
        )
        if not video_path or not os.path.isfile(str(video_path)):
            raise RuntimeError(
                f"[VGent] Cannot build graph on-demand: video file not found.\n"
                f"doc['video'] = {video_path!r}\n"
                "Ensure the dataset has downloaded the video to a local path "
                "before graph construction."
            )

        video_id = os.path.splitext(os.path.basename(str(video_path)))[0]
        output_dir = os.path.join(
            self.vgent_graph_dir,
            f"{task}_{self.vgent_model_name.replace('/', '_').replace('.', '_')}",
            video_id,
        )

        if os.path.isdir(output_dir):
            return output_dir

        lock = _get_jit_lock(output_dir)
        with lock:
            if os.path.isdir(output_dir):
                return output_dir

            eval_logger.info(
                f"[VGent] Building graph on-demand: "
                f"video='{video_path}', output='{output_dir}'"
            )
            try:
                built_dir = _build_graph_for_video(
                    video_path=str(video_path),
                    output_dir=output_dir,
                    model_name=self.vgent_model_id,
                    task=task,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"[VGent] build_graph_for_video failed for '{video_path}'.\n"
                    f"Underlying error: {exc}"
                ) from exc

            if not os.path.isdir(built_dir or output_dir):
                raise RuntimeError(
                    f"[VGent] build_graph_for_video did not create the expected "
                    f"directory '{output_dir}'."
                )

            if self.vgent_delete_video_after_build:
                try:
                    os.remove(str(video_path))
                    eval_logger.info(f"[VGent] Deleted video after graph build: '{video_path}'")
                except OSError as exc:
                    eval_logger.warning(f"[VGent] Could not delete video '{video_path}': {exc}")

        return output_dir

    # ------------------------------------------------------------------
    # Core augmentation: inject VGent context into messages
    # ------------------------------------------------------------------

    def _augment_messages(
        self,
        raw_messages: list[dict],
        task: str,
        doc: dict,
    ) -> list[dict]:
        """
        Look up the VGent graph for this sample, run retrieval-reasoning,
        and prepend the returned context to the user message text.
        """
        graph_path = _resolve_graph_path(self.vgent_graph_dir, task, doc)
        if graph_path is None:
            if self.vgent_build_on_demand:
                video_path = _extract_video_path(raw_messages)
                if video_path:
                    doc["video_path"] = video_path
                graph_path = self._build_graph_jit(task, doc)
            else:
                video_id = (
                    doc.get("video_id")
                    or doc.get("video")
                    or doc.get("videoID")
                    or doc.get("id")
                    or "<unknown>"
                )
                expected_paths = [
                    os.path.join(self.vgent_graph_dir, task, str(video_id)),
                    os.path.join(self.vgent_graph_dir, str(video_id)),
                ]
                raise FileNotFoundError(
                    f"[VGent] No pre-built graph found for video '{video_id}' "
                    f"(task='{task}').\n"
                    "Searched:\n"
                    + "\n".join(f"  {p}" for p in expected_paths)
                    + "\n\nBuild the graph with:\n"
                    "  conda activate vgent\n"
                    f"  cd Vgent && torchrun --nproc_per_node=1 -m vgent_graph \\\n"
                    f"      --model_name lmms_eval_async_openai \\\n"
                    f"      --task {task} \\\n"
                    "      --data_path <DATA_PATH>\n"
                    f"Then re-run lmms-eval with vgent_graph_dir={self.vgent_graph_dir}\n"
                    "Or enable on-demand building with vgent_build_on_demand=True."
                )

        user_text = _extract_user_text(raw_messages)

        try:
            context = _run_vgent_query(
                graph_dir=graph_path,
                query=user_text,
                max_clips=self.vgent_max_clips,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[VGent] run_vgent_query failed for graph '{graph_path}'.\n"
                f"Underlying error: {exc}"
            ) from exc

        if not context:
            raise RuntimeError(
                f"[VGent] run_vgent_query returned an empty context for graph "
                f"'{graph_path}' (query: {user_text[:120]!r}).\n"
                "Check the graph file for corruption or re-run vgent_graph to rebuild."
            )

        augmented = copy.deepcopy(raw_messages)
        augmented_text = (
            f"{self.vgent_context_prefix}\n{context}\n\n"
            f"{self.vgent_question_prefix}\n{user_text}"
        )

        for msg in reversed(augmented):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = augmented_text
            elif isinstance(content, list):
                replaced = False
                for c in content:
                    if c.get("type") == "text":
                        c["text"] = augmented_text
                        replaced = True
                        break
                if not replaced:
                    msg["content"].insert(0, {"type": "text", "text": augmented_text})
            break

        eval_logger.debug(
            f"[VGent] Augmented request for task='{task}', "
            f"graph='{graph_path}', context_len={len(context)}"
        )
        return augmented

    # ------------------------------------------------------------------
    # Override async forward to inject VGent context
    # ------------------------------------------------------------------

    async def maybe_forward_with_tool(self, request: Instance, idx: int):
        """
        Augment with VGent context then delegate to the parent async implementation.
        Runs inside the asyncio event loop managed by AsyncOpenAIChat.generate_until,
        so all requests are truly concurrent and vLLM's continuous-batching
        scheduler sees them all at the same time.
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
        doc = self.task_dict[task][split][doc_id]
        raw_messages = doc_to_messages(doc)

        # Augmentation is CPU/disk-bound (graph lookup + embedding similarity).
        # Run in a thread-pool so the event loop is not blocked.
        loop = asyncio.get_running_loop()
        augmented_messages = await loop.run_in_executor(
            None, self._augment_messages, raw_messages, task, doc
        )

        # Build a patched request whose doc_to_messages returns the augmented list.
        patched_request = types.SimpleNamespace(
            args=(
                ctx,
                lambda _doc, _msgs=augmented_messages: _msgs,
                gen_kwargs,
                doc_id,
                task,
                split,
            )
        )
        return await super().maybe_forward_with_tool(patched_request, idx)
