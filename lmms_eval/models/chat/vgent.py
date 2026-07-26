"""
Vgent model backend for lmms-eval.

Wraps the ``async_openai`` chat backend (``lmms_eval/models/chat/async_openai.py``)
and augments every request with Vgent's graph-based retrieval-reasoning context
before forwarding to the vLLM OpenAI-compatible server.

Graphs are built automatically the first time each video is encountered.

The workflow per sample is::

    lmms-eval downloads video  →  Vgent builds graph  →  query graph
                                (optionally delete video)

Vgent graph directory layout
-----------------------------
    <vgent_graph_dir>/
        <task_name>/
            <video_id>/
                graph.pkl

Environment variables
----------------------
    VGENT_PATH            sys.path prefix for ``vgent_adapter`` import.
    VGENT_GRAPH_DIR       Overrides ``vgent_graph_dir`` model arg.
"""

from __future__ import annotations

import asyncio
import os
import sys

from lmms_eval.api.instance import Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.async_openai import AsyncOpenAIChat
from loguru import logger as eval_logger

# ---------------------------------------------------------------------------
# Lazy Vgent import helper
# ---------------------------------------------------------------------------

_vgent_loaded: bool = False
_run_vgent_query = None        # callable(video_id, query, video_path, output_dir, question, candidates, doc, subtitle_path, model_name, task) -> str
_init_vgent_instance = None    # callable(model_name, task, openai_client, model_version) -> None


def _load_vgent(vgent_path: str | None) -> bool:
    """
    Try to import ``vgent_adapter.run_vgent_query`` and ``vgent_adapter.init_vgent_instance``.

    Returns True if the query function was imported successfully.
    """
    global _vgent_loaded, _run_vgent_query, _init_vgent_instance

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
        _init_vgent_instance = _va.init_vgent_instance
        _vgent_loaded = True
        eval_logger.info("[Vgent] Imported run_vgent_query")
        return True
    except ImportError as exc:
        _vgent_loaded = True
        eval_logger.warning(
            f"[Vgent] Could not import vgent_adapter ({exc}). "
            "Set VGENT_PATH or pass vgent_path=<dir> in --model_args."
        )
        return False


# ---------------------------------------------------------------------------
# Helper: extract fields from raw messages or doc
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


def _extract_question(doc: dict) -> str | None:
    """Return the question text from the document, if present."""
    return doc.get("question") or doc.get("query") or doc.get("prompt")


def _extract_candidates(doc: dict) -> list[str] | None:
    """Return the candidate options from the document, if present."""
    candidates = doc.get("options") or doc.get("candidates") or doc.get("choices")
    if candidates is None:
        return None
    return list(candidates)


# ---------------------------------------------------------------------------
# Helper: resolve video ID for a document
# ---------------------------------------------------------------------------

def _resolve_video_id(
    doc: dict,
) -> str | None:
    video_id = (
        doc.get("videoID")
        or doc.get("video_id")
        or doc.get("video")
        or doc.get("id")
        or ""
    )
    if not video_id:
        return None

    video_id = os.path.splitext(str(video_id))[0]

    return video_id


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
# Vgent model class
# ---------------------------------------------------------------------------

@register_model("vgent")
class VgentModel(AsyncOpenAIChat):
    """
    lmms-eval model backend that layers Vgent graph-RAG on top of a vLLM
    OpenAI-compatible server (``vllm serve``).

    All parameters accepted by the base ``async_openai`` backend are forwarded
    transparently. Vgent-specific parameters are consumed here.

    Parameters
    ----------
    vgent_graph_dir : str
        Root directory for Vgent knowledge graphs (pre-built or JIT-built).
    vgent_path : str, optional
        Directory prepended to ``sys.path`` so ``vgent_adapter`` is importable.
        Can also be set via the ``VGENT_PATH`` env var.
    vgent_model_name : str, optional
        Model identifier passed to Vgent's graph builder. Should match the
        ``model`` arg. Default: \"\" (uses ``model``).
    """

    is_simple = False

    def __init__(
        self,
        # Vgent-specific args
        vgent_graph_dir: str = "",
        vgent_path: str = "",
        vgent_model_name: str = "",
        # All other kwargs forwarded to AsyncOpenAIChat
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vgent_graph_dir = vgent_graph_dir or os.environ.get("VGENT_GRAPH_DIR", "")
        self.vgent_model_name = vgent_model_name or kwargs.get("model", self.model_version)
        self.vgent_model_id = VGENT_MODEL_ID.get(self.vgent_model_name, "lmms_eval_async_openai")

        # Attempt to import Vgent now (non-fatal)
        vgent_available = _load_vgent(vgent_path or os.environ.get("VGENT_PATH", ""))

        if not vgent_available:
            raise ImportError(
                "[Vgent] Could not import vgent_adapter.run_vgent_query.\n"
                "Make sure VGENT_PATH (or vgent_path=) points to the Vgent repo root\n"
                "and that vgent_adapter.py exists there."
            )

        if not self.vgent_graph_dir:
            raise ValueError(
                "[Vgent] vgent_graph_dir is required but was not set.\n"
                "Pass it via --model_args: vgent_graph_dir=/path/to/graphs\n"
                "or export VGENT_GRAPH_DIR=/path/to/graphs before running."
            )

        # Pre-initialize Vgent on the main thread before any coroutines run.
        # This avoids loguru thread-safety issues and ensures the singleton
        # Vgent instance is created before async workers start.
        if _init_vgent_instance is not None:
            eval_logger.info("[Vgent] Pre-initializing Vgent instance on main thread...")
            _init_vgent_instance(
                self.vgent_model_id,
                "custom",
                openai_client=self.client,
                openai_model_version=self.model_version,
            )

        os.makedirs(self.vgent_graph_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Core Vgent query method
    # ------------------------------------------------------------------

    def _run_vgent_query(
        self,
        raw_messages: list[dict],
        task: str,
        doc: dict,
    ) -> list[dict]:
        """
        Follow Vgent workflow as implemented in vgent_rag.py
        """
        user_text = _extract_user_text(raw_messages)
        question = _extract_question(doc) or user_text
        candidates = _extract_candidates(doc) or []
        doc["letters"] = [chr(ord("A") + i) for i in range(len(candidates))]
        video_id = _resolve_video_id(doc)
        video_path = _extract_video_path(raw_messages)
        if video_path:
            doc["video_path"] = video_path
            subtitle_path = os.path.join(os.path.dirname(os.path.dirname(video_path)), "subtitle", f"{video_id}.srt")
        else:
            subtitle_path = None

        output_dir = os.path.join(
            self.vgent_graph_dir,
            f"{task}_{self.vgent_model_name.replace('/', '_').replace('.', '_')}",
            video_id,
        )

        try:
            resp = _run_vgent_query(
                video_id=video_id,
                query=user_text,
                video_path=video_path,
                output_dir=output_dir,
                question=question,
                candidates=candidates,
                doc=doc,
                subtitle_path=subtitle_path,
                model_name=self.vgent_model_id,
                task=task,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[Vgent] run_vgent_query failed for video '{video_id}'.\n"
                f"Underlying error: {exc}"
            ) from exc

        return resp

    # ------------------------------------------------------------------
    # Override async forward to inject Vgent context
    # ------------------------------------------------------------------

    async def maybe_forward_with_tool(self, request: Instance, idx: int):
        """
        Augment with Vgent context then delegate to the parent async implementation.
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
        response = await loop.run_in_executor(
            None, self._run_vgent_query, raw_messages, task, doc
        )

        return response, idx, TokenCounts()
