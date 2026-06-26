"""
Deep Video Discovery (DVD) model backend for lmms-eval.

Wraps the ``async_openai`` chat backend and augments every request with DVD's
agentic deep-search answer before forwarding to the vLLM server.

DVD operates as a ReAct agent that:
  1. Searches a per-video vector database (clip captions + embeddings)
  2. Optionally inspects raw video frames via frame_inspect_tool
  3. Calls ``finish(answer)`` when it has enough evidence

Environment variables
---------------------
    DVD_PATH          sys.path prefix for ``dvd_adapter`` import.
    DVD_DB_DIR        Overrides ``dvd_db_dir`` model arg.
    DVD_MAX_ITER      Overrides ``dvd_max_iterations`` model arg.
    DVD_LITE_MODE     Overrides ``dvd_lite_mode`` model arg.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import types
from typing import Optional

from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.async_openai import AsyncOpenAIChat


# ---------------------------------------------------------------------------
# Lazy DVD import helper
# ---------------------------------------------------------------------------

_dvd_loaded: bool = False
_run_dvd_query = None
_init_dvd_instance = None

def _load_dvd(dvd_path: Optional[str]) -> bool:
    global _dvd_loaded, _run_dvd_query, _init_dvd_instance

    if _dvd_loaded:
        return _run_dvd_query is not None

    search_paths = []
    if dvd_path:
        search_paths.append(dvd_path)
    env_path = os.environ.get("DVD_PATH", "")
    if env_path:
        search_paths.append(env_path)

    for p in search_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import dvd_adapter as _da  # type: ignore[import]

        _run_dvd_query = _da.run_dvd_query
        _init_dvd_instance = _da.init_dvd_instance
        _dvd_loaded = True
        eval_logger.info("[DVD] Imported dvd_adapter successfully.")
        return True
    except ImportError as exc:
        _dvd_loaded = True
        eval_logger.warning(
            f"[DVD] Could not import dvd_adapter ({exc}). "
            "Set DVD_PATH or pass dvd_path=<dir> in --model_args."
        )
        return False


# ---------------------------------------------------------------------------
# Helpers for extracting video path and text from raw messages
# ---------------------------------------------------------------------------

def _extract_video_path(raw_messages: list[dict]) -> str:
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
# DVD model backend
# ---------------------------------------------------------------------------

@register_model("dvd")
class DVDModel(AsyncOpenAIChat):
    """
    lmms-eval model backend that answers long-video questions using the
    Deep Video Discovery (DVD) agentic framework, served via a vLLM
    OpenAI-compatible API.

    All ``async_openai`` parameters are forwarded to the base class.

    DVD-specific parameters
    -----------------------
    dvd_path : str
        Directory containing mcp_server.py (the deepvideodiscovery repo root).
        Also settable via the ``DVD_PATH`` env var.
    dvd_venv_python : str
        Path to the Python interpreter inside the DVD virtual environment, e.g.
        ``./deepvideodiscovery/.venv/bin/python``.  The MCP server subprocess
        is launched with this interpreter so that DVD's deps (numpy<2, etc.) do
        not conflict with the lmms-eval/vLLM environment.
        Also settable via the ``DVD_VENV_PYTHON`` env var.
    dvd_db_dir : str
        Root directory where DVD per-video databases are stored/cached.
        Settable via ``DVD_DB_DIR``.
    dvd_max_iterations : int
        Maximum DVD agent reasoning iterations per question.  Default: 15.
    dvd_lite_mode : bool
        Use subtitle-only (no frame pixels) mode.  Default: True.
    dvd_embed_model : str
        Embedding model name served by the embedding vLLM instance.  Default: "BAAI/bge-m3".
    dvd_embed_dim : int
        Embedding vector dimension.  Default: 1024.
    dvd_embed_base_url : str
        Embedding server base URL (separate ``vllm serve --task embed`` process).
    dvd_clip_secs : int
        Clip duration in seconds for video segmentation.  Default: 10.
    dvd_video_fps : float
        Target FPS for frame extraction (non-lite mode).  Default: 2.
    dvd_global_browse_topk : int
        Clips retrieved by global_browse_tool.  Default: 300.
    """

    is_simple = False

    def __init__(
        self,
        dvd_path: str = "",
        dvd_venv_python: str = "",
        dvd_db_dir: str = "",
        dvd_max_iterations: int = 15,
        dvd_lite_mode: bool = True,
        dvd_embed_model: str = "BAAI/bge-m3",
        dvd_embed_dim: int = 1024,
        dvd_embed_base_url: str = "",
        dvd_clip_secs: int = 10,
        dvd_video_fps: float = 2.0,
        dvd_global_browse_topk: int = 300,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dvd_db_dir = dvd_db_dir or os.environ.get("DVD_DB_DIR", "./.dvd_dbs")
        self.dvd_max_iterations = int(
            os.environ.get("DVD_MAX_ITER", str(dvd_max_iterations))
        )
        self.dvd_lite_mode = str(
            os.environ.get("DVD_LITE_MODE", str(dvd_lite_mode))
        ).lower() not in ("false", "0", "no")
        self.dvd_embed_model = dvd_embed_model
        self.dvd_embed_dim = dvd_embed_dim
        self.dvd_embed_base_url = dvd_embed_base_url or os.environ.get("DVD_EMBED_BASE_URL", "")
        self.dvd_clip_secs = dvd_clip_secs
        self.dvd_video_fps = dvd_video_fps
        self.dvd_global_browse_topk = dvd_global_browse_topk

        # Resolve paths
        resolved_dvd_path = dvd_path or os.environ.get("DVD_PATH", "")
        resolved_venv_python = (
            dvd_venv_python
            or os.environ.get("DVD_VENV_PYTHON", "")
            or os.path.join(resolved_dvd_path, ".venv", "bin", "python")
        )

        # Import adapter
        if not _load_dvd(resolved_dvd_path):
            raise ImportError(
                "[DVD] Could not import dvd_adapter.\n"
                "Set DVD_PATH or pass dvd_path=<deepvideodiscovery-repo-root> "
                "in --model_args."
            )

        # Initialize DVD on the main thread so all module-level patches are
        # applied before any async workers start calling DVD code.
        eval_logger.info("[DVD] Initializing DVD adapter on main thread...")
        _init_dvd_instance(
            dvd_path=resolved_dvd_path,
            base_url=str(self.client.base_url),
            api_key=self.client.api_key,
            vlm_model=self.model_version,
            embed_model=dvd_embed_model,
            embed_dim=dvd_embed_dim,
            embed_base_url=self.dvd_embed_base_url or None,
            lite_mode=self.dvd_lite_mode,
            max_iterations=self.dvd_max_iterations,
            clip_secs=dvd_clip_secs,
            video_fps=dvd_video_fps,
            global_browse_topk=dvd_global_browse_topk,
            dvd_db_dir=self.dvd_db_dir,
            dvd_venv_python=resolved_venv_python,
        )

        os.makedirs(self.dvd_db_dir, exist_ok=True)
        eval_logger.info(
            f"[DVD] Ready. db_dir='{self.dvd_db_dir}', "
            f"lite_mode={self.dvd_lite_mode}, "
            f"max_iterations={self.dvd_max_iterations}"
        )

    # ------------------------------------------------------------------
    # Override async forward: run DVD agent, then forward final answer
    # ------------------------------------------------------------------

    async def maybe_forward_with_tool(self, request: Instance, idx: int):
        """
        1. Extract the video path and question from the request.
        2. Run the DVD agent (in a thread-pool executor so the event loop is free).
        3. Replace the question in the original messages with the DVD answer.
        4. Forward to the parent AsyncOpenAIChat for the final structured answer.
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
        doc = self.task_dict[task][split][doc_id]
        raw_messages = doc_to_messages(doc)

        # Extract video path (from doc fields or message content)
        video_path = (
            str(doc.get("video") or doc.get("video_path") or doc.get("videoPath") or "")
            or _extract_video_path(raw_messages)
        )
        question = _extract_user_text(raw_messages)

        # Run DVD agent
        dvd_answer = await self._run_dvd_for_sample(
            video_path,
            question,
            task,
        )

        eval_logger.debug(
            f"[DVD] task='{task}' doc_id={doc_id} "
            f"dvd_answer={dvd_answer[:120]!r}"
        )

        # Build a text-only message list containing just the DVD answer so
        # the parent class can apply its standard response_format / sampling.
        augmented_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Based on the following video analysis:\n\n{dvd_answer}\n\n"
                            f"Answer the question concisely: {question}"
                        ),
                    }
                ],
            }
        ]

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

    async def _run_dvd_for_sample(
        self,
        video_path: str,
        question: str,
        task: str,
    ) -> str:
        """
        Async helper: build DVD database if needed, then run the agent.
        """
        if not video_path or not os.path.isfile(video_path):
            eval_logger.warning(
                f"[DVD] Video file not found: {video_path!r}. "
                "Returning empty DVD context."
            )
            return ""

        # Locate SRT subtitle alongside the video file if it exists
        srt_path = os.path.splitext(video_path)[0] + ".srt"
        if not os.path.isfile(srt_path):
            srt_path = None

        try:
            answer = await _run_dvd_query(
                video_path=video_path,
                question=question,
                dvd_db_dir=self.dvd_db_dir,
                max_iterations=self.dvd_max_iterations,
                lite_mode=self.dvd_lite_mode,
                srt_path=srt_path,
            )
        except Exception as exc:
            eval_logger.error(f"[DVD] run_dvd_query failed: {exc}")
            answer = ""

        return answer or ""
