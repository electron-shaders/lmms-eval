"""
VideoRAG model backend for lmms-eval.

Wraps the ``async_openai`` chat backend and augments every request with
VideoRAG's graph-based retrieval-augmented generation context before
forwarding to the vLLM OpenAI-compatible server.

VideoRAG operates by:
  1. Splitting a video into segments and extracting transcripts + captions
  2. Building a knowledge graph over the video segments
  3. Retrieving relevant segments via entity/visual retrieval
  4. Generating a final answer using the retrieved context

Environment variables
---------------------
    VIDEORAG_PATH         sys.path prefix for ``videorag_adapter`` import.
    VIDEORAG_WORKING_DIR  Overrides ``videorag_working_dir`` model arg.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.async_openai import AsyncOpenAIChat
from loguru import logger as eval_logger

# ---------------------------------------------------------------------------
# Lazy VideoRAG import helper
# ---------------------------------------------------------------------------

_videorag_loaded: bool = False
_run_videorag_query = None
_init_videorag_instance = None


def _load_videorag(videorag_path: str | None) -> bool:
    """
    Try to import ``videorag_adapter.run_videorag_query`` and
    ``videorag_adapter.init_videorag_instance``.

    Returns True if the query function was imported successfully.
    """
    global _videorag_loaded, _run_videorag_query, _init_videorag_instance

    if _videorag_loaded:
        return _run_videorag_query is not None

    search_paths = []
    if videorag_path:
        search_paths.append(videorag_path)
    env_path = os.environ.get("VIDEORAG_PATH", "")
    if env_path:
        search_paths.append(env_path)

    for p in search_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import videorag_adapter as _va  # type: ignore[import]

        _run_videorag_query = _va.run_videorag_query
        _init_videorag_instance = _va.init_videorag_instance
        _videorag_loaded = True
        eval_logger.info("[VideoRAG] Imported videorag_adapter successfully.")
        return True
    except ImportError as exc:
        _videorag_loaded = True
        eval_logger.warning(
            f"[VideoRAG] Could not import videorag_adapter ({exc}). "
            "Set VIDEORAG_PATH or pass videorag_path=<dir> in --model_args."
        )
        return False


# ---------------------------------------------------------------------------
# Helpers for extracting video path and text from raw messages
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
# VideoRAG model backend
# ---------------------------------------------------------------------------

@register_model("videorag")
class VideoRAGModel(AsyncOpenAIChat):
    """
    lmms-eval model backend that answers long-video questions using the
    VideoRAG graph-based retrieval-augmented generation framework, served
    via a vLLM OpenAI-compatible API.

    All ``async_openai`` parameters are forwarded to the base class.

    VideoRAG-specific parameters
    ----------------------------
    videorag_path : str
        Directory containing ``videorag_adapter.py`` (the VideoRAG-algorithm
        repo root).
    videorag_working_dir : str
        Root directory where VideoRAG stores per-video caches (graphs,
        segments, embeddings).
    videorag_embed_model : str
        Embedding model name served by the embedding vLLM instance.
        Default: ``BAAI/bge-m3``.
    videorag_embed_dim : int
        Embedding vector dimension.  Default: 1024.
    videorag_embed_base_url : str
        Embedding server base URL.
    videorag_segment_length : int
        Video segment length in seconds.  Default: 30.
    videorag_retrieval_topk : int
        Number of segments retrieved per query.  Default: 4.
    videorag_query_mode : str
        Query mode: ``videorag`` (open-ended) or
        ``videorag_multiple_choice`` (MC).  Default: ``videorag_multiple_choice``.
    videorag_load_caption_model : bool
        Whether to load the real MiniCPM caption model.  Default: False
        (uses debug/no-op captions for faster benchmarking).
    """

    is_simple = False

    def __init__(
        self,
        videorag_path: str = "",
        videorag_venv_python: str = "",
        videorag_working_dir: str = "",
        videorag_embed_model: str = "BAAI/bge-m3",
        videorag_embed_dim: int = 1024,
        videorag_embed_base_url: str = "",
        videorag_segment_length: int = 30,
        videorag_retrieval_topk: int = 4,
        videorag_query_mode: str = "videorag_multiple_choice",
        videorag_load_caption_model: bool = False,
        videorag_host: str = "127.0.0.1",
        videorag_port: int = 9003,
        videorag_cuda_device: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.videorag_working_dir = (
            videorag_working_dir
            or os.environ.get("VIDEORAG_WORKING_DIR", "")
            or "./.videorag_cache"
        )
        self.videorag_query_mode = videorag_query_mode
        self.videorag_embed_model = videorag_embed_model
        self.videorag_embed_dim = int(videorag_embed_dim)
        self.videorag_embed_base_url = (
            videorag_embed_base_url
            or os.environ.get("VIDEORAG_EMBED_BASE_URL", "")
        )
        self.videorag_segment_length = int(videorag_segment_length)
        self.videorag_retrieval_topk = int(videorag_retrieval_topk)
        self.videorag_load_caption_model = videorag_load_caption_model
        self.videorag_host = videorag_host
        self.videorag_port = int(videorag_port)
        self.videorag_cuda_device = (
            videorag_cuda_device
            or os.environ.get("VIDEORAG_CUDA_DEVICE", "")
            or os.environ.get("VIDEORAG_GPUS", "")
        )

        # Resolve paths
        resolved_videorag_path = (
            videorag_path or os.environ.get("VIDEORAG_PATH", "")
        )
        resolved_venv_python = (
            videorag_venv_python
            or os.environ.get("VIDEORAG_VENV_PYTHON", "")
            or os.path.expanduser("~/miniconda/envs/videorag/bin/python")
        )

        # Import adapter
        if not _load_videorag(resolved_videorag_path):
            raise ImportError(
                "[VideoRAG] Could not import videorag_adapter.\n"
                "Set VIDEORAG_PATH or pass videorag_path=<VideoRAG-algorithm-repo-root> "
                "in --model_args."
            )

        # Initialize VideoRAG on the main thread
        eval_logger.info("[VideoRAG] Initializing VideoRAG adapter on main thread...")
        _init_videorag_instance(
            videorag_path=resolved_videorag_path,
            base_url=str(self.client.base_url),
            api_key=self.client.api_key,
            vlm_model=self.model_version,
            embed_model=videorag_embed_model,
            embed_dim=self.videorag_embed_dim,
            embed_base_url=self.videorag_embed_base_url or None,
            working_dir=self.videorag_working_dir,
            video_segment_length=self.videorag_segment_length,
            segment_retrieval_top_k=self.videorag_retrieval_topk,
            videorag_query_mode=self.videorag_query_mode,
            load_caption_model=self.videorag_load_caption_model,
            videorag_venv_python=resolved_venv_python,
            videorag_host=self.videorag_host,
            videorag_port=self.videorag_port,
            videorag_cuda_device=self.videorag_cuda_device,
        )

        os.makedirs(self.videorag_working_dir, exist_ok=True)
        eval_logger.info(
            f"[VideoRAG] Ready. working_dir='{self.videorag_working_dir}', "
            f"query_mode={self.videorag_query_mode}, "
            f"segment_length={self.videorag_segment_length}s"
        )

    # ------------------------------------------------------------------
    # Override async forward: run VideoRAG pipeline, then return answer
    # ------------------------------------------------------------------

    async def maybe_forward_with_tool(self, request: Instance, idx: int):
        """
        1. Extract the video path and question from the request.
        2. Run VideoRAG's insert + query pipeline.
        3. Return the answer directly — VideoRAG already produces the
           final answer, so no second VLM round-trip is needed.

        Returns
        -------
        (content, idx, token counts) matching the contract expected by the
        parent's generate_until loop.
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

        # Run VideoRAG pipeline asynchronously
        videorag_answer = await _run_videorag_query(
            video_path=video_path,
            question=question,
            mode=self.videorag_query_mode,
        )

        eval_logger.debug(
            f"[VideoRAG] task='{task}' doc_id={doc_id} "
            f"answer={str(videorag_answer)[:120]!r}"
        )

        # VideoRAG does not expose per-query usage across its internal calls.
        # Aggregate service usage is captured from the vLLM endpoints.
        return videorag_answer, idx, None
