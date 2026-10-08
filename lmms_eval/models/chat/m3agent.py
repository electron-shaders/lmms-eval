"""Evaluate the released M3-Agent controller with its multimodal memory graph."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.async_openai import AsyncOpenAIChat
from lmms_eval.models.model_utils.concurrency_control import parse_bool


def _load_adapter(root):
    path = root / "m3agent_adapter.py"
    if not path.is_file():
        raise FileNotFoundError(f"Set m3agent_path to the M3-Agent checkout containing {path.name}: {path}")
    spec = importlib.util.spec_from_file_location("m3agent_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.M3AgentAdapter


def _request_content(messages):
    videos, texts = [], []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content:
            if part.get("type") == "text":
                texts.append(part["text"])
            elif part.get("type") == "video":
                videos.append(part["url"])
    if len(videos) > 1:
        raise ValueError("M3-Agent supports one video per question")
    return videos[0] if videos else "", "\n".join(texts)


@register_model("m3agent")
class M3AgentModel(AsyncOpenAIChat):
    is_simple = False

    def __init__(
        self, m3agent_path="", m3agent_python="", m3agent_graph_dir="",
        m3agent_memory_manifest="", m3agent_build_memory=False,
        m3agent_memorization_model="ByteDance-Seed/M3-Agent-Memorization",
        m3agent_voice_backend="local", m3agent_asr_model="openai/whisper-large-v3-turbo", m3agent_asr_device="cuda:0",
        m3agent_face_device="cuda:0",
        m3agent_embed_base_url="http://127.0.0.1:9001/v1", m3agent_embed_model="BAAI/bge-m3",
        m3agent_embed_api_key="", m3agent_reembed=True,
        m3agent_cuda_devices=None, m3agent_worker_timeout=7200,
        m3agent_total_round=5, m3agent_topk=2, m3agent_threshold=0.5,
        m3agent_segment_seconds=30, m3agent_attn_implementation="flash_attention_2", **kwargs,
    ):
        root = Path(m3agent_path or os.getenv("M3AGENT_PATH") or Path(__file__).resolve().parents[4] / "M3-Agent").expanduser().resolve()
        self.graph_dir = Path(m3agent_graph_dir or os.getenv("M3AGENT_GRAPH_DIR") or root / "data/memory_graphs").expanduser().resolve()
        self.manifest = {}
        self.manifest_root = root
        if m3agent_memory_manifest:
            manifest = Path(m3agent_memory_manifest).expanduser().resolve()
            self.manifest_root = manifest.parent
            with manifest.open() as stream:
                self.manifest = json.load(stream)
            if not isinstance(self.manifest, dict):
                raise ValueError("m3agent_memory_manifest must be a JSON object keyed by video path or ID")
        self.root = root
        self.rounds = int(m3agent_total_round)
        self.topk = int(m3agent_topk)
        self.threshold = float(m3agent_threshold)
        if min(self.rounds, self.topk, int(m3agent_segment_seconds)) < 1:
            raise ValueError("M3-Agent rounds, topk and segment_seconds must be positive")
        reembed = parse_bool(m3agent_reembed)
        if reembed and not m3agent_embed_base_url:
            raise ValueError("m3agent_reembed=True requires m3agent_embed_base_url")
        adapter_class = _load_adapter(root)
        kwargs.setdefault("model", "ByteDance-Seed/M3-Agent-Control")
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("base_url", "http://127.0.0.1:9000/v1")
        kwargs.setdefault("api_key", "EMPTY")
        kwargs.setdefault("fail_on_request_error", False)
        super().__init__(**kwargs)
        self.num_cpus = self.batch_size_per_gpu
        self.adaptive_concurrency = False
        self.prefix_aware_queue = False
        self.adapter = adapter_class(
            m3agent_path=str(root), python=m3agent_python or None,
            cuda_devices=m3agent_cuda_devices or os.getenv("M3AGENT_CUDA_DEVICES"), worker_timeout=m3agent_worker_timeout,
            graph_dir=str(self.graph_dir), build_memory=parse_bool(m3agent_build_memory),
            memorization_model=m3agent_memorization_model,
            voice_backend=m3agent_voice_backend, asr_model=m3agent_asr_model, asr_device=m3agent_asr_device,
            face_device=m3agent_face_device,
            embed_base_url=m3agent_embed_base_url, embed_model=m3agent_embed_model,
            embed_api_key=m3agent_embed_api_key, reembed=reembed,
            segment_seconds=int(m3agent_segment_seconds), attn_implementation=m3agent_attn_implementation,
        )

    def _memory_inputs(self, doc, video_path):
        keys = [video_path, doc.get("videoID"), doc.get("video_id"), doc.get("video"), doc.get("id")]
        if video_path:
            keys.append(Path(video_path).stem)
        entry = next((self.manifest[str(key)] for key in keys if key is not None and str(key) in self.manifest), {})
        if isinstance(entry, str):
            entry = {"mem_path": entry}
        if not isinstance(entry, dict):
            raise ValueError("Memory manifest values must be graph paths or objects containing mem_path")
        result = {}
        for field in ("mem_path", "clip_path", "intermediate_path"):
            # The original M3 memorization CLI calls this intermediate_outputs.
            doc_value = doc.get(field) or (doc.get("intermediate_outputs") if field == "intermediate_path" else None)
            value = doc_value or entry.get(field) or (entry.get("intermediate_outputs") if field == "intermediate_path" else None)
            if value:
                path = Path(value).expanduser()
                base = self.root if doc_value else self.manifest_root
                result[field] = str((base / path).resolve())
        return result

    async def maybe_forward_with_tool(self, request, idx):
        _, doc_to_messages, generation_kwargs, doc_id, task, split = request.args
        doc = self.task_dict[task][split][doc_id]
        video_path, question = _request_content(doc_to_messages(doc))
        if video_path:
            video_path = str(Path(video_path).expanduser().resolve())
        memory = self._memory_inputs(doc, video_path)
        if not video_path and not memory.get("mem_path"):
            raise ValueError("M3-Agent needs a video message or a memory graph path")
        # Use the task's complete prompt, including choices, subtitles and answer
        # formatting. Reading doc['question'] alone loses that task information.
        try:
            answer = await self.adapter.query(
                self.client, self.model_version, question, video_path=video_path,
                before_clip=doc.get("before_clip"), generation_kwargs=dict(generation_kwargs or {}),
                total_round=self.rounds, topk=self.topk, threshold=self.threshold, **memory,
            )
        except self.adapter.preparation_error as exc:
            # An unrecoverable graph failure affects this source only. The
            # adapter keeps the worker available for other videos; do not retry
            # preparation or cancel their pending questions.
            eval_logger.warning(f"M3-Agent request {idx} returned an empty answer after graph preparation failed: {exc}")
            answer = ""
        return answer, idx, None

    def clean(self):
        if hasattr(self, "adapter"):
            self.adapter.close()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.client.close())
        else:
            loop.create_task(self.client.close())
        super().clean()
