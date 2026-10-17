"""Frozen EmbeddingGemma 2 retrieval followed by one Qwen video answer."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from lmms_eval.api.exceptions import FatalEvaluationError
from lmms_eval.api.instance import TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.async_openai import AsyncOpenAIChat
from lmms_eval.models.model_utils.usage_metrics import log_usage


@register_model("vanilla_rag")
class VanillaRAGModel(AsyncOpenAIChat):
    is_simple = False

    def __init__(
        self,
        rag_embed_model="google/embeddinggemma-2",
        rag_embed_revision=None,
        rag_embed_device="cuda:0",
        rag_cache_dir=".vanilla_rag_cache",
        rag_chunk_seconds=10,
        rag_top_k=10,
        rag_embed_fps=1,
        rag_embed_max_frames=32,
        rag_embed_batch_size=4,
        rag_clip_fps=1,
        rag_clip_longest_edge=448,
        rag_query_field="question",
        rag_selection="dense",
        rag_max_new_tokens=16384,
        rag_seed=42,
        rag_trace_path=None,
        rag_vlm_revision=None,
        **kwargs,
    ):
        from .core import VideoRetriever, positive
        from .encoder import (
            EmbeddingGemmaEncoder,
            check_dependencies,
        )
        from .messages import generation_payload

        if rag_selection not in ("dense", "uniform"):
            raise ValueError("rag_selection must be dense or uniform")
        if kwargs.get("mcp_server_path"):
            raise ValueError("vanilla_rag uses one generation call and does not accept an MCP server")
        self.selection, self.query_field = rag_selection, rag_query_field
        self.top_k = positive(rag_top_k, "rag_top_k", integer=True)
        self.clip_fps = positive(rag_clip_fps, "rag_clip_fps")
        self.clip_longest_edge = positive(rag_clip_longest_edge, "rag_clip_longest_edge", integer=True)
        self.max_new_tokens = positive(rag_max_new_tokens, "rag_max_new_tokens", integer=True)
        self.seed = int(rag_seed)
        self.vlm_revision = rag_vlm_revision
        self.trace_path = Path(rag_trace_path).expanduser() if rag_trace_path else None
        self.encoder = None
        if rag_selection == "dense":
            check_dependencies()
            self.encoder = EmbeddingGemmaEncoder(rag_embed_model, rag_embed_revision, rag_embed_device)
        self.retriever = VideoRetriever(rag_cache_dir, encoder=self.encoder, chunk_seconds=rag_chunk_seconds, embed_fps=rag_embed_fps, embed_max_frames=rag_embed_max_frames, embed_batch_size=rag_embed_batch_size)
        kwargs.setdefault("model", "Qwen/Qwen3.5-9B")
        kwargs.setdefault("enable_thinking", True)
        kwargs["max_retries"] = positive(kwargs.get("max_retries", 3), "max_retries", integer=True)
        kwargs["fail_on_request_error"] = True
        kwargs["adaptive_concurrency"] = False
        kwargs["prefix_aware_queue"] = False
        # Validate the preset before constructing an API client.
        generation_payload(kwargs["model"], enable_thinking=kwargs["enable_thinking"], max_tokens=self.max_new_tokens, seed=self.seed)
        super().__init__(**kwargs)
        if self.world_size != 1:
            raise ValueError("Use one evaluator process; distribute the answering VLM with tensor parallelism")
        self.num_cpus = self.batch_size_per_gpu
        self._prepared = {}
        self._workloads = {}

    def loglikelihood(self, requests):
        raise NotImplementedError("vanilla_rag supports single-turn generate_until video tasks")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("vanilla_rag supports single-turn generate_until video tasks")

    def _prepare(self, raw_messages, doc):
        from .messages import (
            evidence_messages,
            task_inputs,
        )

        source, question, messages = task_inputs(raw_messages, doc, self.query_field)
        selected, trace = self.retriever.retrieve(source, question, self.top_k, self.selection)
        started = time.monotonic()
        messages, sampled = evidence_messages(messages, source, selected, fps=self.clip_fps, longest_edge=self.clip_longest_edge)
        trace.update(selected=sampled, selected_frames=sum(item["frames"] for item in sampled), evidence_s=time.monotonic() - started, query=question)
        trace["configuration"] = {
            "chunk_seconds": self.retriever.chunk_seconds,
            "top_k": self.top_k,
            "embed_fps": self.retriever.embed_fps,
            "embed_max_frames": self.retriever.embed_max_frames,
            "embed_batch_size": self.retriever.embed_batch_size,
            "clip_fps": self.clip_fps,
            "clip_longest_edge": self.clip_longest_edge,
            "query_field": self.query_field,
        }
        return messages, trace

    def _write_trace(self, trace):
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            # The single evaluator event loop owns trace writes.
            with self.trace_path.open("a") as stream:
                stream.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")

    async def maybe_forward_with_tool(self, request, idx):
        from .messages import (
            apply_answer_stops,
            generation_payload,
            task_stops,
        )

        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
        if idx not in self._prepared:
            try:
                doc = self.task_dict[task][split][doc_id]
                raw = doc_to_messages(doc)
                self._prepared[idx] = await asyncio.to_thread(self._prepare, raw, doc)
            except Exception as exc:
                self._write_trace({"task": task, "split": split, "doc_id": doc_id, "status": "preparation_failed", "error": str(exc)})
                raise FatalEvaluationError(f"vanilla_rag preparation failed for {task}/{doc_id}: {exc}") from exc
        messages, trace = self._prepared[idx]
        trace.update(task=task, split=split, doc_id=doc_id, model=self.model_version, model_revision=self.vlm_revision)
        payload = generation_payload(self.model_version, gen_kwargs, enable_thinking=self.enable_thinking, max_tokens=self.max_new_tokens, seed=self.seed)
        trace["generation"] = payload
        trace["task_stop"] = task_stops(gen_kwargs)
        trace["answer_stop"] = task_stops(gen_kwargs, enable_thinking=self.enable_thinking)
        trace["attempts"] = trace.get("attempts", 0) + 1
        if self.system_prompt:
            messages = self._apply_system_prompt(messages, self.system_prompt)
        started = time.monotonic()
        try:
            response = await self.client.chat.completions.create(messages=messages, **payload)
        except Exception as exc:
            self._write_trace({**trace, "status": "request_failed", "error": str(exc)})
            raise
        trace["generation_s"] = time.monotonic() - started
        choice = response.choices[0]
        raw_answer = choice.message.content or ""
        text = apply_answer_stops(raw_answer, gen_kwargs, enable_thinking=self.enable_thinking)
        usage = getattr(response, "usage", None)
        counts = TokenCounts(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            reasoning_tokens=getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
        )
        trace.update(status="completed", finish_reason=choice.finish_reason, raw_answer=raw_answer, answer=text, tokens=counts.to_dict())
        self._workloads[idx] = trace
        self._write_trace(trace)
        # The usage ledger sums output + reasoning, whereas the OpenAI API's
        # completion_tokens (and GenerationResult) already include reasoning.
        log_usage(
            model_name=self.model_version,
            task_name=task,
            input_tokens=counts.input_tokens or 0,
            output_tokens=max(0, (counts.output_tokens or 0) - (counts.reasoning_tokens or 0)),
            reasoning_tokens=counts.reasoning_tokens or 0,
            source="model",
        )
        self._prepared.pop(idx, None)
        return text, idx, counts

    def _create_client(self):
        # The parent loop owns transport retries and their accounting.
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0)

    def _run_async(self, coroutine):
        async def run():
            if self.client.is_closed():
                self.client = self._create_client()
            try:
                return await coroutine
            finally:
                self.retriever.cancelled.set()
                await self.client.close()

        return asyncio.run(run())

    def generate_until(self, requests):
        self.retriever.cancelled.clear()
        self._prepared.clear()
        self._workloads.clear()
        try:
            results = super().generate_until(requests)
            for idx, result in enumerate(results):
                result.workload = {**(result.workload or {}), **self._workloads.get(idx, {})}
            return results
        finally:
            self._prepared.clear()
            self._workloads.clear()

    def clean(self):
        if self.encoder is not None:
            self.encoder.close()
        if not self.client.is_closed():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.client.close())
            else:
                loop.create_task(self.client.close())
        super().clean()
