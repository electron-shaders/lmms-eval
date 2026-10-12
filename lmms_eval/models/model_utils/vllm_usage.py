"""Per-request accounting for native vLLM outputs, without retokenizing text."""

from dataclasses import asdict, is_dataclass
from math import isfinite

from loguru import logger as eval_logger

from lmms_eval.api.instance import GenerationResult, TokenCounts


def _duration(start, end):
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and isfinite(start) and isfinite(end) and 0 < start <= end:
        return end - start
    return None


def _request_timings(metrics, output_tokens, completion_count):
    """Keep clock domains separate; never substitute batch time for request time."""
    values = {
        "latency_s": None,
        "ttft_s": None,
        "queue_time_s": None,
        "prefill_time_s": None,
        "decode_time_s": None,
        "inference_time_s": None,
        "tpot_s": None,
    }
    if not metrics:
        return values
    if "first_token_ts" in metrics:
        # vLLM v1: core timestamps are monotonic; arrival_time is wall clock.
        queued = metrics.get("queued_ts")
        scheduled = metrics.get("scheduled_ts")
        first = metrics.get("first_token_ts")
        last = metrics.get("last_token_ts")
        values["latency_s"] = _duration(queued, last)
        values["latency_scope"] = "engine_queue_to_last_token"
        values["queue_time_s"] = _duration(queued, scheduled)
        ttft = metrics.get("first_token_latency")
        if isinstance(ttft, (int, float)) and isfinite(ttft) and ttft > 0:
            values["ttft_s"] = ttft
    else:
        # Older vLLM RequestMetrics uses wall-clock timestamps throughout.
        arrival = metrics.get("arrival_time")
        scheduled = metrics.get("first_scheduled_time")
        first = metrics.get("first_token_time")
        last = metrics.get("last_token_time")
        values["latency_s"] = _duration(arrival, metrics.get("finished_time"))
        values["latency_scope"] = "frontend_arrival_to_finish"
        values["queue_time_s"] = _duration(arrival, scheduled)
        values["ttft_s"] = _duration(arrival, first)
    values["prefill_time_s"] = _duration(scheduled, first)
    values["decode_time_s"] = _duration(first, last)
    values["inference_time_s"] = _duration(scheduled, last)
    if completion_count == 1 and output_tokens is not None and output_tokens > 1 and values["decode_time_s"] is not None:
        values["tpot_s"] = values["decode_time_s"] / (output_tokens - 1)
    return values


class VLLMUsageRecorder:
    def __init__(self, client, reasoning_parser=None):
        self.client = client
        self.reasoning_parser = reasoning_parser
        self._parser_cls = None
        self._tokenizer = None
        self._parser_unavailable = False
        self._warned = False

    def _reasoning_counts(self, output, template_options):
        unknown = [None] * len(output.outputs)
        if not self.reasoning_parser or self._parser_unavailable:
            return unknown
        try:
            if self._parser_cls is None:
                from vllm.reasoning import ReasoningParser, ReasoningParserManager

                parser_cls = ReasoningParserManager.get_reasoning_parser(self.reasoning_parser)
                if getattr(parser_cls, "count_reasoning_tokens", None) is getattr(ReasoningParser, "count_reasoning_tokens", None):
                    # The base implementation returns zero even when unsupported.
                    self._parser_unavailable = True
                    return unknown
                self._tokenizer = self.client.get_tokenizer()
                self._parser_cls = parser_cls
            counts = []
            for completion in output.outputs:
                token_ids = getattr(completion, "token_ids", None)
                if token_ids is None:
                    counts.append(None)
                    continue
                parser = self._parser_cls(self._tokenizer, chat_template_kwargs=template_options.get("chat_template_kwargs", {}))
                count = parser.count_reasoning_tokens(token_ids)
                if not isinstance(count, int) or not 0 <= count <= len(token_ids):
                    raise ValueError("vLLM returned an invalid reasoning token count")
                counts.append(count)
            return counts
        except Exception as exc:
            if not self._warned:
                eval_logger.warning("Cannot count vLLM reasoning tokens; leaving them unknown: {}", exc)
                self._warned = True
            if self._parser_cls is None:
                self._parser_unavailable = True
            return unknown

    def record(self, output, template_options):
        prompt_ids = getattr(output, "prompt_token_ids", None)
        encoder_ids = getattr(output, "encoder_prompt_token_ids", None)
        input_tokens = None if prompt_ids is None else len(prompt_ids) + (len(encoder_ids) if encoder_ids is not None else 0)
        reasoning_counts = self._reasoning_counts(output, template_options)
        completions = []
        for completion, reasoning_tokens in zip(output.outputs, reasoning_counts):
            token_ids = getattr(completion, "token_ids", None)
            completions.append(
                {
                    "index": getattr(completion, "index", len(completions)),
                    "output_tokens": len(token_ids) if token_ids is not None else None,
                    "reasoning_tokens": reasoning_tokens,
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                }
            )
        # All returned candidates cost tokens, even though evaluation uses the first.
        output_tokens = sum(c["output_tokens"] for c in completions) if completions and all(c["output_tokens"] is not None for c in completions) else None
        reasoning_tokens = sum(reasoning_counts) if reasoning_counts and all(c is not None for c in reasoning_counts) else None
        counts = TokenCounts(input_tokens=input_tokens, output_tokens=output_tokens, reasoning_tokens=reasoning_tokens)
        cached_tokens = getattr(output, "num_cached_tokens", None)
        metrics = getattr(output, "metrics", None)
        raw_metrics = asdict(metrics) if is_dataclass(metrics) else dict(vars(metrics)) if metrics is not None else None
        workload = {
            "llm_calls": 1,
            "request_id": getattr(output, "request_id", None),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None,
            "reasoning_tokens": reasoning_tokens,
            "non_reasoning_output_tokens": output_tokens - reasoning_tokens if output_tokens is not None and reasoning_tokens is not None else None,
            "cached_input_tokens": cached_tokens,
            "uncached_input_tokens": input_tokens - cached_tokens if input_tokens is not None and cached_tokens is not None else None,
            "cache_creation_input_tokens": getattr(output, "num_cache_creation_tokens", None),
            "usage_complete": input_tokens is not None and output_tokens is not None,
            "completions": completions,
            "server_metrics": raw_metrics,
            **_request_timings(raw_metrics, output_tokens, len(completions)),
        }
        return GenerationResult(text=output.outputs[0].text, token_counts=counts, workload=workload)
