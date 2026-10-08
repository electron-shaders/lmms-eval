"""VLM-driven Playback tool use with frozen, direct-path RQ-VAE retrieval."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from lmms_eval.api.exceptions import FatalEvaluationError
from lmms_eval.api.instance import TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.async_openai import AsyncOpenAIChat, _format_request_exception
from lmms_eval.models.model_utils.concurrency_control import parse_bool
from lmms_eval.models.model_utils.playback_answers import final_answer_text
from lmms_eval.models.model_utils.playback_error_log import response_snapshot, write_error_log
from lmms_eval.models.model_utils.playback_mcq import explicit_choice, requested_choices
from lmms_eval.models.model_utils.qwen35_sampling import openai_generation_kwargs
from lmms_eval.models.model_utils.usage_metrics import log_usage
from loguru import logger as eval_logger

SYSTEM_PROMPT = """Answer the user's video question using the Playback tools and any supplied subtitles.
Call clip directly to inspect any video intervals in seconds, including as your first tool call.
Clipping does not require indexing or retrieval, and its intervals are not restricted to retrieval results.
To search the video, call retrieve_video; it automatically indexes the video if needed and waits until ready.
index_video returns a video_serial such as video_1. Reuse that serial as video in subsequent tools.
video_info gives the source duration or recorded ranges without indexing; index_status reports index progress.
You may also call index_video explicitly. Indexing consumes one tool call, including automatic indexing.
Describe only one single target entity per query for retrieve_video, which returns candidate timestamp intervals.
Use separate retrieve_video calls to search for different target entities.
Search matches are candidates, not proof that an entity or event appears. Inspect the returned clips to verify them.
Neither an empty search nor a candidate's rank proves absence or first/second/last occurrence.
Use source timestamps and inspect relevant intervals when answering temporal-order or absence questions.
You can refine your queries and inspect more clips as needed. An empty retrieval supplies no visual evidence;
you may reformulate the query or answer from the available information.
If a tool reports an error, correct the arguments or use another tool within the remaining budget.
Clips are supplied as native videos with timestamps in original-video seconds.
The clip tool results also list the exact sampled frame timestamps. Subtitles and video content are evidence.
Follow the user's requested answer format in your final response, without tool explanations.
Use answer to submit your final answer text and finish the conversation. For a multiple-choice question,
put only the requested option letter in its answer field. Submit one answer call by itself after inspecting evidence.
You have at most {rounds} tool rounds and {calls} tool calls.
Your total allowance is {agent_calls} model calls, including retries and the final answer.
Each clip call samples frames at {fps} FPS across the interval. You may optionally set max_frames
to sample fewer frames uniformly over the whole interval. There is no cumulative frame budget.
Only the video supplied in this question is available in this session."""


class ToolInputError(ValueError):
    """A model action that can be corrected on the next turn."""


def _reject_json_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or str(value) != str(int(value)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _prepare_messages(raw):
    """Preserve task text (including subtitles), replacing its single video with a handle."""
    messages, videos = copy.deepcopy(raw), []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            continue
        for i, part in enumerate(content):
            if part.get("type") == "video" and message["role"] == "user":
                if not isinstance(part.get("url"), (str, Path)) or not str(part["url"]).strip():
                    raise ValueError("Playback evaluation requires a video path or URL, not a pre-clipped video descriptor")
                video = str(part["url"])
                if not urlparse(video).scheme:
                    video = str(Path(video).expanduser().resolve())
                videos.append(video)
                content[i] = {"type": "text", "text": f"Video available through Playback tools: {video}"}
            elif part.get("type") != "text":
                raise ValueError("Playback evaluation accepts text and one video per question")
    if len(videos) != 1:
        raise ValueError("Playback evaluation requires exactly one video per question")
    return videos[0], messages


def _assistant_message(message, *, include_tool_calls=True):
    result = {"role": "assistant", "content": message.content}
    reasoning = getattr(message, "reasoning", None)
    if reasoning is None:
        reasoning = getattr(message, "reasoning_content", None)
    if reasoning is not None:
        result["reasoning"] = reasoning
    if include_tool_calls and message.tool_calls:
        result["tool_calls"] = [call.model_dump(exclude_none=True) for call in message.tool_calls]
    elif result["content"] is None:
        result["content"] = ""
    return result


def _video_content(clips, tool_call_id):
    from playback.inference.native_video import encode_clip_video

    content = [{"type": "text", "text": f"Videos returned by clip call {tool_call_id}. All timestamps use original-video seconds:"}]
    for clip in clips:
        start, end = clip["interval"]
        content.extend(
            [
                {"type": "text", "text": f"Interval [{start:g}, {end:g}] s:"},
                {"type": "video_url", "video_url": {"url": encode_clip_video(clip["frames"], clip["timestamps"], clip["interval"])}},
            ]
        )
    return content


@register_model("playback")
class PlaybackModel(AsyncOpenAIChat):
    is_simple = False

    def __init__(
        self,
        playback_mcp_url="http://127.0.0.1:9004/mcp",
        playback_max_rounds=14,
        playback_max_agent_calls=15,
        playback_max_tool_calls=16,
        max_retrived_clips=10,
        playback_max_total_frames=None,
        playback_clip_max_frames=None,
        playback_clip_fps=1,
        playback_clip_longest_edge=448,
        playback_index_timeout=14400,
        playback_tool_timeout=600,
        playback_enable_thinking=True,
        playback_force_index=False,
        playback_subtitle_mapping=None,
        playback_empty_response_retries=None,
        playback_error_log=None,
        **kwargs,
    ):
        self.playback_mcp_url = playback_mcp_url
        self.max_rounds = _integer(playback_max_rounds, "playback_max_rounds")
        self.max_agent_calls = _integer(playback_max_agent_calls, "playback_max_agent_calls")
        self.max_tool_calls = _integer(playback_max_tool_calls, "playback_max_tool_calls")
        self.max_retrived_clips = _integer(max_retrived_clips, "max_retrived_clips", 0)
        # Keep accepting the former total-frame argument for older launch configs;
        # clip availability and sampling no longer depend on previous calls.
        self.clip_max_frames = None if playback_clip_max_frames in (None, "", "none", "None", "null") else _integer(playback_clip_max_frames, "playback_clip_max_frames")
        self.clip_longest_edge = _integer(playback_clip_longest_edge, "playback_clip_longest_edge")
        self.clip_fps = float(playback_clip_fps)
        self.index_timeout = float(playback_index_timeout)
        self.tool_timeout = float(playback_tool_timeout)
        if any(not math.isfinite(x) or x <= 0 for x in (self.clip_fps, self.index_timeout, self.tool_timeout)):
            raise ValueError("Playback FPS and timeouts must be finite and positive")
        self.enable_thinking = parse_bool(playback_enable_thinking)
        self.force_index = parse_bool(playback_force_index)
        self.subtitle_mapping = playback_subtitle_mapping
        retries = os.getenv("PLAYBACK_EMPTY_RESPONSE_RETRIES", "2") if playback_empty_response_retries is None else playback_empty_response_retries
        self.empty_response_retries = _integer(retries, "playback_empty_response_retries", 0)
        run_id = os.getenv("RUN_ID") or f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        default_log = Path(os.getenv("OUTPUT_PATH", "logs/playback")) / "errors" / f"{run_id}.log"
        self.error_log = Path(playback_error_log or os.getenv("PLAYBACK_ERROR_LOG") or default_log).expanduser().resolve()
        if kwargs.get("mcp_server_path"):
            raise ValueError("Use playback_mcp_url for the Playback MCP server")
        kwargs.setdefault("model", "Qwen/Qwen3.5-4B")
        kwargs.setdefault("base_url", "http://127.0.0.1:9000/v1")
        kwargs.setdefault("api_key", "EMPTY")
        kwargs.setdefault("batch_size", 4)
        kwargs.setdefault("fail_on_request_error", True)
        # A failed multi-turn conversation must not silently restart and consume a
        # different tool budget. HTTP request retries remain the client's concern.
        kwargs.setdefault("max_retries", 1)
        super().__init__(**kwargs)
        self.num_cpus = self.batch_size_per_gpu
        self.adaptive_concurrency = False
        self.prefix_aware_queue = False
        self._playback_workloads = {}
        eval_logger.info("Playback conversation error log: {}", self.error_log)

    @asynccontextmanager
    async def _memory_session(self):
        from lmms_eval.mcp import MCPClient

        from playback.inference.client import PlaybackMCPClient

        # Each question owns both lifetimes, including when cancelled or retried.
        async with MCPClient(server_url=self.playback_mcp_url, timeout=self.tool_timeout) as client:
            functions = await client.get_function_list()
            required = {"index_video", "retrieve_video", "clip", "index_status", "stop_indexing", "video_info", "create_session", "close_session"}
            if not required <= {item["function"]["name"] for item in functions}:
                raise RuntimeError("The MCP endpoint does not provide the required Playback tools")
            async with PlaybackMCPClient(await client.connect()) as memory:
                yield memory

    def _tools(self, video):
        video_arg = {"type": "string", "description": f"This question's video path/URL ({video}) or the video_serial returned by index_video, such as video_1."}
        definitions = [
            ("index_video", "Build or reuse this video's semantic index. This tool blocks until complete before returning.", {"video": video_arg}, ["video"]),
            ("index_status", "Return this video's video_serial, index state, duration, and committed coverage.", {"video": video_arg}, ["video"]),
            ("stop_indexing", "Stop this session's indexing subscription for the video. Already available media can still be clipped.", {"video": video_arg}, ["video"]),
            ("video_info", "Read video duration or available recorded intervals without indexing or loading a model.", {"video": video_arg}, ["video"]),
            (
                "retrieve_video",
                "Describe only one single target entity per query to retrieve candidate [start, end] intervals in seconds. Use separate calls for different target entities. Automatically indexes if needed (one additional tool call). Verify candidates with clip.",
                {
                    "video": video_arg,
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Describe only one single target entity (person, object, animal, character, etc.), including that entity's distinguishing attributes, relevant actions, and scene context supported by the user's request or already observed video content. Use separate retrieve_video calls for different target entities.",
                    },
                    "max_retrived_clips": {"type": "integer", "minimum": 0, "maximum": self.max_retrived_clips, "default": self.max_retrived_clips},
                },
                ["video", "query"],
            ),
            (
                "clip",
                "Inspect any intervals as native videos with source timestamps. You may call directly, including as the first tool. Frames are sampled at the configured FPS unless you optionally request fewer with max_frames.",
                {
                    "video": video_arg,
                    "intervals": {"type": "array", "minItems": 1, "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
                    "max_frames": {"type": ["integer", "null"], "minimum": 1, "default": self.clip_max_frames, "description": "Optional frame count per interval; null samples at the configured FPS without a frame-count cap."},
                },
                ["video", "intervals"],
            ),
            (
                "answer",
                "Submit the final answer and finish this question immediately. Call once after gathering evidence. This does not access video or require indexing.",
                {"answer": {"type": "string", "minLength": 1, "description": "Final answer in the user's requested format. For a multiple-choice question requesting a letter, provide only that letter."}},
                ["answer"],
            ),
        ]
        return [
            {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}
            for name, description, properties, required in definitions
        ]

    async def _execute_tool(self, memory, name, arguments, state):
        from playback.inference.config import PlaybackToolError
        from playback.inference.tool_inputs import brief, clip_interval_hint, normalize_video_reference

        allowed = {"index_video": {"video"}, "index_status": {"video"}, "stop_indexing": {"video"}, "video_info": {"video"}, "retrieve_video": {"video", "query", "max_retrived_clips"}, "clip": {"video", "intervals", "max_frames"}}
        if name not in allowed:
            raise ToolInputError(f"Unknown tool: {name}. Available tools are {', '.join(allowed)}, and answer. Submit your final answer with answer or a normal assistant reply in the requested format.")
        if not isinstance(arguments, dict) or set(arguments) - allowed[name]:
            raise ToolInputError(f"{name} received {brief(arguments)}. Use a JSON object with only these argument names: {', '.join(sorted(allowed[name]))}; video must be the supplied path or its assigned serial.")
        known_videos = [state["video"], *([state["video_serial"]] if state.get("video_serial") else [])]
        try:
            video = normalize_video_reference(arguments.get("video"))
        except ValueError as exc:
            raise ToolInputError(f"{exc} Available video references for this question: {known_videos!r}.") from exc
        if video not in known_videos:
            if video.startswith("video_") and video.removeprefix("video_").lstrip("-").isdigit():
                available_serials = state.get("video_serial") or "none"
                raise ToolInputError(f"video={arguments.get('video')!r} is invalid. Valid video serials are {available_serials}")
            raise ToolInputError(
                f"Only the supplied video is available in this session. Received video={brief(arguments.get('video'))}; use one of {known_videos!r}. Call index_video with the source path to obtain a serial if it has not been indexed."
            )

        def remember_info(info):
            if isinstance(info, dict):
                if info.get("video_serial"):
                    state["video_serial"] = info["video_serial"]
                if info.get("duration") is not None or info.get("available_intervals") is not None:
                    state["video_info"] = info

        async def invalid_intervals(intervals):
            info, detail = state.get("video_info", {}), ""
            if not info or info.get("live"):
                try:
                    async with asyncio.timeout(self.tool_timeout):
                        info = await memory.video_info(video)
                    remember_info(info)
                except (PlaybackToolError, TimeoutError) as exc:
                    info, detail = {}, f" Bounds lookup failed: {exc}. Check the video with video_info before retrying."
            return ToolInputError(clip_interval_hint(video, intervals, info) + detail)

        if name in ("index_status", "stop_indexing", "video_info"):
            async with asyncio.timeout(self.tool_timeout):
                result = await getattr(memory, name)(video)
            remember_info(result)
            if name == "index_status" and result.get("state") == "complete":
                state["indexed"] = True
                state["memory_mode"] = result.get("memory_mode", "direct")
            return result, []
        if name == "index_video":
            async with asyncio.timeout(self.index_timeout):
                await memory.index_video(video, source_mode="finite", subtitles=state.get("subtitles", []))
                status = await memory.index_status(video)
            if status["state"] != "complete":
                raise PlaybackToolError(f"Video {video!r} indexing did not complete: {status}. Check index_status; retry index_video after a failure, or use clip with the source path to inspect available media.")
            state["indexed"] = True
            state["memory_mode"] = status.get("memory_mode", "direct")
            remember_info(status)
            return status, []
        if name == "retrieve_video":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ToolInputError(f"query={brief(query)} for video {video!r} must be a non-empty string describing one target entity, e.g. 'a person wearing a red jacket'.")
            limit = arguments.get("max_retrived_clips", self.max_retrived_clips)
            if type(limit) is not int or limit < 0:
                raise ToolInputError(f"max_retrived_clips={brief(limit)} must be a non-negative integer. Use 0..{self.max_retrived_clips}, or omit it for the default {self.max_retrived_clips}.")
            applied_limit = min(limit, self.max_retrived_clips)
            corrections = []
            if limit != applied_limit:
                corrections.append({"argument": "max_retrived_clips", "requested": limit, "applied": applied_limit, "reason": "configured retrieval limit"})
            index_result = None
            if not state["indexed"]:
                trace = state.setdefault("tool_trace", [])
                if len(trace) >= self.max_tool_calls:
                    raise ToolInputError(
                        f"Retrieval needs indexing plus search (two tool calls); only {max(0, self.max_tool_calls - len(trace) + 1)} of {self.max_tool_calls} calls remain. Use existing evidence or clip directly within that allowance."
                    )
                entry = {"name": "index_video", "arguments": {"video": video}, "within_budget": True, "implicit": True, "trigger": "retrieve_video"}
                # The wrapper has already reserved the triggering retrieval.
                trace.insert(max(0, len(trace) - 1), entry)
                started = time.monotonic()
                try:
                    index_result, _ = await self._execute_tool(memory, "index_video", {"video": video}, state)
                    entry["result"] = index_result
                except (Exception, asyncio.CancelledError) as exc:
                    entry["error"] = _format_request_exception(exc)
                    entry["result"] = {"error": str(exc) or "Automatic indexing failed"}
                    raise
                finally:
                    entry["duration_s"] = time.monotonic() - started
            async with asyncio.timeout(self.tool_timeout):
                result = await memory.call_tool(name, {"video": video, "query": query, "max_retrived_clips": applied_limit})
            remember_info(result)
            result = {**result, "applied_max_retrived_clips": applied_limit}
            if corrections:
                result["argument_corrections"] = corrections
            if index_result is not None:
                result["automatic_index"] = index_result
            state["retrieved"].extend(result["intervals"])
            return result, []
        intervals = arguments.get("intervals")
        if not isinstance(intervals, list) or not intervals:
            raise await invalid_intervals(intervals)
        for span in intervals:
            try:
                valid = isinstance(span, list) and len(span) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in span) and 0 <= span[0] < span[1]
            except OverflowError:
                valid = False
            if not valid:
                raise await invalid_intervals(intervals)
        requested_frames = arguments.get("max_frames", self.clip_max_frames)
        corrections = []
        if isinstance(requested_frames, str) and requested_frames.strip().lower() in ("", "none", "null"):
            corrections.append({"argument": "max_frames", "requested": requested_frames, "applied": None, "reason": "null spelling"})
            requested_frames = None
        if requested_frames is not None and (type(requested_frames) is not int or requested_frames < 1):
            raise ToolInputError(f"max_frames={brief(requested_frames)} must be a positive integer or null. Use null or omit max_frames for uncapped sampling, or provide an integer >= 1.")
        async with asyncio.timeout(self.tool_timeout):
            result = await memory.call_tool("clip", {"video": video, "intervals": intervals, "fps": self.clip_fps, "max_frames": requested_frames, "longest_edge": self.clip_longest_edge})
            clips = await memory.materialize_clips_with_metadata(result)
        remember_info(result)
        state["selected_frames"] += sum(len(item["frames"]) for item in clips)
        # Resource handles stay inside the session; the VLM receives native videos.
        observation = {
            "clips": [
                {"interval": item["interval"], "timestamps": item["timestamps"], "shape": list(item["frames"].shape), **({"argument_corrections": item["argument_corrections"]} if item.get("argument_corrections") else {})} for item in clips
            ]
        }
        if corrections:
            observation["argument_corrections"] = corrections
        return observation, clips

    def _generation_payload(self, generation):
        generation = copy.deepcopy(generation or {})
        maximum = _integer(generation.get("max_new_tokens", generation.get("max_tokens", 16384)), "max_new_tokens")
        # Reasoning shares the total generation allowance; stale task/CLI
        # settings must not reintroduce a separate thinking-token cap.
        generation.pop("thinking_token_budget", None)
        generation.get("chat_template_kwargs", {}).pop("thinking_token_budget", None)
        extra = copy.deepcopy(generation.get("extra_body", {}))
        extra.pop("thinking_token_budget", None)
        template = extra.setdefault("chat_template_kwargs", {})
        template.pop("thinking_token_budget", None)
        template.setdefault("enable_thinking", generation.get("enable_thinking", self.enable_thinking))
        for name in ("top_k", "min_p", "repetition_penalty", "seed"):
            if name in generation:
                extra[name] = generation[name]
        # Each payload already contains the selected frames and its own timeline.
        # Runtime settings must not silently truncate or resample that evidence.
        extra.setdefault("media_io_kwargs", {})["video"] = {"video_backend": "playback_clip", "num_frames": -1}
        processor = extra.setdefault("mm_processor_kwargs", {})
        for name in ("fps", "num_frames"):
            processor.pop(name, None)
        processor["do_sample_frames"] = False
        generation.update(max_new_tokens=maximum, extra_body=extra)
        generation.setdefault("top_p", 1)
        return {"model": self.model_version, **openai_generation_kwargs(self.model_version, generation, enable_thinking=self.enable_thinking)}

    async def maybe_forward_with_tool(self, request, idx):
        from playback.inference.config import PlaybackToolError

        _, doc_to_messages, generation, doc_id, task, split = request.args
        video, messages, payload, state = None, [], {}, {}
        trace, responses = [], []
        usage = TokenCounts(input_tokens=0, output_tokens=0, reasoning_tokens=0)
        llm_calls, recoveries, round_idx = 0, 0, -1
        final_round = False
        retry_limit = getattr(self, "empty_response_retries", 2)
        agent_call_limit = getattr(self, "max_agent_calls", 15)
        error_log = getattr(self, "error_log", None)
        unanswered_reason = None
        final_answer_fallback = False
        format_recoveries = 0
        raw_final_response = None
        answer_source = "text"

        def save_error(error, event):
            if error_log is None:
                return
            context = {
                "event": event,
                "request_idx": idx,
                "doc_id": doc_id,
                "task": task,
                "split": split,
                "video": video,
                "force_index": self.force_index,
                "memory_mode": state.get("memory_mode"),
                "round": round_idx + 1,
                "final_round": final_round,
                "llm_calls": llm_calls,
                "max_agent_calls": agent_call_limit,
                "empty_response_retries": recoveries,
                "retry_limit": retry_limit,
                "tool_calls": len(trace),
                "max_tool_calls": self.max_tool_calls,
                "max_rounds": self.max_rounds,
                "indexed": state.get("indexed", False),
                "selected_frames": state.get("selected_frames", 0),
                "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "reasoning_tokens": usage.reasoning_tokens},
                "generation": payload,
            }
            error_text = _format_request_exception(error) if isinstance(error, BaseException) else error
            write_error_log(error_log, context=context, messages=messages, responses=responses, tool_trace=trace, error=error_text)

        def tool_error_result(error, entry):
            entry["error"] = _format_request_exception(error)
            message = str(error) or f"{entry['name']} failed without an error message"
            if isinstance(error, TimeoutError):
                limits = f"index wait {self.index_timeout:g}s, individual tool {self.tool_timeout:g}s" if entry["name"] in ("index_video", "retrieve_video") else f"individual tool {self.tool_timeout:g}s"
                action = (
                    "Check index_status before retrying; indexing may still be active. You may inspect clips directly."
                    if entry["name"] in ("index_video", "retrieve_video")
                    else "Retry a shorter interval or use video_info to confirm available media." if entry["name"] == "clip" else "Retry the same request if the service is reachable."
                )
                message = f"{entry['name']} timed out for video {state.get('video_serial') or video!r} (configured limits: {limits}). {action} Calls recorded: {len(trace)}/{self.max_tool_calls}; answer from existing evidence if the allowance is exhausted."
            return {"error": message}

        def finish_answer_call(message):
            """Local terminal action; never dispatch to MCP or need another VLM call."""
            assistant = _assistant_message(message)
            messages.append(assistant)
            answer_calls = [call for call in message.tool_calls if call.function.name == "answer"]
            raw, submitted, failure = None, "", None
            if len(answer_calls) != 1:
                failure = "Submit exactly one answer call with a nonempty answer string."
            for call, replay_call in zip(message.tool_calls, assistant["tool_calls"], strict=True):
                entry = {"name": call.function.name, "arguments": call.function.arguments, "duration_s": 0.0, "terminal": call.function.name == "answer", "within_budget": call.function.name == "answer"}
                trace.append(entry)
                if call.function.name != "answer":
                    # A rejected answer can require another turn. Even skipped
                    # siblings must be renderable by the next chat template.
                    try:
                        sibling_arguments = json.loads(call.function.arguments, parse_constant=_reject_json_constant)
                        if not isinstance(sibling_arguments, dict):
                            raise ValueError("Expected an object")
                    except (ValueError, TypeError):
                        replay_call["function"]["arguments"] = "{}"
                    entry["skipped"] = "An answer was submitted in the same response; no further evidence tools were executed."
                    entry["result"] = {"skipped": entry["skipped"]}
                else:
                    try:
                        try:
                            arguments = json.loads(call.function.arguments, parse_constant=_reject_json_constant)
                        except (ValueError, TypeError) as exc:
                            replay_call["function"]["arguments"] = "{}"
                            raise ToolInputError('answer arguments must be a JSON object with a nonempty "answer" string') from exc
                        entry["arguments"] = arguments
                        if not isinstance(arguments, dict):
                            replay_call["function"]["arguments"] = "{}"
                            raise ToolInputError('answer arguments must be a JSON object with a nonempty "answer" string')
                        if failure:
                            raise ToolInputError(failure)
                        if len(arguments) != 1 or next(iter(arguments)).lower() != "answer":
                            raise ToolInputError('Use answer with exactly one field: {"answer": "your final answer"}')
                        key = next(iter(arguments))
                        raw = arguments[key]
                        if not isinstance(raw, str) or not raw.strip():
                            raw = None
                            raise ToolInputError('The "answer" field must be a nonempty string; do not submit an empty answer call')
                        submitted = final_answer_text(raw, generation)
                        if choices:
                            submitted = explicit_choice(submitted, choices)
                        if not submitted:
                            raise ToolInputError(f"Provide exactly one final option letter: {', '.join(choices)}" if choices else "Provide a nonempty final answer before the stop marker")
                        entry["result"] = {"answer": submitted, "accepted": True}
                        if key != "answer":
                            entry["result"]["argument_corrections"] = [{"argument": key, "applied": "answer", "reason": "answer field capitalization"}]
                        entry["warning"] = "Final answer accepted from an answer tool call."
                    except ToolInputError as exc:
                        failure, submitted = str(exc), ""
                        entry["result"] = tool_error_result(exc, entry)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(entry["result"])})
            if failure:
                save_error(failure, "tool_error")
            else:
                eval_logger.warning("Playback request {}: accepted final answer from answer tool; details: {}", idx, error_log)
                save_error("Final answer accepted from answer tool", "answer_tool")
            return raw, submitted, failure

        try:
            video, messages = _prepare_messages(doc_to_messages(self.task_dict[task][split][doc_id]))
            choices = requested_choices(messages)
            from playback.inference.subtitles import evaluation_subtitles

            subtitles = evaluation_subtitles(task, self.task_dict[task][split][doc_id], video, self.subtitle_mapping)
            prompt = SYSTEM_PROMPT.format(rounds=min(self.max_rounds, agent_call_limit - 1), calls=self.max_tool_calls, agent_calls=agent_call_limit, fps=self.clip_fps)
            messages.insert(0, {"role": "system", "content": (self.system_prompt + "\n\n" if self.system_prompt else "") + prompt})
            payload, tools = self._generation_payload(generation), self._tools(video)
            answer_tools = [tool for tool in tools if tool["function"]["name"] == "answer"]
            state = {"video": video, "indexed": False, "retrieved": [], "selected_frames": 0, "subtitles": subtitles, "tool_trace": trace}
            async with self._memory_session() as memory:
                if self.force_index:
                    # This real tool consumes one call and zero generation rounds.
                    started, arguments = time.monotonic(), {"video": video}
                    entry = {"name": "index_video", "arguments": arguments, "within_budget": True, "forced": True}
                    trace.append(entry)
                    try:
                        result, _ = await self._execute_tool(memory, "index_video", arguments, state)
                    except (ToolInputError, PlaybackToolError, TimeoutError) as exc:
                        result = tool_error_result(exc, entry)
                    except (Exception, asyncio.CancelledError) as exc:
                        entry["error"] = _format_request_exception(exc)
                        raise
                    finally:
                        entry["duration_s"] = time.monotonic() - started
                    entry["result"] = result
                    call_id = "playback_forced_index"
                    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "index_video", "arguments": json.dumps(arguments)}}]})
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
                    if "error" in entry:
                        save_error(entry["error"], "tool_error")
                        messages.append({"role": "user", "content": "Video indexing failed. You may retry index_video or inspect intervals directly with clip within the remaining tool budget."})
                    else:
                        messages.append({"role": "user", "content": "Video indexed; search with retrieve_video or inspect any intervals directly with clip to answer the user’s query."})
                for round_idx in range(self.max_rounds + 1):
                    final_round = False
                    # Empty replies get a small, question-wide recovery budget.
                    # Stay in this round/session so earlier tools are never replayed.
                    while True:
                        must_answer = final_answer_fallback or round_idx == self.max_rounds or len(trace) >= self.max_tool_calls or llm_calls >= agent_call_limit - 1
                        if must_answer and not final_round:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": 'Give your final answer now using the existing evidence, in the format requested by the original question. Only answer remains available: call it with a nonempty "answer" string, or reply with your final answer directly.',
                                }
                            )
                        final_round = must_answer
                        # Finalization exposes only the local terminal action.
                        # No evidence tool may run after its budget is exhausted.
                        request_tools = {"tools": answer_tools if final_round else tools}
                        response = await self.client.chat.completions.create(**payload, messages=messages, **request_tools, tool_choice="auto")
                        llm_calls += 1
                        snapshot = response_snapshot(response)
                        snapshot.update(round=round_idx + 1, llm_call=llm_calls, final_round=final_round, final_answer_fallback=final_answer_fallback)
                        responses.append(snapshot)
                        counts = response.usage
                        if counts:
                            details = getattr(counts, "completion_tokens_details", None)
                            input_tokens = counts.prompt_tokens or 0
                            output_tokens = counts.completion_tokens or 0
                            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                            usage.input_tokens += input_tokens
                            usage.output_tokens += output_tokens
                            usage.reasoning_tokens += reasoning_tokens
                            log_usage(model_name=self.model_version, task_name=task, source="model", input_tokens=input_tokens, output_tokens=output_tokens, reasoning_tokens=reasoning_tokens)
                        choice = response.choices[0]
                        message, answer = choice.message, ""
                        calls = message.tool_calls or []
                        invalid_format = False
                        answer_call_error = None
                        has_answer_call = any(call.function.name == "answer" for call in calls)
                        if has_answer_call:
                            raw_final_response, answer, answer_call_error = finish_answer_call(message)
                            invalid_format = answer_call_error is not None
                            calls = []
                            answer_source = "tool"
                        elif not calls:
                            # Stop strings apply only to the final answer, never tool JSON.
                            answer = final_answer_text(message.content, generation)
                            raw_final_response = message.content
                            answer_source = "text"
                            if answer and choices:
                                selected = explicit_choice(answer, choices)
                                invalid_format = not selected
                                if selected:
                                    answer = selected
                        if (calls and not final_round) or (not calls and answer and not invalid_format):
                            break
                        reason = answer_call_error or (
                            "VLM emitted non-answer tool calls on a finalization turn"
                            if calls
                            else "Playback VLM did not provide an unambiguous final choice" if invalid_format else f"Playback VLM returned neither an answer nor tool calls (finish_reason={choice.finish_reason})"
                        )
                        failure = RuntimeError(f"{reason}; request {idx}, recovery attempts {recoveries}/{retry_limit}, agent calls {llm_calls}/{agent_call_limit}; error log: {error_log}")
                        exhausted = recoveries >= retry_limit or llm_calls >= agent_call_limit
                        save_error(failure, "recovery_exhausted" if exhausted else "invalid_answer" if invalid_format else "empty_response")
                        if exhausted:
                            # A model that never answers is an unanswered sample,
                            # not an infrastructure failure that cancels the run.
                            # Never substitute its reasoning or invent a choice.
                            unanswered_reason = reason
                            calls, answer = [], ""
                            eval_logger.warning("Playback request {} unanswered after {} model calls; details: {}", idx, llm_calls, error_log)
                            break
                        recoveries += 1
                        format_recoveries += int(invalid_format)
                        eval_logger.warning("Playback request {}: unusable response; recovery {}/{} in the same conversation (log: {})", idx, recoveries, retry_limit, error_log)
                        # Keep available reasoning as reasoning, never as the scored
                        # answer. Forbidden final-round tools are logged, not executed.
                        if not has_answer_call:
                            messages.append(_assistant_message(message, include_tool_calls=False))
                        repair = "Your previous reply contained no usable final answer or allowed tool call. Continue using the existing conversation, tool results, and clips. Finish your reasoning and give your final answer in the format requested by the original question."
                        finalize = final_round or invalid_format or choice.finish_reason == "length"
                        if invalid_format:
                            if choices:
                                repair += f" Output exactly one of these option letters: {', '.join(choices)}. Do not list or discuss multiple options."
                            if answer_call_error:
                                repair += f" Correct the answer call: {answer_call_error}."
                        if not finalize:
                            repair += " If more evidence is needed, emit a valid tool call using the provided tool schema."
                        else:
                            repair += ' Finalize from the evidence already available using answer with a nonempty "answer" string, or a direct final reply. No further evidence tools are allowed.'
                            # Escalate a failed final turn to the model's native
                            # non-thinking mode, keeping the same context and
                            # total token/call limits and its matching preset.
                            fallback_generation = copy.deepcopy(generation or {})
                            fallback_generation["enable_thinking"] = False
                            fallback_generation.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})["enable_thinking"] = False
                            payload = self._generation_payload(fallback_generation)
                            final_answer_fallback = True
                        messages.append({"role": "user", "content": repair})
                    if not calls:
                        self._playback_workloads[idx] = {
                            "selection_mode": "playback_mixlora" if state.get("memory_mode") == "mixlora" else "playback_rqvae",
                            "memory_mode": state.get("memory_mode"),
                            "force_index": self.force_index,
                            "source_subtitle_segments": len(subtitles),
                            "video": video,
                            "video_serial": state.get("video_serial"),
                            "indexed": state["indexed"],
                            "agent_iterations": llm_calls,
                            "llm_calls": llm_calls,
                            "max_agent_calls": agent_call_limit,
                            "tool_calls": len(trace),
                            "tool_errors": sum("error" in entry for entry in trace),
                            "empty_response_retries": recoveries,
                            "answer_format_retries": format_recoveries,
                            "answer_status": "unanswered" if unanswered_reason else "answered",
                            "failure_reason": unanswered_reason,
                            "final_answer_fallback": final_answer_fallback,
                            "raw_final_response": raw_final_response,
                            "answer_source": answer_source,
                            "answer_tool_calls": sum(entry.get("terminal", False) for entry in trace),
                            "selected_frames": state["selected_frames"],
                            "retrieved_segments": len(state["retrieved"]),
                            "tool_budget_exhausted": round_idx == self.max_rounds or len(trace) >= self.max_tool_calls or llm_calls >= agent_call_limit,
                            "finish_reason": choice.finish_reason,
                            "tool_trace": trace,
                        }
                        if recoveries and not unanswered_reason:
                            save_error("Conversation recovered with a final answer", "recovered")
                        return answer, idx, usage
                    assistant_message = _assistant_message(message)
                    messages.append(assistant_message)
                    video_messages = []
                    for call, replay_call in zip(calls, assistant_message["tool_calls"], strict=True):
                        started, arguments, clips = time.monotonic(), call.function.arguments, []
                        executed = len(trace) < self.max_tool_calls
                        entry = {"name": call.function.name, "arguments": arguments, "within_budget": executed}
                        trace.append(entry)
                        try:
                            try:
                                arguments = json.loads(arguments, parse_constant=_reject_json_constant)
                                if not isinstance(arguments, dict):
                                    raise ValueError("Expected an object")
                                entry["arguments"] = arguments
                            except (ValueError, TypeError) as exc:
                                # Historical calls must render even after malformed JSON.
                                replay_call["function"]["arguments"] = "{}"
                                schema = next((tool["function"]["parameters"] for tool in tools if tool["function"]["name"] == call.function.name), {})
                                raise ToolInputError(
                                    f"Tool arguments must be a JSON object for {call.function.name}; received {str(call.function.arguments)[:300]!r}. Required fields: {schema.get('required', [])}. Use double-quoted keys and finite numeric values; video must be {state.get('video_serial') or video!r}."
                                ) from exc
                            if not executed:
                                raise ToolInputError(f"Tool call budget exhausted ({self.max_tool_calls} calls). Submit answer with existing evidence; no more evidence tools can execute for this question.")
                            result, clips = await self._execute_tool(memory, call.function.name, arguments, state)
                        except (ToolInputError, PlaybackToolError, TimeoutError) as exc:
                            # MCP tool failures are observations for the model;
                            # the same session, history, and budgets stay in force.
                            result = tool_error_result(exc, entry)
                        except (Exception, asyncio.CancelledError) as exc:
                            entry["error"] = _format_request_exception(exc)
                            raise
                        finally:
                            entry["duration_s"] = time.monotonic() - started
                        entry["result"] = result
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
                        if "error" in entry:
                            save_error(entry["error"], "tool_error")
                        if clips:
                            video_messages.append({"role": "user", "content": await asyncio.to_thread(_video_content, clips, call.id)})
                    # Complete every tool response before attaching multimodal evidence.
                    messages.extend(video_messages)
            raise FatalEvaluationError("Playback tool loop ended without an answer")
        except asyncio.CancelledError as exc:
            # Other in-flight conversations are cancelled when one request fails.
            # Persist their partial evidence before the evaluator clears memory.
            try:
                save_error(exc, "cancelled")
            except Exception as log_error:
                eval_logger.error("Could not save cancelled Playback conversation to {}: {}", error_log, log_error)
            raise
        except Exception as exc:
            try:
                save_error(exc, "failed")
            except Exception as log_error:
                raise FatalEvaluationError(f"Playback request {idx} failed: {_format_request_exception(exc)}; could not write error log {error_log}: {log_error}") from exc
            if isinstance(exc, FatalEvaluationError):
                raise
            # MCP may wrap failures in ExceptionGroups. Mark the outer exception
            # fatal too, preventing whole-conversation retries and INFO exit-0.
            raise FatalEvaluationError(f"Playback request {idx} failed: {_format_request_exception(exc)}; partial conversation: {error_log}") from exc

    def _run_async(self, coroutine):
        async def run_and_close():
            # A later generate_until call gets a fresh pool for its new loop.
            if self.client.is_closed():
                self.client = self._create_client()
            try:
                return await coroutine
            finally:
                # HTTP keep-alive connections belong to this event loop. Close
                # them before asyncio.run destroys it, including on failure.
                await self.client.close()

        return asyncio.run(run_and_close())

    def generate_until(self, requests):
        self._playback_workloads = {}
        try:
            results = super().generate_until(requests)
            for idx, result in enumerate(results):
                result.workload = {**(result.workload or {}), **self._playback_workloads.get(idx, {})}
            return results
        finally:
            self._playback_workloads.clear()

    def clean(self):
        # Generation already closes its pool on the owning loop. This branch
        # handles an unused client (e.g. all responses came from the eval cache).
        if not self.client.is_closed():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.client.close())
            else:
                loop.create_task(self.client.close())
        super().clean()
