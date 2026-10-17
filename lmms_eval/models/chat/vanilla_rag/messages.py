"""Task-preserving request preparation and baseline generation settings."""

from __future__ import annotations

import copy
import re

from .core import VideoSource, positive, read_window


def task_inputs(messages, doc, query_field="question"):
    question = doc.get(query_field)
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Task document needs nonempty text in rag_query_field={query_field!r}")
    result, references = copy.deepcopy(messages), []
    for message in result:
        content = message.get("content", "")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("Task messages must contain text and one video")
        for part in content:
            if part.get("type") == "video" and message.get("role") == "user":
                references.append(part["url"])
            elif part.get("type") != "text":
                raise ValueError("vanilla_rag accepts text and exactly one video per single-turn question")
    if len(references) != 1:
        raise ValueError("vanilla_rag requires exactly one task-supplied video")
    return VideoSource.resolve(references[0]), question, result


def evidence_messages(messages, source, selected, *, fps=1, longest_edge=448):
    import torch
    from playback.inference.native_video import encode_clip_video

    fps = positive(fps, "rag_clip_fps")
    longest_edge = positive(longest_edge, "rag_clip_longest_edge", integer=True)
    content = [{"type": "text", "text": f"Selected excerpts from the supplied video interval [{source.start_us / 1e6:g}, {source.end_us / 1e6:g}] s. Clips are chronological; timestamps are source-video seconds."}]
    sampled = []
    for item in selected:
        start, end = item["interval"]
        frames, times = read_window(source.path, round(start * 1e6), round(end * 1e6), fps=fps, longest_edge=longest_edge)
        payload = encode_clip_video(torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(), times, [start, end])
        content.extend([{"type": "text", "text": f"Interval [{start:g}, {end:g}] s:"}, {"type": "video_url", "video_url": {"url": payload}}])
        sampled.append({**item, "timestamps": times, "frames": len(times)})
    result = copy.deepcopy(messages)
    for message in result:
        if isinstance(message.get("content"), list):
            message["content"] = [piece for part in message["content"] for piece in (content if part.get("type") == "video" else [part])]
    return result, sampled


def thinking(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1", "yes"):
        return True
    if str(value).lower() in ("false", "0", "no"):
        return False
    raise ValueError("enable_thinking must be a boolean")


def task_stops(generation, *, enable_thinking=False):
    stop = (generation or {}).get("until")
    if not stop:
        return []
    if not isinstance(stop, (str, list)) or (isinstance(stop, list) and not all(isinstance(item, str) for item in stop)):
        raise ValueError("until must be a string or list of strings")
    return [item for item in ([stop] if isinstance(stop, str) else stop) if item and (not thinking(enable_thinking) or item.strip())]


def apply_answer_stops(answer, generation, *, enable_thinking=True):
    """Apply task delimiters to the final answer, after reasoning extraction."""
    # Qwen's parsed content commonly starts with the newlines after </think>.
    # Those are chat-template padding, not an empty answer terminated by until.
    answer = answer.lstrip()
    ends = [answer.index(stop) for stop in task_stops(generation, enable_thinking=enable_thinking) if stop in answer]
    return answer[: min(ends)] if ends else answer


def generation_payload(model, generation=None, *, enable_thinking=True, max_tokens=16384, seed=42):
    """Apply explicit experiment settings over short/greedy task YAML defaults."""
    enabled = thinking(enable_thinking)
    generation = copy.deepcopy(generation or {})
    if re.search(r"qwen3[._-]?5[-_]9b(?:$|[^a-z0-9])", model.lower()):
        presence = 1.5
    elif re.search(r"qwen3[._-]?8[-_]27b(?:$|[^a-z0-9])", model.lower()):
        presence = 0.0 if enabled else 1.5
    else:
        raise ValueError("vanilla_rag generation presets support Qwen3.5-9B and Qwen3.8-27B (including local paths containing those names)")
    # Only explicitly supported generation controls are forwarded. In particular,
    # inherited thinking caps and media resampling must not alter the experiment.
    extra = {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "seed": int(seed),
        "chat_template_kwargs": {"enable_thinking": enabled},
        "media_io_kwargs": {"video": {"video_backend": "playback_clip", "num_frames": -1}},
        "mm_processor_kwargs": {"do_sample_frames": False},
    }
    result = {"model": model, "max_tokens": positive(max_tokens, "rag_max_new_tokens", integer=True), "temperature": 1.0 if enabled else 0.7, "top_p": 0.95 if enabled else 0.8, "presence_penalty": presence, "extra_body": extra}
    stop = task_stops(generation)
    # lmms-eval injects the few-shot delimiter (usually two newlines). Sending
    # it to a reasoning model can stop inside <think>, before any answer exists.
    # Thinking runs ignore whitespace-only delimiters and apply other task
    # delimiters to parsed final content instead.
    if stop and not enabled:
        result["stop"] = stop
    return result
