"""Best-effort bridge from lmms-eval into the launcher telemetry sidecar."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = "1.0"
_WORKLOAD_TOTALS: dict[str, float] = {}
_WORKLOAD_BY_TASK: dict[str, dict[str, float]] = {}
_WORKLOAD_SELECTION_MODES: dict[str, int] = {}
_WORKLOAD_SAMPLES = 0
_ADDITIVE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "visual_tokens",
    "decoded_frames",
    "selected_frames",
    "llm_calls",
    "embedding_requests",
    "tool_calls",
    "agent_iterations",
    "retrieved_segments",
    "cache_hits",
)


def _run_dir() -> Optional[Path]:
    value = os.environ.get("EXPERIMENT_METRICS_DIR")
    return Path(value).resolve() if value else None


def _append(path: Path, record: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        # Telemetry must never make an evaluation fail.
        return


def mark_phase(phase: str) -> None:
    run_dir = _run_dir()
    if run_dir is None or int(os.environ.get("RANK", "0")) != 0:
        return
    _append(run_dir / "events.jsonl", {"type": "phase", "phase": phase, "timestamp": time.time()})


def log_workload(
    *,
    task_name: str,
    doc_id: Any,
    workload: Optional[dict[str, Any]] = None,
    token_counts: Any = None,
) -> None:
    global _WORKLOAD_SAMPLES
    run_dir = _run_dir()
    record: dict[str, Any] = {
        "timestamp": time.time(),
        "rank": int(os.environ.get("RANK", "0")),
        "pid": os.getpid(),
        "task_name": task_name,
        "doc_id": doc_id,
    }
    if workload:
        record.update(workload)
    if token_counts is not None:
        counts = token_counts.to_dict() if hasattr(token_counts, "to_dict") else dict(token_counts)
        for key, value in counts.items():
            record.setdefault(key, value)
    _WORKLOAD_SAMPLES += 1
    task_totals = _WORKLOAD_BY_TASK.setdefault(task_name, {"samples": 0.0})
    task_totals["samples"] += 1
    for key in _ADDITIVE_FIELDS:
        value = record.get(key)
        if isinstance(value, (int, float)):
            _WORKLOAD_TOTALS[key] = _WORKLOAD_TOTALS.get(key, 0.0) + float(value)
            task_totals[key] = task_totals.get(key, 0.0) + float(value)
    selection_mode = record.get("selection_mode")
    if selection_mode:
        key = str(selection_mode)
        _WORKLOAD_SELECTION_MODES[key] = _WORKLOAD_SELECTION_MODES.get(key, 0) + 1
    if run_dir is not None:
        _append(run_dir / "workload_samples.jsonl", record)


def reset_workload_metrics() -> None:
    global _WORKLOAD_SAMPLES
    _WORKLOAD_TOTALS.clear()
    _WORKLOAD_BY_TASK.clear()
    _WORKLOAD_SELECTION_MODES.clear()
    _WORKLOAD_SAMPLES = 0


def summarize_workload_metrics() -> dict[str, Any]:
    return {
        "samples": _WORKLOAD_SAMPLES,
        "totals": dict(_WORKLOAD_TOTALS),
        "by_task": {task: dict(values) for task, values in _WORKLOAD_BY_TASK.items()},
        "selection_modes": dict(_WORKLOAD_SELECTION_MODES),
    }


def aggregate_workload_metrics(local_summary: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch.distributed as dist
    except ImportError:
        return local_summary
    if not dist.is_available() or not dist.is_initialized():
        return local_summary
    gathered: list[Optional[dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_summary)
    summaries = [summary for summary in gathered if summary]
    merged: dict[str, Any] = {"samples": 0, "totals": {}, "by_task": {}, "selection_modes": {}}
    for summary in summaries:
        merged["samples"] += int(summary.get("samples", 0))
        for key, value in (summary.get("totals") or {}).items():
            merged["totals"][key] = merged["totals"].get(key, 0.0) + float(value)
        for task, values in (summary.get("by_task") or {}).items():
            task_totals = merged["by_task"].setdefault(task, {})
            for key, value in values.items():
                task_totals[key] = task_totals.get(key, 0.0) + float(value)
        for mode, value in (summary.get("selection_modes") or {}).items():
            merged["selection_modes"][mode] = merged["selection_modes"].get(mode, 0) + int(value)
    return merged


def resource_metrics_link() -> Optional[dict[str, str]]:
    run_dir = _run_dir()
    if run_dir is None:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": os.environ.get("EXPERIMENT_METRICS_RUN_ID", run_dir.name),
        "summary_path": os.environ.get("EXPERIMENT_METRICS_SUMMARY", str(run_dir / "summary.json")),
    }


def record_result_path(path: str | Path) -> None:
    run_dir = _run_dir()
    if run_dir is None:
        return
    _append(
        run_dir / "events.jsonl",
        {"type": "result", "path": str(Path(path).resolve()), "timestamp": time.time()},
    )
