"""Bounded local-video probing and sampling for vanilla RAG."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class MediaInfo:
    duration_us: int
    fps: float
    width: int
    height: int


def probe(path: str | Path) -> MediaInfo:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Source contains no video stream")
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else container.duration / av.time_base if container.duration is not None else None
        if duration is None or duration <= 0:
            # Count PTS only, without accumulating decoded frames.
            origin = float((stream.start_time or 0) * stream.time_base)
            duration = 0.0
            for frame in container.decode(stream):
                if frame.pts is not None:
                    duration = max(duration, float(frame.pts * frame.time_base) - origin + 1 / float(stream.average_rate or 25))
        if duration <= 0:
            raise ValueError("Source contains no timestamped video frames")
        return MediaInfo(round(duration * 1e6), float(stream.average_rate or 25), stream.width, stream.height)


def validate_sampling(fps: float, max_frames: int | None, longest_edge: int):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    if isinstance(longest_edge, bool) or not isinstance(longest_edge, int) or longest_edge < 1:
        raise ValueError("longest_edge must be a positive integer")


def sample_targets(start_us: int, end_us: int, fps: float, max_frames: int | None):
    count = max(1, math.ceil((end_us - start_us) / 1e6 * fps))
    if max_frames is not None and count > max_frames:
        return np.linspace(start_us, end_us - 1, max_frames)
    return start_us + np.arange(count) * (1e6 / fps)


def _rgb(frame, longest_edge, size=None):
    if size is None:
        scale = min(1.0, longest_edge / max(frame.width, frame.height))
        size = max(1, round(frame.width * scale)), max(1, round(frame.height * scale))
    return frame.reformat(width=size[0], height=size[1], format="rgb24").to_ndarray(), size


def read_window(path: str | Path, start_us: int, end_us: int, *, fps: float = 1, max_frames: int | None = None, longest_edge: int = 448, targets=None):
    """Seek and sample a half-open range. Memory grows only with selected frames."""
    validate_sampling(fps, max_frames, longest_edge)
    targets = sample_targets(start_us, end_us, fps, max_frames) if targets is None else np.asarray(targets)
    if len(targets) == 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8), []
    images, times, size, index = [], [], None, 0
    previous = last_frame = None
    last_pts = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        container.seek(max(0, int((start_us / 1e6 + origin) / stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Cannot sample a frame without a timestamp")
            pts = round((float(frame.pts * frame.time_base) - origin) * 1e6)
            if pts >= end_us:
                break
            if pts < start_us:
                previous = frame
                continue
            last_frame, last_pts = frame, pts
            if pts >= targets[index]:
                rgb, size = _rgb(frame, longest_edge, size)
                images.append(rgb)
                times.append(pts / 1e6)
                while index < len(targets) and targets[index] <= pts:
                    index += 1
                if index == len(targets):
                    break
        # The final target may fall inside the last frame's display duration.
        if index < len(targets) and last_frame is not None and (not times or times[-1] != last_pts / 1e6):
            rgb, size = _rgb(last_frame, longest_edge, size)
            images.append(rgb)
            times.append(last_pts / 1e6)
        # A range shorter than one frame still has displayed visual content.
        if not images and previous is not None:
            rgb, _ = _rgb(previous, longest_edge)
            images.append(rgb)
            times.append(start_us / 1e6)
    if not images:
        raise ValueError("Requested interval contains no decodable frames")
    return np.stack(images), times
