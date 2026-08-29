"""Reliable evaluator-side video backends for qwen-vl-utils."""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
import torch
from loguru import logger as eval_logger


def _opencv_video_reader(ele: dict[str, Any], qwen_vp: Any):
    import cv2

    video_path = os.fspath(ele["video"])
    if video_path.startswith("file://"):
        video_path = video_path[7:]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        if total_frames <= 0:
            raise RuntimeError(f"OpenCV got invalid total_frames {total_frames} for {video_path}")

        start_frame, end_frame, total_frames = qwen_vp.calculate_video_frame_range(
            ele,
            total_frames,
            video_fps,
        )
        nframes = qwen_vp.smart_nframes(
            ele,
            total_frames=total_frames,
            video_fps=video_fps,
        )
        indices = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()

        frames: list[np.ndarray] = []
        last_frame: np.ndarray | None = None
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if ok:
                last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif last_frame is None:
                raise RuntimeError(f"OpenCV could not decode frame {frame_index} from {video_path}")
            # Broken containers commonly over-report their final frame. Reuse
            # the last valid frame so the requested tensor length stays stable.
            frames.append(last_frame.copy())
    finally:
        cap.release()

    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video_metadata = {
        "fps": video_fps,
        "frames_indices": indices,
        "total_num_frames": total_frames,
        "video_backend": "opencv",
    }
    return video, video_metadata, sample_fps


def patch_qwen_vl_utils() -> bool:
    """Use Decord with an OpenCV fallback and bypass torchvision video I/O."""
    try:
        import qwen_vl_utils.vision_process as qwen_vp
    except ImportError:
        return False

    if getattr(qwen_vp, "_lmms_eval_video_reader_patched", False):
        return True

    failed_decord_paths: set[str] = set()
    failed_paths_lock = threading.Lock()

    def read_video_opencv(ele: dict[str, Any]):
        return _opencv_video_reader(ele, qwen_vp)

    original_decord_reader = getattr(qwen_vp, "_read_video_decord", None)

    def read_video_decord_with_fallback(ele: dict[str, Any]):
        video_path = os.fspath(ele["video"])
        with failed_paths_lock:
            use_opencv = video_path in failed_decord_paths
        if use_opencv or original_decord_reader is None:
            return read_video_opencv(ele)
        try:
            return original_decord_reader(ele)
        except Exception as exc:
            with failed_paths_lock:
                failed_decord_paths.add(video_path)
            eval_logger.warning(
                "Decord failed for {}; retrying with OpenCV: {}",
                video_path,
                exc,
            )
            return read_video_opencv(ele)

    qwen_vp.VIDEO_READER_BACKENDS["opencv"] = read_video_opencv
    qwen_vp.VIDEO_READER_BACKENDS["torchvision"] = read_video_opencv

    if original_decord_reader is not None and qwen_vp.is_decord_available():
        primary_backend = "decord"
        primary_reader = read_video_decord_with_fallback
        qwen_vp.VIDEO_READER_BACKENDS["decord"] = primary_reader
    else:
        primary_backend = "opencv"
        primary_reader = read_video_opencv

    # TorchCodec can appear importable while its FFmpeg-linked shared library
    # is unusable. Route both auto-selection and explicit TorchCodec selection
    # through the reliable evaluator backend.
    qwen_vp.VIDEO_READER_BACKENDS["torchcodec"] = primary_reader
    if hasattr(qwen_vp.get_video_reader_backend, "cache_clear"):
        qwen_vp.get_video_reader_backend.cache_clear()
    qwen_vp.get_video_reader_backend = lambda: primary_backend
    qwen_vp._lmms_eval_video_reader_patched = True
    eval_logger.info(
        "Patched qwen-vl-utils video reader: primary={}, fallback=opencv",
        primary_backend,
    )
    return True
