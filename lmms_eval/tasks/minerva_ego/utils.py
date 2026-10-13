import os
import re
from collections import defaultdict
from pathlib import Path

from lmms_eval.tasks.minerva import utils as minerva_utils
from lmms_eval.utils import eval_logger


def minerva_ego_doc_to_visual(doc):
    video_dir = os.getenv("MINERVA_EGO_VIDEO_DIR", "").strip()
    if not video_dir:
        raise ValueError("Set MINERVA_EGO_VIDEO_DIR to the local HD-EPIC Videos directory before evaluating MINERVA-Ego.")

    video_id = doc.get("video_id")
    if not isinstance(video_id, str) or not re.fullmatch(r"P\d+-\d{8}-\d{6}", video_id):
        raise ValueError(f"Invalid MINERVA-Ego video_id {video_id!r}; expected an HD-EPIC ID such as 'P01-20240202-110250' (without an extension).")

    root = Path(os.path.expandvars(video_dir)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MINERVA_EGO_VIDEO_DIR is not a directory: {root}. Mount the local HD-EPIC Videos directory and update this variable.")

    participant = video_id.split("-", 1)[0]
    candidates = [directory / f"{video_id}.{ext}" for directory in (root / participant, root) for ext in minerva_utils._VIDEO_EXTENSIONS]
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate.resolve())]
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"MINERVA-Ego video {video_id!r} was not found. Check MINERVA_EGO_VIDEO_DIR and mount the prepared HD-EPIC videos. Searched: {searched}")


def minerva_ego_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    video = minerva_ego_doc_to_visual(doc)[0]
    prompt = minerva_utils.minerva_doc_to_text(doc, lmms_eval_specific_kwargs)
    return [{"role": "user", "content": [{"type": "video", "url": video}, {"type": "text", "text": prompt}]}]


def _question_types(labels):
    if isinstance(labels, str):
        labels = [labels]
    return list(dict.fromkeys(label.strip() for label in (labels or []) if isinstance(label, str) and label.strip())) or ["UNKNOWN"]


def minerva_ego_process_results(doc, results):
    raw_pred = results[0] if isinstance(results, list) and results else ""
    pred_letter = minerva_utils._extract_choice_letter(raw_pred)
    gold_letter = minerva_utils.minerva_doc_to_target(doc)
    question_types = _question_types(doc.get("question_type"))
    score = float(pred_letter in minerva_utils.OPTIONS and pred_letter == gold_letter)
    return {
        "minerva_ego_acc": {
            "id": doc.get("key", ""),
            "question_type": question_types,
            "pred_answer": pred_letter,
            "answer": gold_letter,
            "score": score,
        },
        "minerva_ego_question_type_acc": {question_type: score for question_type in question_types},
    }


def minerva_ego_aggregate_results(results):
    total_correct = sum(result.get("score", 0.0) == 1.0 for result in results)
    overall = 100.0 * total_correct / len(results) if results else 0.0
    eval_logger.info(f"MINERVA-Ego - overall accuracy: {overall:.2f}%")
    return overall


def minerva_ego_aggregate_question_type_results(results):
    """Aggregate each question type only over questions tagged with that question type."""
    by_question_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        for question_type, score in result.items():
            by_question_type[question_type]["correct"] += int(score == 1.0)
            by_question_type[question_type]["total"] += 1

    accuracies = {question_type: 100.0 * stats["correct"] / stats["total"] for question_type, stats in sorted(by_question_type.items())}
    for question_type, accuracy in accuracies.items():
        eval_logger.info(f"MINERVA-Ego - question type [{question_type}] accuracy: {accuracy:.2f}%")
    return accuracies
