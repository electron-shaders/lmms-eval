"""InfiniBench task adapters; scoring follows the authors' evaluation/eval_script.py."""

import ast
import json
import os
from collections import defaultdict
from pathlib import Path

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.infinibench import judge
from lmms_eval.tasks.infinibench.media import ensure_split_media

MCQ_SKILLS = ("character_actions", "scene_transitions", "choronological_understanding", "global_appearance")
OPEN_ENDED_SKILLS = ("summarization", "spoiler_questions", "deep_context_understanding", "linking_multiple_events")


def _select_subset(dataset, subset, skills):
    # Filter by skill, not options/answers: both are withheld on the test split.
    selected = dataset.filter(lambda skill: skill in skills, input_columns=["skill_name"])
    # Carry the variant into exports so MCQ, QA, and combined runs do not
    # overwrite one another when they share an output directory.
    return selected.add_column("infinibench_subset", [subset] * len(selected))


def infinibench_process_docs_mcq(dataset):
    return _select_subset(dataset, "mcq", MCQ_SKILLS)


def infinibench_process_docs_qa(dataset):
    return _select_subset(dataset, "qa", OPEN_ENDED_SKILLS)


def _skill(doc):
    skill = doc["skill_name"]
    if skill not in MCQ_SKILLS + OPEN_ENDED_SKILLS:
        raise ValueError(f"Unknown InfiniBench skill: {skill!r}")
    return skill


def _split(doc):
    split = doc["split"]
    if split in ("dev", "val", "validation"):
        return "validation"
    if split in ("train", "test"):
        return split
    raise ValueError(f"Unknown InfiniBench split: {split!r}")


def _options(doc):
    options = doc.get("options")
    if options is None or (isinstance(options, str) and options.strip() in ("", "None", "null")):
        return []
    if isinstance(options, str):
        # The published JSON stores Python literals, including nested action lists.
        options = ast.literal_eval(options)
    if not isinstance(options, (list, tuple)):
        raise ValueError(f"InfiniBench options must be a list (question {doc.get('question_id')})")
    return list(options)


def _resolve_media(doc, field, lmms_eval_specific_kwargs=None):
    value = doc.get(field)
    if not value:
        raise ValueError(f"InfiniBench question {doc.get('question_id')} has no {field}")
    relative = Path(value).expanduser()
    if relative.is_absolute():
        candidates = [relative]
    else:
        kwargs = lmms_eval_specific_kwargs or {}
        default_root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "infinibench"
        root = Path(kwargs.get("data_dir") or os.environ.get("INFINIBENCH_DATA_DIR") or default_root).expanduser()
        split = _split(doc)
        # Archives contain TV_shows/ and Movies/ directly. Support either a
        # shared extraction directory or one extraction directory per split.
        candidates = [root / split / relative, root / relative]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    if not relative.is_absolute():
        downloaded = ensure_split_media(split, root, required_path=relative) / relative
        if downloaded.is_file():
            return str(downloaded.resolve())
    locations = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"InfiniBench {field} not found; tried {locations}. Check INFINIBENCH_DATA_DIR and the split's extracted archive (see tasks/infinibench/README.md).")


def infinibench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return [_resolve_media(doc, "video_path_mp4", lmms_eval_specific_kwargs)]


def infinibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    skill = _skill(doc)
    parts = []
    use_subtitles = os.environ.get("INFINIBENCH_USE_SUBTITLES", kwargs.get("use_subtitles", True))
    if isinstance(use_subtitles, str):
        if use_subtitles.lower() not in ("true", "false", "1", "0"):
            raise ValueError("INFINIBENCH_USE_SUBTITLES must be true, false, 1, or 0")
        use_subtitles = use_subtitles.lower() in ("true", "1")
    if use_subtitles:
        subtitles_path = _resolve_media(doc, "video_subtitles", kwargs)
        subtitles = Path(subtitles_path).read_text(encoding="utf-8-sig")
        parts.append(f"Subtitles:\n{subtitles.strip()}")
    parts.append(f"Question: {doc['question']}")
    options = _options(doc) if skill in MCQ_SKILLS else []
    if options:
        # Keep list order: it is the distinction between several MCQ choices.
        choices = [option if isinstance(option, str) else json.dumps(option, ensure_ascii=False) for option in options]
        parts.append("Options:\n" + "\n".join(f"{index}. {option}" for index, option in enumerate(choices)))
        parts.append("Answer with only the zero-based option index.")
    else:
        if skill in MCQ_SKILLS and _split(doc) != "test":
            raise ValueError(f"Missing MCQ options for InfiniBench question {doc.get('question_id')}")
        # All public test choices are withheld, even for the four MCQ skills.
        parts.append("Answer the question based on the video" + (" and subtitles." if use_subtitles else "."))
    return kwargs.get("pre_prompt", "") + "\n\n".join(parts) + kwargs.get("post_prompt", "")


def infinibench_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": infinibench_doc_to_visual(doc, lmms_eval_specific_kwargs)[0]},
                {"type": "text", "text": infinibench_doc_to_text(doc, lmms_eval_specific_kwargs)},
            ],
        }
    ]


def infinibench_doc_to_target(doc):
    if _split(doc) == "test":
        return ""
    answer = doc.get("answer_idx") if _skill(doc) in MCQ_SKILLS else doc.get("answer")
    return "" if answer is None else str(answer)


def _prediction_record(doc, results):
    _skill(doc)
    raw_prediction = str(results[0]) if results and results[0] is not None else ""
    record = {key: doc[key] for key in ("question_id", "skill_name", "split", "video_path_mp4", "question")}
    if "infinibench_subset" in doc:
        record["infinibench_subset"] = doc["infinibench_subset"]
    # Only strip surrounding whitespace. The official evaluator compares
    # str(pred) == str(answer_idx); it does not guess letters or random choices.
    record.update(pred=raw_prediction.strip() if _skill(doc) in MCQ_SKILLS and _options(doc) else raw_prediction, raw_pred=raw_prediction)
    return record


def infinibench_process_results(doc, results):
    if _split(doc) == "test":
        raise ValueError("Use infinibench_test for the hidden-label test split")
    skill = _skill(doc)
    record = _prediction_record(doc, results)
    if skill in MCQ_SKILLS:
        if doc.get("answer_idx") is None:
            raise ValueError(f"Missing InfiniBench answer_idx for question {doc.get('question_id')}")
        record["answer_idx"] = doc["answer_idx"]
        record["mcq_correct"] = int(str(record["pred"]) == str(record["answer_idx"]))
        metrics = {"mcq_accuracy": record, f"{skill}_accuracy": record}
    else:
        if doc.get("answer") is None:
            raise ValueError(f"Missing InfiniBench answer for question {doc.get('question_id')}")
        record["answer"] = doc["answer"]
        record.update(judge.score_open_ended(doc["question"], doc["answer"], record["pred"]))
        metrics = {"gpt_score": record, f"{skill}_gpt_score": record, "judge_success_rate": float(record["gpt_score"] is not None)}
    metrics["submission_count"] = record
    return metrics


def infinibench_process_results_test(doc, results):
    # No judging or fabricated accuracy on hidden test labels.
    return {"submission_count": _prediction_record(doc, results)}


def infinibench_aggregate_mcq(results):
    by_skill = defaultdict(list)
    for result in results:
        by_skill[result["skill_name"]].append(result["mcq_correct"])
    return sum(sum(values) / len(values) for values in by_skill.values()) / len(by_skill) if by_skill else 0.0


def infinibench_aggregate_gpt(results):
    by_skill = defaultdict(list)
    for result in results:
        by_skill[result["skill_name"]].append(result["gpt_score"])
    skill_means = []
    for values in by_skill.values():
        valid = [value for value in values if value is not None]
        # Match eval_open_ended_skills: exclude failed judgments, but keep an
        # entirely failed skill with average_score=0 in the macro average.
        skill_means.append(sum(valid) / len(valid) if valid else 0.0)
    return sum(skill_means) / len(skill_means) if skill_means else 0.0


def infinibench_aggregate_submissions(results, args):
    by_split_skill = defaultdict(list)
    for result in results:
        subset = result.get("infinibench_subset", "")
        if subset not in ("", "mcq", "qa"):
            raise ValueError(f"Unknown InfiniBench subset: {subset!r}")
        by_split_skill[(_split(result), subset, _skill(result))].append(result)
    for split, subset in sorted({(split, subset) for split, subset, _ in by_split_skill}):
        task_suffix = "val" if split == "validation" else split
        if subset:
            task_suffix += f"_{subset}"
        skills = MCQ_SKILLS if subset == "mcq" else OPEN_ENDED_SKILLS if subset == "qa" else MCQ_SKILLS + OPEN_ENDED_SKILLS
        for skill in skills:
            # Clear unobserved skills too, so a --limit run in a reused output
            # directory cannot mix current predictions with an earlier run.
            path = generate_submission_file(f"{skill}.json", args, subpath=f"submissions/infinibench_{task_suffix}")
            Path(path).write_text(json.dumps(by_split_skill[(split, subset, skill)], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(results)
