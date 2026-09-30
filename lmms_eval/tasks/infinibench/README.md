# InfiniBench

The combined tasks cover all eight skills in the official
[Vision-CAIR/InfiniBench dataset](https://huggingface.co/datasets/Vision-CAIR/InfiniBench):

| Task | Hugging Face split | Questions | Behavior |
| --- | --- | ---: | --- |
| `infinibench_train` | `train` | 76,475 | MCQ accuracy and open-ended judging |
| `infinibench_val` | `validation` | 5,137 | MCQ accuracy and open-ended judging |
| `infinibench_test` | `test` | 6,006 | Prediction export; no local scoring |

Append `_mcq` or `_qa` to select only the four MCQ or four free-response skills:

| Split | MCQ task | Questions | QA task | Questions |
| --- | --- | ---: | --- | ---: |
| Train | `infinibench_train_mcq` | 36,233 | `infinibench_train_qa` | 40,242 |
| Validation | `infinibench_val_mcq` | 2,330 | `infinibench_val_qa` | 2,807 |
| Test | `infinibench_test_mcq` | 3,336 | `infinibench_test_qa` | 2,670 |

Filtering uses the official skill categories and occurs before `--limit` is
applied. MCQ tasks never invoke the GPT judge. QA tasks use the free-response
judge on train/validation; both test variants only export predictions. The
`_mcq` test variant selects the MCQ **skills**, whose choices are withheld in
this public release, so its prompts still request free-text answers.

Annotations and media are pinned to dataset revision
`1a46b0b303303515fa5872e3704c8b27a8ab7b0a`. Each task downloads only its own
split's annotation files. Validation rows use the authors' `split: dev` label.

## Media downloads

When a required video or subtitle is missing, the task automatically downloads
that split's multipart video archive from the official Hugging Face repository:

- [Train media](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/tree/main/train)
- [Validation media](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/tree/main/validation)
- [Test media](https://huggingface.co/datasets/Vision-CAIR/InfiniBench/tree/main/test)

The release stores media in archives rather than individually downloadable
videos. The first media access therefore fetches the **whole requested split**
(approximately 92 GB for train, 12 GB for validation, or 12 GB for test).
`--limit` limits evaluated questions, not the archive download size. Downloaded
archives remain in the Hugging Face cache; extracted media require additional
disk space. Extraction streams the parts without making a concatenated archive.
A file lock coordinates concurrent workers and completed downloads/extractions
are reused.

Extracted files default to `$HF_HOME/infinibench/<split>/`, or
`~/.cache/huggingface/infinibench/<split>/` when `HF_HOME` is unset. Override the
extraction root with `INFINIBENCH_DATA_DIR`:

```text
$INFINIBENCH_DATA_DIR/
  validation/
    TV_shows/videos/castle/season_1/episode_2.mp4
    TV_shows/subtitles/castle/season_1/episode_2.srt
    Movies/videos/...
    Movies/subtitles/...
  train/...
  test/...
```

Existing media are used before attempting downloads. A shared extraction layout
with `TV_shows/` and `Movies/` directly under `INFINIBENCH_DATA_DIR` also works.

## Run

Use any lmms-eval video model, for example:

```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks infinibench_val \
  --batch_size 1 \
  --limit 5 \
  --output_path ./results/infinibench_val \
  --log_samples
```

Replace the task name with `infinibench_train` or `infinibench_test` to select
another split. Remove `--limit 5` for a complete run. Both chat
(`doc_to_messages`) and simple (`doc_to_visual`/`doc_to_text`) model interfaces
are supported. Frame sampling is controlled by the chosen model backend.

For example, use `--tasks infinibench_val_mcq` for only MCQs, or
`--tasks infinibench_val_qa` for only free-response questions. Variants share the
same media cache; selecting a subset still downloads the split's whole archive
on first media access.

Subtitles are included by default, matching the input modalities reported on the
authors' leaderboard. The entire video and SRT text are used; ground-truth
temporal intervals are never used to select clips. Set
`INFINIBENCH_USE_SUBTITLES=0` for a video-only ablation. Task-specific kwargs
also support `use_subtitles`, `data_dir`, `pre_prompt`, and `post_prompt`.

Open-ended scoring uses the standard `OPENAI_API_KEY` environment variable and
the OpenAI SDK's optional `OPENAI_BASE_URL`. The judge is initialized lazily;
task discovery, MCQ scoring, and the test task require no API client. Missing
credentials or exhausted judge requests produce missing judgments, which are
reported through `judge_success_rate` and each exported `gpt_justification`.
Check this rate before interpreting `gpt_score`.

## Evaluation protocol

The reference implementation is the authors'
[evaluation/eval_script.py at c4d4aa9](https://github.com/Vision-CAIR/Infinibench/blob/c4d4aa9c627a2af5fa1cfa9bf15f1f62b638d1da/evaluation/eval_script.py).
Its judge prompts are preserved verbatim, along with `gpt-4o-mini`, temperature
0.1, JSON output, 500 output tokens, and a 30-second request timeout.

- **MCQ skills:** `character_actions`, `scene_transitions`,
  `choronological_understanding`, and `global_appearance`. The upstream spelling
  `choronological_understanding` is intentional. Responses are compared with the
  zero-based `answer_idx` using string equality, as in `mcq_accuracy`.
- **Open-ended skills:** `summarization`, `spoiler_questions`,
  `deep_context_understanding`, and `linking_multiple_events`. Scores are integers
  from 0 to 10 with a textual justification.
- **`mcq_accuracy` (0–1):** mean of the MCQ skill accuracies, with equal weight per
  skill. Invalid/empty model responses count as incorrect.
- **`gpt_score` (0–10):** mean of the open-ended skill means, with equal weight per
  skill. Failed judgments are omitted within a skill; a skill with no successful
  judgments contributes zero, following the reference evaluator.
- **Per-skill metrics** and **`judge_success_rate` (0–1)** are also reported.
  Limited runs average only the skills actually encountered. No combined
  MCQ/GPT metric is invented; the reference script reports them separately.

The official evaluation directory specifies scoring, not a model inference
template. This adapter labels choices with zero-based indices and asks for the
index only. Nested options are displayed in their original order. Surrounding
whitespace is stripped from MCQ predictions; letters and explanatory text are
not guessed into indices. Open-ended responses are passed to the judge unchanged
after lmms-eval's standard filters/reasoning-tag handling.

Transport differences from the reference are limited to lazy client creation,
explicit task-owned retries (three retries with exponential backoff), and JSON
parsing with a Python-literal fallback. Boolean scores and non-textual
justifications are rejected rather than accepted as valid judgments.

## Prediction files and the test split

Every task writes one JSON list per skill to
`<output_path>/submissions/<task_name>/<skill_name>.json`. Rows preserve
`question_id`, `skill_name`, `split`, `video_path_mp4`, `question`, `pred`, and
`raw_pred`. Scored splits additionally include the reference answer/index and
the score/justification. Skills absent from a limited run receive empty lists,
so reusing an output directory does not retain predictions from an earlier run.
Subset tasks write only their four skills under their own task directory, such
as `submissions/infinibench_val_mcq/` and `submissions/infinibench_val_qa/`, so
combined and subset runs can share an output root without overwriting exports.
These lists match the authors' local evaluation input
format, and can be passed as `--pred_dir` to their evaluator. `submission_count`
is the number of exported predictions, not a performance score. Without
`--output_path`, files are placed under `./submissions/`.

The requested public **test** release withholds all answers **and all MCQ
options**. Consequently `infinibench_test` generates free-text answers for every
skill and exports them without judging or fabricating MCQ scores. These exports
are not a ready-made Codabench submission: the
[2025 challenge](https://www.codabench.org/competitions/10065/) uses separate
challenge annotations, integer MCQ predictions, different open-ended filenames,
and JSON dictionaries inside a ZIP. This task intentionally uses the requested
`Vision-CAIR/InfiniBench` release and does not substitute challenge data.

The adapted judging code retains the upstream BSD 3-Clause license in
[LICENSE.md](LICENSE.md).
