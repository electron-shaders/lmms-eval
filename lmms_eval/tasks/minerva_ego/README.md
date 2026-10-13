# MINERVA-Ego

Five-choice video QA using the [MINERVA-Ego JSON](https://github.com/google-deepmind/neptune/raw/refs/heads/main/minerva-ego_20260205.json) and local prepared HD-EPIC videos. The task loads that JSON by default and inherits MINERVA's prompt, target extraction, answer parser, and generation settings (32 new tokens, temperature 0, no sampling).

## Local videos

Set `MINERVA_EGO_VIDEO_DIR` to your mounted `Videos` directory:

```bash
export MINERVA_EGO_VIDEO_DIR="/absolute/path/to/HD-EPIC/Videos"
```

Expected layout:

```text
Videos/
  P01/
    P01-20240202-110250.mp4
  P02/
    P02-20240203-120000.mp4
```

Flat layouts (`Videos/P01-20240202-110250.mp4`) also work. Participant folders take precedence over flat files; within each folder, extension priority is `mp4`, `webm`, `mkv`, `mov`. User (`~`) and environment variables in the directory are expanded, and returned paths are absolute. Videos will not be downloaded and must be prepared locally.

Both simple and chat models receive the full video path and the question with choices A–E. Chat content places the video before the shared MINERVA prompt. Frame sampling is controlled by the model backend. Reasoning traces, objects of interest, and referenced object masks remain annotation metadata and are excluded from model inputs.

## Local annotation JSON

If you prefer using a predownloaded annotation JSON, use the standard custom-task YAML override. Save the following as `/path/to/custom_tasks/minerva_ego_local.yaml`, replacing the include and annotation paths with absolute paths:

```yaml
include: /path/to/lmms-eval/lmms_eval/tasks/minerva_ego/minerva_ego.yaml
task: minerva_ego_local
dataset_kwargs:
  data_files:
    test: /path/to/minerva-ego_20260205.json
```

Run with `--include_path /path/to/custom_tasks --tasks minerva_ego_local`. This replaces the remote annotation source while preserving the inherited task settings. Note that your custom `task` name (e.g., `minerva_ego_local`) replaces `minerva_ego` in the JSON path.

## Scoring

Each question contributes once to the overall score and once to the score for each distinct label in `question_type`. `minerva_ego_acc` is the overall accuracy on a 0–100 scale, and `minerva_ego_question_type_acc` is the accuracy for each question type also on a 0–100 scale. Absent or empty labels use `UNKNOWN` as fallback. Empty overall aggregation returns zero; empty aggregation by question type returns an empty dictionary. Records retain the question `key` as `id`, question type labels, predicted and ground truth letters, and a 0/1 score.

With `--output_path`, the results JSON records overall accuracy under `results.minerva_ego["minerva_ego_acc,none"]` and accuracies by question type under `results.minerva_ego["minerva_ego_question_type_acc,none"]`. The latter maps each evaluated question type name to its accuracy on the 0–100 scale. Each question type's denominator includes only questions tagged with that question type, counting repeated labels once. These aggregate scores are saved without `--log_samples`; scores by question type are also logged to the console.

The shared MINERVA parser selects an answer letter. If it returns no valid choice, MINERVA-Ego awards zero credit, without a random-choice fallback.

## Five-sample smoke test

From the lmms-eval repository, with the videos mounted:

```bash
uv run python -m lmms_eval \
  --model dummy \
  --model_args "read_bytes=65536,response=A,allow_remote=false,fail_on_missing=true" \
  --tasks minerva_ego \
  --batch_size 1 \
  --limit 5 \
  --output_path ./logs/minerva_ego_dummy
```

This checks task execution and local byte reads, without model inference or video decoding. To check decoding with real videos, add `decode_num_frames=8` to `--model_args`.
