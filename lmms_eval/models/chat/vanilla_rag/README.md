# Vanilla video RAG

The `vanilla_rag` lmms-eval backend splits a video into nonoverlapping chunks,
selects evidence, and sends the selected clips in chronological order to a base
VLM for one answer. Recommended sampling parameter presets are included for
`Qwen/Qwen3.5-9B` and `Qwen/Qwen3.8-27B`.

- **Dense:** embed chunks and the question with `google/embeddinggemma-2`, then
  select the top-k chunks by cosine similarity. Video embeddings are persisted.
- **Uniform:** select evenly spaced chunks without loading the embedding model.
  The chunk size, evidence sampling, and answering model are configurable.

Tasks must supply exactly one local video per single-turn question and a nonempty
question field (default: `question`), which is used for retrieval. Task text,
including subtitles and answer choices, is preserved.

## Environment setup

Use Linux x86-64, Python 3.12, `uv`, and an NVIDIA driver compatible with CUDA
12.9. From the lmms-eval checkout containing this package, create separate serving
and retrieval environments:

```bash
# Serving: use the CUDA 12.9 vLLM wheel explicitly.
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python --torch-backend=cu129 -e . \
  'https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  'torch==2.13.0' 'torchvision==0.28.0' 'numpy==2.3.5' \
  'transformers==5.16.1' 'sentence-transformers==6.0.1'

# Retrieval and evaluation: newer EmbeddingGemma 2 support.
uv venv --python 3.12 .venv-vanilla-rag
uv pip install --python .venv-vanilla-rag/bin/python --torch-backend=cu129 -e . \
  'torch==2.13.0' 'torchvision==0.28.0' 'numpy==2.3.5' \
  'transformers==5.19.0' 'sentence-transformers==6.1.0'

# Dependency check: will not check model weights or datasets.
.venv-vanilla-rag/bin/python \
  -m lmms_eval.models.chat.vanilla_rag.runtime check --selection dense
```

The separate environments retain the tested vLLM dependencies while satisfying
the newer retrieval requirements checked by [encoder.py](encoder.py). Video
decoding uses PyAV. Ensure model and dataset access is available; set `HF_HOME`
if you want both processes to share a model cache.

## Launch and configure

Start the VLM on a GPU (like GPU 0) using [vllm_server.py](vllm_server.py).
Keep this process running while evaluating in another terminal on the same node:

```bash
# --model can also be set to Qwen/Qwen3.8-27B
CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/python \
  -m lmms_eval.models.chat.vanilla_rag.vllm_server \
  --model Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --seed 42 \
  --limit-mm-per-prompt '{"image":0,"video":10}' \
  --media-io-kwargs '{"video":{"num_frames":-1,"video_backend":"vanilla_rag_clip"}}' \
  --mm-processor-kwargs '{"do_sample_frames":false}'
```

This entry point registers [native_video.py](native_video.py)'s `vanilla_rag_clip`
loader, which preserves the supplied frames and source timestamps. Use it for
both dense and uniform selection. The server's video limit must be at least
`rag_top_k`; adjust the port, context length, model, and GPU allocation as needed.

Once the server is ready, run five examples with retrieval on another GPU (like GPU 1):

```bash
# change model=Qwen/Qwen3.5-9B to model=Qwen/Qwen3.8-27B if used in the server
CUDA_VISIBLE_DEVICES=1 .venv-vanilla-rag/bin/python -m lmms_eval \
  --model vanilla_rag \
  --model_args 'model=Qwen/Qwen3.5-9B,base_url=http://127.0.0.1:8000/v1,api_key=EMPTY,rag_selection=dense,rag_embed_device=cuda:0,rag_trace_path=./results/retrieval.jsonl' \
  --tasks videomme_long_w_subtitle \
  --limit 5 \
  --log_samples \
  --output_path ./results
```

Here `cuda:0` is the evaluator's local device index after `CUDA_VISIBLE_DEVICES`
exposes physical GPU 1. Use `rag_selection=uniform` for the uniform baseline,
which does not load an embedder; alternatively, use `rag_embed_device=cpu` for
dense retrieval on CPU. Both alternatives avoids the need for additional GPU, and 
you can change the evaluator's prefix to `CUDA_VISIBLE_DEVICES=''` in those cases.
Change the model name in both commands to use `Qwen/Qwen3.8-27B`.

Configure the backend through comma-separated `--model_args` entries. These are
the package defaults from [model.py](model.py):

| Argument | Default / meaning |
| --- | --- |
| `model`, `enable_thinking` | `Qwen/Qwen3.5-9B`, `True` |
| `base_url`, `api_key` | Set explicitly for your server, as in the example |
| `rag_selection` | `dense`; also accepts `uniform` |
| `rag_chunk_seconds`, `rag_top_k` | `10` seconds per chunk, up to `10` selected chunks |
| `rag_embed_model` | `google/embeddinggemma-2`; must be an EmbeddingGemma 2 checkpoint |
| `rag_embed_device` | `cuda:0`; also supports `cpu` |
| `rag_embed_fps`, `rag_embed_max_frames` | `1` fps, at most `32` frames per indexed chunk (maximum allowed: `56`) |
| `rag_embed_batch_size` | `4` chunks per embedding batch |
| `rag_clip_fps`, `rag_clip_longest_edge` | Evidence at `1` fps, longest edge `448` pixels |
| `rag_query_field` | `question`, the document field used for retrieval |
| `rag_max_new_tokens`, `rag_seed` | `16384`, `42`; generation presets are in [messages.py](messages.py) |
| `rag_cache_dir` | `.vanilla_rag_cache`, relative to the evaluator's working directory |
| `rag_trace_path` | Unset; provide a JSONL path to record retrieval, timing, answers, and token usage |
| `rag_embed_revision` | Unset; optionally pin the embedding checkpoint to a commit hash |
| `rag_vlm_revision` | Unset; trace metadata only. Pin the actual server checkpoint with `--revision` |

Dense indexes are reused across questions and answering models when the source,
embedding checkpoint, and indexing configuration match. Traces record selected
intervals, scores, timestamps, cache hits, and generation settings. To resolve a
model revision before pinning a run, use [runtime.py](runtime.py):
`python -m lmms_eval.models.chat.vanilla_rag.runtime revision MODEL REVISION`.
