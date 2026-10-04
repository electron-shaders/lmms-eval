"""Official Qwen3.5-4B sampling for the general tasks used by these baselines.

Apply at the generation boundary: task YAML defaults (often greedy decoding)
must not replace the model's preset. Other models retain their own settings.
Source: https://huggingface.co/Qwen/Qwen3.5-4B#best-practices
"""

from __future__ import annotations

import copy
import json
import re


def is_qwen35_4b(model):
    return bool(re.search(r"qwen3[._-]?5[-_]4b(?:$|[^a-z0-9])", str(model).lower()))


def thinking_enabled(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1", "yes"):
        return True
    if str(value).lower() in ("false", "0", "no"):
        return False
    raise ValueError("enable_thinking must be a boolean")


def qwen35_sampling_params(model, *, enable_thinking=True):
    if not is_qwen35_4b(model):
        return {}
    thinking = thinking_enabled(enable_thinking)
    return {
        "temperature": 1.0 if thinking else 0.7,
        "top_p": 0.95 if thinking else 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }


def qwen35_generation_kwargs(model, generation, *, enable_thinking=True):
    result = copy.deepcopy(generation or {})
    if not is_qwen35_4b(model):
        return result
    extra = result.setdefault("extra_body", {})
    template = extra.setdefault("chat_template_kwargs", {})
    thinking = thinking_enabled(template.get("enable_thinking", result.pop("enable_thinking", enable_thinking)))
    template["enable_thinking"] = thinking
    preset = qwen35_sampling_params(model, enable_thinking=thinking)
    result.update(preset)
    result["do_sample"] = True
    # The SDK merges extra_body last. Remove stale duplicate overrides before
    # splitting standard API fields from vLLM-specific fields below.
    for name in preset:
        extra.pop(name, None)
    return result


def openai_generation_kwargs(model, generation, *, enable_thinking=True, max_tokens=1024):
    generation = qwen35_generation_kwargs(model, generation, enable_thinking=enable_thinking)
    extra = copy.deepcopy(generation.get("extra_body", {}))
    result = {
        "max_tokens": generation.get("max_new_tokens", generation.get("max_tokens", max_tokens)),
        "temperature": generation.get("temperature", 0),
    }
    for name in ("top_p", "presence_penalty", "frequency_penalty"):
        if generation.get(name) is not None:
            result[name] = generation[name]
    for name in ("top_k", "min_p", "repetition_penalty", "seed", "thinking_token_budget"):
        if name in generation:
            extra[name] = generation[name]
    if extra:
        result["extra_body"] = extra
    return result


class GeneratedTokenPresencePenalty:
    """Transformers equivalent of the API penalty on generated tokens only."""

    def __init__(self, penalty):
        self.penalty = penalty
        self.prompt_length = None

    def __call__(self, input_ids, scores):
        import torch

        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[1]
        generated = input_ids[:, self.prompt_length :]
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated, True)
        return scores - seen.to(scores.dtype) * self.penalty


if __name__ == "__main__":
    import sys

    # Invoked directly by the shell launcher without importing model packages.
    params = qwen35_sampling_params(sys.argv[1], enable_thinking=sys.argv[2])
    if params:
        params.update(do_sample=True, enable_thinking=thinking_enabled(sys.argv[2]))
    print(",".join(f"{key}={json.dumps(value)}" for key, value in params.items()))
