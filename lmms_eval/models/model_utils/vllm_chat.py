"""Batch native vLLM chat requests with matching template options."""

import json


def chat_with_template_groups(client, request_items, sampling_params_cls, *, chat_template=None):
    """Keep per-request thinking modes aligned with sampling and restore order.

    vLLM accepts per-request SamplingParams, but only one set of chat-template
    options per chat() call. Group after TP synchronization so every rank makes
    the same calls and receives outputs in the original merged request order.
    """
    groups = {}
    for index, (messages, params, options) in enumerate(request_items):
        key = json.dumps(options, sort_keys=True)
        groups.setdefault(key, []).append((index, messages, params, options))

    outputs = [None] * len(request_items)
    for group in groups.values():
        response = client.chat(
            messages=[messages for _, messages, _, _ in group],
            sampling_params=[sampling_params_cls(**params) for _, _, params, _ in group],
            chat_template=chat_template,
            **group[0][3],
        )
        if len(response) != len(group):
            raise RuntimeError(f"vLLM returned {len(response)} outputs for {len(group)} chat requests")
        for (index, _, _, _), output in zip(group, response):
            outputs[index] = output
    return outputs
