"""Postprocess final chat replies without treating paragraph breaks as EOS."""

from __future__ import annotations


def final_answer_text(content: str | None, generation: dict | None = None) -> str:
    answer = (content or "").strip()
    stops = (generation or {}).get("until", []) or []
    for stop in [stops] if isinstance(stops, str) else stops:
        # TaskConfig supplies the few-shot separator (usually a blank line) as
        # an implicit stop. A complete chat reply may put its answer after it.
        if stop and stop.strip():
            answer = answer.split(stop, 1)[0]
    return answer.strip()
