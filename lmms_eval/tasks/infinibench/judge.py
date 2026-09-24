"""GPT judging adapted from Vision-CAIR/Infinibench's BSD-licensed evaluator.

Reference: c4d4aa9c627a2af5fa1cfa9bf15f1f62b638d1da/evaluation/eval_script.py.
The original copyright notice and license are in LICENSE.md in this directory.
"""

import ast
import json
import os
import time
from functools import lru_cache

from loguru import logger as eval_logger

SYSTEM_PROMPT = (
    "You are an intelligent and fair evaluator AI that specializes in assessing the correctness and semantic alignment "
    "between ground truth answers and predicted responses for question-answering tasks, including those based on video content.\n\n"
    "Your role is to evaluate how well a predicted answer matches the correct (reference) answer based on the following detailed criteria:\n"
    "------\n"
    "## EVALUATION INSTRUCTIONS:\n"
    "- Focus on **semantic similarity**, **factual correctness**, and **completeness**.\n"
    "- Accept paraphrases, synonyms, or rephrasings **as valid**, as long as they preserve the original meaning.\n"
    "- **Do not penalize** for stylistic differences or changes in tone, unless they impact factual accuracy.\n"
    "- **Penalize** if:\n"
    "  - The predicted answer omits **key factual elements** present in the correct answer.\n"
    "  - The prediction includes **hallucinated content** or unfounded details.\n"
    "  - The prediction **contradicts** the correct answer.\n"
    "- Use human-like judgment: apply reasoning beyond surface text similarity.\n"
    "- When uncertain, provide a **conservative but fair** score.\n"
    "- Use a scoring scale from **0 (completely incorrect)** to **10 (perfect match)**.\n"
    "## OUTPUT FORMAT:\n"
    "Return a JSON object with **two fields**:\n"
    '- "score": an integer from 0 to 10\n'
    '- "justification": a concise explanation (1-3 sentences) of your reasoning\n\n'
    "### Example Output:\n"
    "{\n"
    '  "score": 7,\n'
    '  "justification": "The predicted answer captures the main idea, but it omits some key details about the setting described in the correct answer."\n'
    "}\n"
    "------\n"
    "Be fair, consistent, and concise. Follow the format exactly."
)


@lru_cache(maxsize=1)
def _get_client():
    # Task discovery, MCQ scoring, and test submissions need no API client.
    if not os.environ.get("OPENAI_API_KEY"):
        eval_logger.warning("InfiniBench open-ended judging requires OPENAI_API_KEY; judgments will be missing. Check judge_success_rate before interpreting gpt_score.")
        return None
    from openai import OpenAI

    # This module owns the retries; avoid multiplying them with SDK retries.
    return OpenAI(max_retries=0)


def _messages(question, answer, prediction):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {prediction}\n\n"
                "Please return your evaluation in the specified JSON format with both a score and a justification."
            ),
        },
    ]


def _parse_response(response):
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        # The authors use literal_eval; retain support for single-quoted dicts.
        result = ast.literal_eval(response)
    if not isinstance(result, dict) or "score" not in result or "justification" not in result:
        raise ValueError("Judge must return score and justification")
    if type(result["score"]) is not int or not 0 <= result["score"] <= 10:
        raise ValueError("Judge score must be an integer from 0 to 10")
    if not isinstance(result["justification"], str):
        raise ValueError("Judge justification must be text")
    return {"gpt_score": result["score"], "gpt_justification": result["justification"]}


def score_open_ended(question, answer, prediction, max_retries=3, retry_delay=1.0):
    client = _get_client()
    if client is None:
        return {"gpt_score": None, "gpt_justification": "OpenAI client not available"}
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=_messages(question, answer, prediction),
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=500,
                timeout=30,
            )
            return _parse_response(completion.choices[0].message.content)
        except Exception as exc:
            retryable = any(term in str(exc).lower() for term in ("connection error", "timeout", "rate limit", "server error", "503", "502", "500", "429", "network", "connection"))
            if retryable and attempt < max_retries:
                time.sleep(retry_delay * (2**attempt) + (time.time() % 1))
            else:
                eval_logger.warning(f"InfiniBench judgment failed: {exc}")
                return {"gpt_score": None, "gpt_justification": f"Evaluation failed: {exc}"}
