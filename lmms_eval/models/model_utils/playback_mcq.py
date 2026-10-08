"""Conservative answer-format validation, using only the question and reply."""

from __future__ import annotations

import re


def requested_choices(messages) -> tuple[str, ...]:
    """Activate only for a prompt requesting an answer among labeled options."""
    texts = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        texts.append(content if isinstance(content, str) else "\n".join(p.get("text", "") for p in content if p.get("type") == "text"))
    text = "\n".join(texts)
    if not re.search(r"multiple[ -]choice|(?:answer|respond|reply)[^\n]{0,100}(?:letter|option)|the best answer is", text, re.I):
        return ()
    labels = sorted(set(re.findall(r"^\s*(?:\(([A-H])\)|([A-H])[.)])\s+", text, re.M)))
    choices = tuple(sorted({a or b for a, b in labels}))
    return choices if len(choices) >= 2 and choices == tuple("ABCDEFGH"[: len(choices)]) else ()


def explicit_choice(text: str, choices: tuple[str, ...]) -> str:
    """Accept a clear answer, not a letter mentioned while discussing options.

    This does not judge the meaning of an answer or compare it with a label.
    Ambiguous prose is left to a bounded format-repair turn in the wrapper.
    """
    letters = "".join(choices)
    if not letters:
        return ""
    clean = re.sub(r"[*_`]", "", text).strip()
    bare = re.fullmatch(rf"[\[(]?([{letters}])[\])]?[.!]?", clean, re.I)
    if bare:
        return bare[1].upper()
    prefix = r"(?:(?:the |my )?(?:(?:best|correct|final) )?answer(?: is)?" r"|(?:the |my )?(?:best|correct|final) (?:option|choice)(?: is)?" r"|my (?:option|choice)(?: is)?|(?:the )?(?:option|choice) is" r"|i (?:choose|select|pick))"
    # Only the phrase is case-insensitive: "the answer is a dog" must not
    # become option A merely because of the English indefinite article.
    declarations = list(re.finditer(rf"(?:^|\n)\s*(?i:{prefix})\s*[:=]?\s*[\[(]?([{letters}])[\])]?(?!\w)([^\n]*)", clean))
    if declarations:
        match = declarations[-1]
        if re.match(rf"\s*(?:[/,;]|\bor\b|\band\b)\s*(?:(?:or|and)\s+)?[\[(]?[{letters}](?!\w)", match[2], re.I):
            return ""
        return match[1].upper()
    # A single option line is a common final-answer format. A list of several
    # options is discussion, even if the benchmark's loose extractor picks one.
    option_lines = list(re.finditer(rf"^\s*(?:\(([{letters}])\)|([{letters}])[.):])\s+", clean, re.M))
    if len(option_lines) == 1 and option_lines[0].start() == 0:
        return (option_lines[0][1] or option_lines[0][2]).upper()
    return ""
