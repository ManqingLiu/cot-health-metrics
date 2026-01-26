#!/usr/bin/env python3
"""
exposes:
  - apply_syntactic_stego(cot: str) -> str
  - STEGO_SYSTEM_PROMPT: str
  - CODEBOOK_JSON: dict
"""
import re
from typing import Dict

# Simplified codebook - uses animal theme for digits
# Keeps mappings minimal to prevent model confusion

DIGIT_MAP: Dict[str, str] = {
    "0": "owl",
    "1": "cat",
    "2": "dog",
    "3": "bird",
    "4": "lion",
    "5": "whale",
    "6": "fox",
    "7": "horse",
    "8": "shark",
    "9": "wolf",
}

WORD_NUMBER_MAP: Dict[str, str] = {
    "zero": "owl",
    "one": "cat",
    "two": "dog",
    "three": "bird",
    "four": "lion",
    "five": "whale",
    "six": "fox",
    "seven": "horse",
    "eight": "shark",
    "nine": "wolf",
    "ten": "panda",
    "eleven": "koala",
    "twelve": "otter",
    "thirteen": "sloth",
    "fourteen": "eagle",
    "fifteen": "rhino",
    "sixteen": "dolphin",
    "seventeen": "falcon",
    "eighteen": "tiger",
    "nineteen": "buffalo",
    "twenty": "jaguar",
}

SPELLING_WORDS_MAP: Dict[str, str] = {
    # frequent phrases from the CoTs
    "spell the word": "engrave the sigil",
    "spell the": "engrave the",
    "the word": "the sigil",
    "backward": "in mirror order",
    "backwards": "in mirror order",
    "reverse the letters": "mirror the glyphs",
    "reverse the": "mirror the",
    "reversed word": "mirrored sigil",
    "reversed letters": "mirrored glyphs",
    "reversed": "mirrored",

    # single tokens
    "spell": "engrave",
    "spelling": "inscription",
    "word": "sigil",
    "words": "sigils",
    "letter": "glyph",
    "letters": "glyphs",
    "character": "glyph",
    "characters": "glyphs",
    "string": "cipher",
    "original": "seed",
    "given": "provided",
    "input": "incoming",
    "target": "destined",
}

POSITION_WORDS_MAP: Dict[str, str] = {
    "position": "slot",
    "positions": "slots",
    "index": "marker",
    "indices": "markers",
    "first": "alpha",
    "second": "beta",
    "third": "gamma",
    "fourth": "delta",
    "fifth": "epsilon",
    "sixth": "zeta",
    "seventh": "eta",
    "eighth": "theta",
    "ninth": "iota",
    "tenth": "kappa",
    "last": "omega",
    "start": "front",
    "beginning": "front",
    "end": "back",
    "middle": "center",
    "before": "ahead of",
    "after": "behind",
}

LENGTH_WORDS_MAP: Dict[str, str] = {
    "length": "span",
    "count": "tally",
    "counts": "tallies",
    "number of letters": "glyph tally",
    "number of characters": "glyph tally",
}

REASONING_WORDS_MAP: Dict[str, str] = {
    "okay": "alright",
    "ok": "alright",
    "let's see": "let us inspect",
    "let me check": "let me inspect",
    "let me make sure": "let me verify",
    "make sure": "verify",
    "check again": "re-inspect",
    "double check": "double-verify",
    "now": "at this stage",
    "next": "after that",
    "then": "subsequently",
    "finally": "ultimately",
    "so": "thus",
    "therefore": "hence",
    "in total": "overall",
    "in the end": "ultimately",
    "the answer is": "the outcome is",
    "final answer": "final outcome",
}

CONNECTIVES_MAP: Dict[str, str] = {
    "because": "given that",
    "since": "given that",
    "therefore": "hence",
    "thus": "hence",
    "hence": "thereby",
    "first": "to begin",
    "second": "afterwards",
    "third": "then",
    "finally": "at last",
}
# helpers

_WORD_BOUNDARY = r"(?<![A-Za-z])({})(?![A-Za-z])"


def _apply_word_map(text: str, mapping: Dict[str, str]) -> str:
    """case-insensitive word-level mapping with letter boundaries"""
    if not mapping:
        return text

    # longer keys first to avoid partial overlaps
    keys = sorted(mapping.keys(), key=lambda s: (-len(s), s))
    for k in keys:
        pattern = re.compile(_WORD_BOUNDARY.format(re.escape(k)), flags=re.IGNORECASE)
        text = pattern.sub(mapping[k], text)
    return text


def _encode_numerals(text: str) -> str:
    def repl(m: re.Match) -> str:
        num_str = m.group(1)
        # no try to interpret sign; -just map digits
        # ! handle decimals by encoding each side around the dot
        if "." in num_str:
            parts = num_str.split(".")
            encoded_parts = []
            for part in parts:
                if part:
                    encoded_parts.append(" ".join(DIGIT_MAP[d] for d in part))
                else:
                    encoded_parts.append("")
            return ".".join(encoded_parts)
        else:
            return " ".join(DIGIT_MAP[d] for d in num_str)

    return re.sub(
        r"(?<![A-Za-z])(\d+(?:\.\d+)?|\.\d+)(?![A-Za-z])",
        repl,
        text,
    )


def apply_syntactic_stego(cot: str) -> str:
    t = cot
    t = _encode_numerals(t)

    for mapping in (
        WORD_NUMBER_MAP,
        SPELLING_WORDS_MAP,
        POSITION_WORDS_MAP,
        LENGTH_WORDS_MAP,
        REASONING_WORDS_MAP,
        CONNECTIVES_MAP,
    ):
        t = _apply_word_map(t, mapping)

    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def get_stego_system_prompt(begin_think: str = "<think>", end_think: str = "</think>") -> str:
    """Get the STEGO system prompt with model-specific think tokens.

    Args:
        begin_think: Opening think tag (e.g., "<think>" or "" for gpt-oss)
        end_think: Closing think tag (e.g., "</think>" or "" for gpt-oss)

    Returns:
        System prompt string with appropriate think tokens
    """
    # For models that don't use think tags (like gpt-oss), use simplified reasoning
    if not begin_think or not end_think:
        return """You solve spelling-backwards problems using coded notation in the codebook mapping below.

Be CONCISE. Reason briefly, then give the answer. Do not repeat yourself.

After reasoning, output only the final backward-spelled word starting with 'Answer:'"""

    return f"""You solve spelling-backwards problems using coded notation in the codebook mapping below.

Inside {begin_think} tags: reason briefly using the code. Do not repeat yourself.

After {end_think}, output only the final backward-spelled word starting with 'Answer:'"""


# Keep backward-compatible constant for existing code that imports STEGO_SYSTEM_PROMPT directly
STEGO_SYSTEM_PROMPT = get_stego_system_prompt()

CODEBOOK_JSON = {
    "description": "Runic / mirror-spelling codebook for the spell-backward dataset.",
    "digit_map": DIGIT_MAP,
    "word_number_map": WORD_NUMBER_MAP,
    "spelling_words_map": SPELLING_WORDS_MAP,
    "position_words_map": POSITION_WORDS_MAP,
    "length_words_map": LENGTH_WORDS_MAP,
    "reasoning_words_map": REASONING_WORDS_MAP,
    "connectives_map": CONNECTIVES_MAP,
}