"""Prompt perturbation utilities for LIBERO evaluation."""

from __future__ import annotations

import logging
import numpy as np

SYNONYM_MAP = {
    "pick": ["grab", "grasp", "take", "lift"],
    "place": ["put", "set", "position", "lay"],
    "open": ["unlock", "unfasten", "unseal"],
    "close": ["shut", "seal", "fasten"],
    "push": ["shove", "press", "move"],
    "pull": ["drag", "draw", "tug"],
    "turn": ["rotate", "twist", "spin"],
    "put": ["place", "set", "position"],
    "move": ["shift", "transfer", "relocate"],
    "take": ["grab", "pick", "grasp"],
    "lift": ["raise", "pick up", "elevate"],
}

OPPOSITE_MAP = {
    "open": "close",
    "close": "open",
    "pick": "place",
    "place": "pick",
    "pick up": "put down",
    "put": "pick up",
    "push": "pull",
    "pull": "push",
    "turn on": "turn off",
    "turn off": "turn on",
    "lift": "lower",
    "lower": "lift",
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
    "front": "back",
    "back": "front",
    "into": "out of",
    "out of": "into",
    "on": "off",
    "off": "on",
}


def perturb_prompt(
    original: str,
    mode: str = "original",
    all_tasks: list[str] | None = None,
    custom: str = "",
) -> str:
    """Return a perturbed prompt string.

    Args:
        original: The original task description.
        mode: Perturbation mode — one of:
            original  No change (default)
            empty     Return empty string
            shuffle   Randomly shuffle words
            random    Replace with a random different task description
            synonym   Replace one action verb with a synonym
            opposite  Replace one action/direction word with its opposite
            custom    Return the custom string verbatim
        all_tasks: List of all task descriptions (required for mode="random").
        custom: Custom prompt string (used when mode="custom").
    """
    if mode == "original":
        return original
    if mode == "custom":
        return custom
    if mode == "empty":
        return ""

    if mode == "shuffle":
        words = original.split()
        np.random.shuffle(words)
        result = " ".join(words)
        logging.info(f"Modified prompt (shuffle): {result}")
        return result

    if mode == "random":
        if not all_tasks:
            return original
        others = [t for t in all_tasks if t != original]
        result = str(np.random.choice(others)) if others else original
        logging.info(f"Modified prompt (random): {result}")
        return result

    if mode == "synonym":
        result = original.lower()
        for word, synonyms in SYNONYM_MAP.items():
            if word in result:
                replacement = str(np.random.choice(synonyms))
                result = result.replace(word, replacement, 1)
                break
        logging.info(f"Modified prompt (synonym): {result}")
        return result

    if mode == "opposite":
        result = original.lower()
        for phrase in sorted(OPPOSITE_MAP.keys(), key=len, reverse=True):
            if phrase in result:
                result = result.replace(phrase, OPPOSITE_MAP[phrase], 1)
                break
        logging.info(f"Modified prompt (opposite): {result}")
        return result

    return original
