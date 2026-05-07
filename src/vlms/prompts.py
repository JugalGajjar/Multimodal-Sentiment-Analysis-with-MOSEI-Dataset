"""Prompt templates for VLM-based affective evaluation (spec §18.2 + §18.3).

Two templates: 3-class sentiment (CH-SIMS, MOSEI's binarized form if you ever
get raw videos) and 7-class emotion (MELD). Both ask the model to return a
strict JSON object so the output parser has a chance at robust extraction.
"""

from __future__ import annotations

SENTIMENT_LABELS: tuple[str, ...] = ("negative", "neutral", "positive")

EMOTION_LABELS: tuple[str, ...] = (
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
)

# MELD's emotion label space matches our spec ordering (with "joy" instead of
# "happy"). Index in this tuple == emotion id used by EMOTION_LABELS.

SENTIMENT_PROMPT_TEMPLATE = """You are given a video utterance represented by sampled frames and its transcript.

Transcript:
"{transcript}"

Classify the speaker's sentiment as one of:
negative, neutral, positive.

Return JSON only:
{{
  "label": "...",
  "confidence": 0.0-1.0,
  "explanation": "one short sentence"
}}"""

EMOTION_PROMPT_TEMPLATE = """You are given a dialogue utterance represented by sampled frames and its transcript.

Transcript:
"{transcript}"

Classify the speaker's emotion as one of:
anger, disgust, fear, joy, neutral, sadness, surprise.

Return JSON only:
{{
  "label": "...",
  "confidence": 0.0-1.0,
  "explanation": "one short sentence"
}}"""


def build_sentiment_prompt(transcript: str) -> str:
    return SENTIMENT_PROMPT_TEMPLATE.format(transcript=transcript)


def build_emotion_prompt(transcript: str) -> str:
    return EMOTION_PROMPT_TEMPLATE.format(transcript=transcript)


def labels_for_task(task: str) -> tuple[str, ...]:
    """Return the valid label set for the named task."""
    if task == "sentiment":
        return SENTIMENT_LABELS
    if task == "emotion":
        return EMOTION_LABELS
    raise ValueError(f"unknown VLM task {task!r}; expected 'sentiment' or 'emotion'")


def build_prompt(transcript: str, task: str) -> str:
    if task == "sentiment":
        return build_sentiment_prompt(transcript)
    if task == "emotion":
        return build_emotion_prompt(transcript)
    raise ValueError(f"unknown VLM task {task!r}")
