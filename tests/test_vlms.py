"""Tests for the VLM evaluation pipeline (spec §10 + §18).

Covers everything that doesn't require loading 14 GB checkpoints — prompts,
output parsing, frame sampling, stratified subsampling. End-to-end pipeline
(including the run script with ``--dry-run``) is exercised separately.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from src.vlms import (
    EMOTION_LABELS,
    SENTIMENT_LABELS,
    build_emotion_prompt,
    build_prompt,
    build_sentiment_prompt,
    extract_json,
    labels_for_task,
    parse_confidence,
    parse_label,
    parse_response,
    sample_frames_adaptive,
    stratified_subsample,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_sentiment_prompt_includes_transcript():
    prompt = build_sentiment_prompt("I am very happy today")
    assert "I am very happy today" in prompt
    assert "negative, neutral, positive" in prompt
    assert "JSON" in prompt


def test_emotion_prompt_includes_transcript_and_labels():
    prompt = build_emotion_prompt("Whatever, I don't care")
    assert "Whatever, I don't care" in prompt
    for label in EMOTION_LABELS:
        assert label in prompt


def test_build_prompt_dispatch():
    s = build_prompt("hi", task="sentiment")
    e = build_prompt("hi", task="emotion")
    assert "negative" in s and "anger" not in s
    assert "anger" in e and "negative" not in e


def test_build_prompt_invalid_task_raises():
    with pytest.raises(ValueError):
        build_prompt("hi", task="multi-label")


def test_labels_for_task_returns_correct_set():
    assert labels_for_task("sentiment") == SENTIMENT_LABELS
    assert labels_for_task("emotion") == EMOTION_LABELS
    with pytest.raises(ValueError):
        labels_for_task("topic")


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------


def test_extract_json_clean():
    obj = extract_json('{"label": "joy", "confidence": 0.8}')
    assert obj == {"label": "joy", "confidence": 0.8}


def test_extract_json_with_markdown_fence():
    raw = '```json\n{"label": "joy", "confidence": 0.6}\n```'
    obj = extract_json(raw)
    assert obj is not None and obj["label"] == "joy"


def test_extract_json_with_prose_around():
    raw = 'Sure, here you go:\n{"label": "anger", "confidence": 0.9}\nHope this helps!'
    obj = extract_json(raw)
    assert obj is not None and obj["label"] == "anger"


def test_extract_json_returns_none_on_garbage():
    assert extract_json("not json at all") is None
    assert extract_json("") is None


def test_parse_label_case_insensitive():
    raw = '{"label": "JOY"}'
    assert parse_label(raw, EMOTION_LABELS) == "joy"


def test_parse_label_freetext_fallback_picks_earliest():
    raw = "The speaker seems angry, though slightly neutral overall."
    # 'angry' isn't a valid label — but 'anger' isn't in the text either.
    # 'neutral' should win as the only valid label present.
    assert parse_label(raw, EMOTION_LABELS) == "neutral"


def test_parse_label_freetext_picks_first_valid_match():
    raw = "Despite some sadness, the speaker is mostly joyful."
    # Both 'sadness' and 'joy' are valid; expect the earliest mention.
    assert parse_label(raw, EMOTION_LABELS) == "sadness"


def test_parse_label_returns_none_when_no_match():
    raw = "The speaker is disturbed."
    assert parse_label(raw, EMOTION_LABELS) is None


def test_parse_confidence_clipped():
    assert parse_confidence('{"confidence": 1.5}') == 1.0
    assert parse_confidence('{"confidence": -0.2}') == 0.0
    assert parse_confidence('{"confidence": 0.4}') == 0.4
    assert parse_confidence("no json here") == 0.0


def test_parse_response_full_record():
    raw = '{"label": "Joy", "confidence": 0.7, "explanation": "Smiling."}'
    out = parse_response(raw, EMOTION_LABELS)
    assert out["label"] == "joy"
    assert out["confidence"] == 0.7
    assert out["explanation"] == "Smiling."
    assert out["parsed_ok"] is True


def test_parse_response_unparseable_marks_failure():
    raw = "I cannot determine the speaker's state."
    out = parse_response(raw, EMOTION_LABELS)
    assert out["label"] is None
    assert out["parsed_ok"] is False


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------


def test_stratified_subsample_preserves_class_proportions_roughly():
    # 70% A, 20% B, 10% C
    items = ["A"] * 70 + ["B"] * 20 + ["C"] * 10
    subset = stratified_subsample(items, label_fn=lambda x: x, n_total=20, seed=0)
    counts = Counter(subset)
    # Each class should have a proportional count (rounded)
    assert sum(counts.values()) == 20
    assert counts["A"] >= 10
    assert counts["B"] >= 2
    assert counts["C"] >= 1


def test_stratified_subsample_returns_all_when_n_exceeds_total():
    items = list(range(10))
    subset = stratified_subsample(items, label_fn=lambda x: x % 2, n_total=100, seed=0)
    assert len(subset) == 10


def test_stratified_subsample_deterministic_with_seed():
    items = list(range(100))
    a = stratified_subsample(items, label_fn=lambda x: x % 5, n_total=20, seed=7)
    b = stratified_subsample(items, label_fn=lambda x: x % 5, n_total=20, seed=7)
    assert sorted(a) == sorted(b)


def test_stratified_subsample_minimum_one_per_class():
    # 95 A, 5 B → asking for 5 should still keep at least one B
    items = ["A"] * 95 + ["B"] * 5
    subset = stratified_subsample(items, label_fn=lambda x: x, n_total=5, seed=0)
    counts = Counter(subset)
    assert counts.get("B", 0) >= 1


# ---------------------------------------------------------------------------
# Frame sampler — exercised against the existing MELD raw clip
# ---------------------------------------------------------------------------


from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MELD_CLIP = REPO_ROOT / "data" / "raw" / "meld" / "MELD.Raw" / "train_splits" / "dia0_utt0.mp4"


@pytest.mark.skipif(not SAMPLE_MELD_CLIP.exists(), reason="MELD raw clip not present")
def test_sample_frames_adaptive_returns_uint8_arrays():
    frames = sample_frames_adaptive(SAMPLE_MELD_CLIP, num_frames_long=8, num_frames_short=4)
    assert isinstance(frames, list)
    assert all(isinstance(f, np.ndarray) and f.dtype == np.uint8 and f.ndim == 3 for f in frames)
    assert len(frames) in (4, 8)


@pytest.mark.skipif(not SAMPLE_MELD_CLIP.exists(), reason="MELD raw clip not present")
def test_sample_frames_adaptive_short_clip_uses_4_frames():
    # Override threshold to a very high value so this clip counts as "short".
    frames = sample_frames_adaptive(
        SAMPLE_MELD_CLIP,
        short_threshold_sec=10000.0,
        num_frames_short=4, num_frames_long=8,
    )
    assert len(frames) == 4
