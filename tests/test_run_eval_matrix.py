"""Tests for the multi-checkpoint eval wrapper.

The expensive parts (actual robustness/explanation eval) are exercised by
``test_robustness.py`` / ``test_explanations.py``. This file just covers the
new code in ``run_eval_matrix.py``: the run-name → (dataset, variant,
modality) parser, plus a smoke test that the wrapper's discovery and
dry-run output handle the realistic mix of run names emitted by
``run_matrix.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_wrapper():
    """Import scripts/evaluate/run_eval_matrix.py without invoking main()."""
    spec = importlib.util.spec_from_file_location(
        "run_eval_matrix",
        REPO_ROOT / "scripts" / "evaluate" / "run_eval_matrix.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# parse_run_name
# ---------------------------------------------------------------------------


def test_parse_xmofe_main_runs():
    parse = _load_wrapper().parse_run_name
    assert parse("ch_sims_xmofe_s0") == ("ch_sims", "xmofe", None)
    assert parse("meld_xmofe_s2") == ("meld", "xmofe", None)
    assert parse("mosei_xmofe_s1") == ("mosei", "xmofe", None)


def test_parse_arch_ablations():
    parse = _load_wrapper().parse_run_name
    assert parse("ch_sims_xmofe_no_reliability_s0") == ("ch_sims", "xmofe_no_reliability", None)
    assert parse("meld_xmofe_no_interaction_s0") == ("meld", "xmofe_no_interaction", None)
    assert parse("mosei_xmofe_no_trimodal_s0") == ("mosei", "xmofe_no_trimodal", None)


def test_parse_loss_ablations_use_xmofe_variant():
    """Tier-4 loss ablations train xmofe with a modified loss config; for
    inference the architectural variant is still xmofe."""
    parse = _load_wrapper().parse_run_name
    assert parse("ch_sims_xmofe_no_faithfulness_s0") == ("ch_sims", "xmofe", None)
    assert parse("meld_xmofe_no_stability_s0") == ("meld", "xmofe", None)
    assert parse("mosei_xmofe_no_entropy_s0") == ("mosei", "xmofe", None)
    # The CH-SIMS-only "no reliability loss" tier-4 entry has the trailing
    # _loss disambiguator distinguishing it from the architectural ablation.
    assert parse("ch_sims_xmofe_no_reliability_loss_s0") == ("ch_sims", "xmofe", None)


def test_parse_unimodal_carries_modality():
    parse = _load_wrapper().parse_run_name
    assert parse("ch_sims_unimodal_text_s0") == ("ch_sims", "unimodal", "text")
    assert parse("meld_unimodal_audio_s0") == ("meld", "unimodal", "audio")
    assert parse("mosei_unimodal_visual_s0") == ("mosei", "unimodal", "visual")


def test_parse_fusion_baselines():
    parse = _load_wrapper().parse_run_name
    assert parse("ch_sims_early_fusion_s0") == ("ch_sims", "early_fusion", None)
    assert parse("meld_late_fusion_s0") == ("meld", "late_fusion", None)
    assert parse("mosei_hybrid_fusion_s0") == ("mosei", "hybrid_fusion", None)


def test_parse_unknown_dataset_raises():
    parse = _load_wrapper().parse_run_name
    with pytest.raises(ValueError, match="dataset"):
        parse("avmnist_xmofe_s0")


def test_parse_unknown_variant_raises():
    parse = _load_wrapper().parse_run_name
    with pytest.raises(ValueError, match="variant"):
        parse("ch_sims_some_new_method_s0")


def test_parse_handles_multi_digit_seeds():
    parse = _load_wrapper().parse_run_name
    assert parse("ch_sims_xmofe_s12") == ("ch_sims", "xmofe", None)


# ---------------------------------------------------------------------------
# Discovery + dry-run smoke
# ---------------------------------------------------------------------------


def test_discover_checkpoints_filters_to_runs_with_best_pt(tmp_path):
    discover = _load_wrapper().discover_checkpoints
    (tmp_path / "ch_sims_xmofe_s0").mkdir()
    (tmp_path / "ch_sims_xmofe_s0" / "best.pt").write_bytes(b"x")
    (tmp_path / "ch_sims_xmofe_s1").mkdir()  # no best.pt — should be skipped
    (tmp_path / "ch_sims_xmofe_s1" / "latest.pt").write_bytes(b"x")
    (tmp_path / "meld_xmofe_s0").mkdir()
    (tmp_path / "meld_xmofe_s0" / "best.pt").write_bytes(b"x")
    runs = discover(tmp_path)
    assert sorted(runs) == ["ch_sims_xmofe_s0", "meld_xmofe_s0"]


def test_discover_checkpoints_returns_empty_when_dir_missing(tmp_path):
    discover = _load_wrapper().discover_checkpoints
    assert discover(tmp_path / "nope") == []


def test_both_outputs_exist_logic(tmp_path):
    fn = _load_wrapper().both_outputs_exist
    # Nothing exists yet.
    assert fn(tmp_path, "run_a", want_robustness=True, want_explanations=True) is False
    # Only robustness exists.
    (tmp_path / "robustness_run_a.json").write_text("{}")
    assert fn(tmp_path, "run_a", want_robustness=True, want_explanations=True) is False
    assert fn(tmp_path, "run_a", want_robustness=True, want_explanations=False) is True
    # Both exist.
    (tmp_path / "explanations_run_a.json").write_text("{}")
    assert fn(tmp_path, "run_a", want_robustness=True, want_explanations=True) is True


def test_dry_run_smoke(tmp_path, capsys):
    """End-to-end dry run with a fake checkpoints/results tree."""
    import sys
    wrapper = _load_wrapper()

    ckpt_dir = tmp_path / "checkpoints"
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    for run_name in [
        "ch_sims_xmofe_s0",
        "meld_xmofe_no_interaction_s0",
        "mosei_unimodal_text_s0",
    ]:
        (ckpt_dir / run_name).mkdir(parents=True)
        (ckpt_dir / run_name / "best.pt").write_bytes(b"x")
    # One run already has its robustness output — should be marked R (uppercase).
    (res_dir / "robustness_ch_sims_xmofe_s0.json").write_text("{}")

    argv_save = sys.argv
    try:
        sys.argv = [
            "run_eval_matrix.py",
            "--checkpoints-dir", str(ckpt_dir),
            "--results-dir", str(res_dir),
            "--dry-run",
        ]
        rc = wrapper.main()
    finally:
        sys.argv = argv_save

    assert rc == 0
    out = capsys.readouterr().out
    assert "ch_sims (1 checkpoints)" in out
    assert "meld (1 checkpoints)" in out
    assert "mosei (1 checkpoints)" in out
    # Already-done robustness should appear as 'R' (uppercase) in status marks.
    assert "Re" in out or "[Re]" in out
