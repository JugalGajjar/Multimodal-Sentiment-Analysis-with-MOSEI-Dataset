"""Tests for the reporting library and table/figure generators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting import (
    fmt_value,
    load_explanation_runs,
    load_robustness_runs,
    load_training_runs,
    load_vlm_runs,
    render,
    to_csv,
    to_latex,
    to_markdown,
    write_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_fmt_value_handles_none_nan_int_float():
    assert fmt_value(None) == "—"
    assert fmt_value(float("nan")) == "—"
    assert fmt_value(0.123456, decimals=3) == "0.123"
    assert fmt_value(42) == "42"
    assert fmt_value("hello") == "hello"


def test_to_markdown_basic():
    rows = [{"a": 1, "b": 2.0}, {"a": 3, "b": 4.0}]
    out = to_markdown(rows, columns=["a", "b"], headers=["A", "B"])
    assert "| A | B |" in out
    assert "| 1 | 2.0000 |" in out
    assert "| 3 | 4.0000 |" in out


def test_to_csv_round_trip():
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    out = to_csv(rows, columns=["a", "b"])
    lines = out.strip().splitlines()
    assert lines[0] == "a,b"
    assert lines[1] == "1,x"
    assert lines[2] == "2,y"


def test_to_latex_contains_booktabs_skeleton():
    rows = [{"variant": "xmofe", "mae": 0.5}]
    out = to_latex(rows, columns=["variant", "mae"], headers=["Variant", "MAE"], caption="cap", label="tab:test")
    assert "\\toprule" in out
    assert "\\midrule" in out
    assert "\\bottomrule" in out
    assert "xmofe" in out
    assert "0.5000" in out
    assert "\\caption{cap}" in out
    assert "\\label{tab:test}" in out


def test_to_latex_escapes_underscores():
    rows = [{"variant": "xmofe_no_reliability", "mae": 0.5}]
    out = to_latex(rows, columns=["variant", "mae"])
    assert r"xmofe\_no\_reliability" in out


def test_render_dispatch():
    rows = [{"a": 1.0}]
    assert "1.0000" in render("md", rows, ["a"])
    assert "1.0000" in render("tex", rows, ["a"])
    assert "1.0" in render("csv", rows, ["a"])
    with pytest.raises(ValueError):
        render("html", rows, ["a"])


def test_write_table_infers_format_from_extension(tmp_path):
    rows = [{"x": 1.0}]
    write_table(tmp_path / "out.tex", rows, ["x"])
    write_table(tmp_path / "out.md", rows, ["x"])
    write_table(tmp_path / "out.csv", rows, ["x"])
    assert "\\toprule" in (tmp_path / "out.tex").read_text()
    assert "| x |" in (tmp_path / "out.md").read_text()
    assert "x\n" in (tmp_path / "out.csv").read_text()


def test_write_table_unknown_extension_raises(tmp_path):
    with pytest.raises(ValueError):
        write_table(tmp_path / "out.xls", [{"x": 1}], ["x"])


# ---------------------------------------------------------------------------
# Loaders — synthetic fixtures
# ---------------------------------------------------------------------------


def test_load_training_runs_picks_best_val_mae(tmp_path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    metrics = run_dir / "metrics.jsonl"
    metrics.write_text(
        json.dumps({"step": 1, "epoch": 0, "val/mae": 0.6}) + "\n"
        + json.dumps({"step": 2, "epoch": 1, "val/mae": 0.4}) + "\n"
        + json.dumps({"step": 3, "epoch": 2, "val/mae": 0.5}) + "\n"
    )
    runs = load_training_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["run_name"] == "run_a"
    assert runs[0]["val/mae"] == 0.4


def test_load_training_runs_picks_max_weighted_f1(tmp_path):
    run_dir = tmp_path / "run_b"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "val/weighted_f1": 0.40}) + "\n"
        + json.dumps({"step": 2, "val/weighted_f1": 0.55}) + "\n"
        + json.dumps({"step": 3, "val/weighted_f1": 0.50}) + "\n"
    )
    runs = load_training_runs(tmp_path)
    assert runs[0]["val/weighted_f1"] == 0.55


def test_load_training_runs_skips_runs_without_val_metrics(tmp_path):
    no_val = tmp_path / "run_c"
    no_val.mkdir()
    (no_val / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "train_step/loss": 0.5}) + "\n"
    )
    runs = load_training_runs(tmp_path)
    assert runs == []


def test_load_robustness_runs_attaches_source_filename(tmp_path):
    rec = {"experiment": "mosei", "variant": "xmofe", "summary": {"missing_avg_drop": 0.05}}
    (tmp_path / "robustness_xmofe.json").write_text(json.dumps(rec))
    (tmp_path / "explanations_xmofe.json").write_text(json.dumps(rec))   # different prefix → ignored
    out = load_robustness_runs(tmp_path)
    assert len(out) == 1
    assert out[0]["_source_file"] == "robustness_xmofe.json"
    assert out[0]["experiment"] == "mosei"


def test_loaders_return_empty_list_when_dir_missing(tmp_path):
    nope = tmp_path / "does_not_exist"
    assert load_training_runs(nope) == []
    assert load_robustness_runs(nope) == []
    assert load_explanation_runs(nope) == []
    assert load_vlm_runs(nope) == []


def test_load_robustness_handles_corrupt_json_without_crash(tmp_path):
    (tmp_path / "robustness_bad.json").write_text("{this is not json}")
    (tmp_path / "robustness_good.json").write_text(json.dumps({"experiment": "mosei"}))
    out = load_robustness_runs(tmp_path)
    assert len(out) == 1
    assert out[0]["experiment"] == "mosei"


# ---------------------------------------------------------------------------
# Integration: end-to-end against a fake "real training" directory
# ---------------------------------------------------------------------------
#
# Validates that the table generators (make_main_tables, make_ablation_tables)
# correctly populate rows when training-run names follow the canonical
# ``{dataset}_{variant_tag}_{timestamp}`` convention emitted by
# ``scripts/train/train_xmofe.py``. The smoke checkpoints we have on disk
# don't all follow that convention, so this test is the one that catches a
# regression where the run-name parsers stop matching what the trainer writes.


def _write_fake_run(logs_dir: Path, run_name: str, val_metric_key: str, val_value: float) -> None:
    run_dir = logs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "epoch": 0, val_metric_key: val_value + 0.1}) + "\n"
        + json.dumps({"step": 2, "epoch": 1, val_metric_key: val_value}) + "\n"
        + json.dumps({"step": 3, "epoch": 2, val_metric_key: val_value + 0.05}) + "\n"
    )


def test_full_pipeline_against_canonical_run_names(tmp_path):
    """Build a fake logs/ tree using the trainer's standard naming and run the
    aggregation + table generators, asserting the rows actually populate."""
    import subprocess

    logs_dir = tmp_path / "logs"
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "tables"
    logs_dir.mkdir()
    results_dir.mkdir()

    # train_xmofe.py emits run names like "{name_prefix}_{variant_tag}_{timestamp}"
    # where name_prefix is the dataset (e.g. "mosei") and variant_tag is the
    # model variant (e.g. "xmofe", "xmofe_no_reliability", "early_fusion").
    _write_fake_run(logs_dir, "mosei_xmofe_20260507_120000", "val/mae", 0.55)
    _write_fake_run(logs_dir, "mosei_xmofe_no_reliability_20260507_130000", "val/mae", 0.58)
    _write_fake_run(logs_dir, "mosei_early_fusion_20260507_140000", "val/mae", 0.62)
    _write_fake_run(logs_dir, "meld_xmofe_20260507_150000", "val/weighted_f1", 0.42)

    # Run collect_results.py
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["python", str(repo_root / "scripts" / "reporting" / "collect_results.py"),
         "--logs-dir", str(logs_dir), "--results-dir", str(results_dir),
         "--output-dir", str(results_dir)],
        check=True, capture_output=True,
    )
    bundle = json.loads((results_dir / "collected.json").read_text())
    assert len(bundle["training"]) == 4

    # Run make_main_tables.py against the fake bundle
    subprocess.run(
        ["python", str(repo_root / "scripts" / "reporting" / "make_main_tables.py"),
         "--collected", str(results_dir / "collected.json"),
         "--format", "md", "--output-dir", str(output_dir),
         "--external-baselines", str(repo_root / "configs" / "reporting" / "external_baselines.yaml")],
        check=True, capture_output=True,
    )
    main_mosei = (output_dir / "main_mosei.md").read_text()
    # Both our 'xmofe' run AND the 'early_fusion' run should appear; mae values
    # should match the best-epoch (= second-record) values we wrote.
    assert "xmofe" in main_mosei
    assert "early_fusion" in main_mosei
    assert "0.5500" in main_mosei

    # Run make_ablation_tables.py
    subprocess.run(
        ["python", str(repo_root / "scripts" / "reporting" / "make_ablation_tables.py"),
         "--collected", str(results_dir / "collected.json"),
         "--format", "md", "--output-dir", str(output_dir)],
        check=True, capture_output=True,
    )
    abl = (output_dir / "ablation_mosei.md").read_text()
    # Both the full xmofe and the no_reliability ablation should appear, but
    # not early_fusion (which isn't an ablation row).
    assert "xmofe" in abl
    assert "xmofe_no_reliability" in abl
    assert "early_fusion" not in abl
