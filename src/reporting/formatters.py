"""Emit table data in the formats reviewers and the LaTeX paper need.

Three target formats:

* ``"tex"``  — booktabs LaTeX tables ready for ``\\input``.
* ``"md"``   — Markdown tables for sanity-check previews and READMEs.
* ``"csv"``  — Comma-separated for spreadsheet tools.

All emitters take a list-of-dict-rows + a column spec and return a string.
``write_table`` is a small helper that picks the format based on the output
file extension when called from CLI scripts.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

VALID_FORMATS = ("tex", "md", "csv")


def fmt_value(value: Any, decimals: int = 4, missing: str = "—") -> str:
    """Format a metric cell for tex/md (CSV uses raw ``str(value)``)."""
    if value is None:
        return missing
    if isinstance(value, float):
        if value != value:  # NaN
            return missing
        return f"{value:.{decimals}f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def to_markdown(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    headers: Sequence[str] | None = None,
    decimals: int = 4,
) -> str:
    if headers is None:
        headers = list(columns)
    if len(headers) != len(columns):
        raise ValueError("headers and columns must be the same length")
    out = io.StringIO()
    out.write("| " + " | ".join(headers) + " |\n")
    out.write("|" + "|".join(["---"] * len(columns)) + "|\n")
    for r in rows:
        out.write(
            "| "
            + " | ".join(fmt_value(r.get(c), decimals=decimals) for c in columns)
            + " |\n"
        )
    return out.getvalue()


def to_csv(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    headers: Sequence[str] | None = None,
) -> str:
    if headers is None:
        headers = list(columns)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(headers)
    for r in rows:
        writer.writerow([r.get(c) if r.get(c) is not None else "" for c in columns])
    return out.getvalue()


def to_latex(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    headers: Sequence[str] | None = None,
    caption: str = "",
    label: str = "",
    decimals: int = 4,
    column_format: str | None = None,
) -> str:
    """Return a ``booktabs`` LaTeX table (no ``\\begin{document}`` wrapping)."""
    if headers is None:
        headers = list(columns)
    n = len(columns)
    if column_format is None:
        # First column left-aligned (typical "row name"), rest centered.
        column_format = "l" + "c" * (n - 1) if n > 0 else ""

    def _esc(s: Any) -> str:
        text = str(s)
        return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    out = io.StringIO()
    out.write("\\begin{table}[h!]\n")
    out.write("    \\centering\n")
    out.write("    \\small\n")
    out.write(f"    \\begin{{tabular}}{{{column_format}}}\n")
    out.write("    \\toprule\n")
    out.write("    " + " & ".join(_esc(h) for h in headers) + " \\\\\n")
    out.write("    \\midrule\n")
    for r in rows:
        cells = [_esc(fmt_value(r.get(c), decimals=decimals)) for c in columns]
        out.write("    " + " & ".join(cells) + " \\\\\n")
    out.write("    \\bottomrule\n")
    out.write("    \\end{tabular}\n")
    if caption:
        out.write(f"    \\caption{{{caption}}}\n")
    if label:
        out.write(f"    \\label{{{label}}}\n")
    out.write("\\end{table}\n")
    return out.getvalue()


def render(
    fmt: str,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    headers: Sequence[str] | None = None,
    *,
    caption: str = "",
    label: str = "",
    decimals: int = 4,
) -> str:
    if fmt == "tex":
        return to_latex(rows, columns, headers, caption=caption, label=label, decimals=decimals)
    if fmt == "md":
        return to_markdown(rows, columns, headers, decimals=decimals)
    if fmt == "csv":
        return to_csv(rows, columns, headers)
    raise ValueError(f"format must be one of {VALID_FORMATS}; got {fmt!r}")


def write_table(
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    headers: Sequence[str] | None = None,
    *,
    caption: str = "",
    label: str = "",
    decimals: int = 4,
    fmt: str | None = None,
) -> Path:
    """Write a table to disk, inferring format from extension if ``fmt`` is None."""
    output_path = Path(output_path)
    if fmt is None:
        ext = output_path.suffix.lstrip(".").lower()
        if ext in VALID_FORMATS:
            fmt = ext
        elif ext == "tex":
            fmt = "tex"
        elif ext in ("markdown", "md"):
            fmt = "md"
        else:
            raise ValueError(
                f"could not infer format from extension {output_path.suffix!r}; "
                f"pass fmt explicitly"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render(fmt, rows, columns, headers, caption=caption, label=label, decimals=decimals),
        encoding="utf-8",
    )
    return output_path
