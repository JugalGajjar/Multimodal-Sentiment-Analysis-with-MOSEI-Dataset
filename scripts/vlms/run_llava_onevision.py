"""Spec-compliant entry point for LLaVA-OneVision evaluation.

Thin shim that pre-fills ``--vlm llava`` and forwards to
:mod:`scripts.vlms.run_vlm`. Use ``run_vlm.py --vlm llava`` directly if you
prefer one canonical script; both produce identical output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.vlms.run_vlm import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--vlm", "llava", *sys.argv[1:]]
    main()
