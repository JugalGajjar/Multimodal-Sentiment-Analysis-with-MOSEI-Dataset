"""Training logger: file + stdout + W&B with graceful offline fallback.

Designed so a training run still works without ``WANDB_API_KEY`` or even
without the ``wandb`` package installed — it just logs locally. When W&B is
configured, we treat it as the primary experiment tracker and keep the file
log as a tail-friendly mirror.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrainingLogger:
    """Routes scalars and artifacts to file, stdout, and (optionally) W&B."""

    def __init__(
        self,
        log_dir: str | Path,
        run_name: str,
        config: dict[str, Any],
        wandb_project: str = "xmofe",
        wandb_entity: str | None = None,
        use_wandb: bool = True,
        resume_run_id: str | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.config = config
        self.wandb_run_id: str | None = None

        # ---- file + stdout logger -------------------------------------
        self.logger = logging.getLogger(f"xmofe.{run_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        # Remove duplicate handlers if logger reused (e.g. in tests).
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(self.log_dir / "training.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # ---- W&B tracker ----------------------------------------------
        self.wandb = self._init_wandb(
            run_name=run_name,
            project=wandb_project,
            entity=wandb_entity,
            use_wandb=use_wandb,
            resume_run_id=resume_run_id,
        )

        # ---- newline-delimited JSON metrics dump -----------------------
        # Always written so test runs / offline runs can still produce a
        # machine-readable training record.
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self._jsonl_handle = self.jsonl_path.open("a", encoding="utf-8")

        self.info(f"run_name={run_name}  log_dir={self.log_dir}  wandb={'on' if self.wandb else 'off'}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None, prefix: str = "") -> None:
        """Log a flat dict of scalars to all configured destinations."""
        flat: dict[str, Any] = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
        # File + stdout: pretty-printed key=value lines
        joined = " ".join(f"{k}={self._fmt(v)}" for k, v in sorted(flat.items()))
        head = f"step={step} " if step is not None else ""
        self.logger.info(f"{head}{joined}")
        # JSONL — always
        record: dict[str, Any] = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **flat,
        }
        self._jsonl_handle.write(json.dumps(record, default=self._json_default) + "\n")
        self._jsonl_handle.flush()
        # W&B
        if self.wandb is not None:
            self.wandb.log(flat, step=step)

    def finish(self) -> None:
        try:
            if self.wandb is not None:
                self.wandb.finish()
        finally:
            self._jsonl_handle.close()
            for h in list(self.logger.handlers):
                self.logger.removeHandler(h)
                h.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_wandb(
        self,
        run_name: str,
        project: str,
        entity: str | None,
        use_wandb: bool,
        resume_run_id: str | None,
    ):
        if not use_wandb:
            return None
        try:
            from dotenv import load_dotenv  # type: ignore[import-not-found]
            load_dotenv()
        except ImportError:
            pass

        if not os.environ.get("WANDB_API_KEY"):
            self.logger.warning(
                "WANDB_API_KEY not set; falling back to local file/stdout logging"
            )
            return None

        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError:
            self.logger.warning("wandb package not installed; local logging only")
            return None

        try:
            run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=self.config,
                dir=str(self.log_dir),
                id=resume_run_id,
                resume="allow" if resume_run_id else None,
            )
            self.wandb_run_id = run.id if run is not None else None
            return wandb
        except Exception as e:  # noqa: BLE001 — keep training resilient if W&B init fails
            self.logger.warning(f"wandb.init failed ({e}); local logging only")
            return None

    @staticmethod
    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    @staticmethod
    def _json_default(o: Any) -> Any:
        # Numpy / torch scalars
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        return str(o)
