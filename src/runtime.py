"""Helper untuk banner versi/seed di sel pertama notebook + run_log di sel terakhir."""

from __future__ import annotations

import csv
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _safe_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not-installed"


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("ascii").strip()
    except Exception:
        return "no-git"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | Path = "configs/experiment.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_banner(notebook_name: str, config: dict[str, Any]) -> None:
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    versions = {
        "python": platform.python_version(),
        "transformers": _safe_version("transformers"),
        "torch": _safe_version("torch"),
        "datasets": _safe_version("datasets"),
        "scikit-learn": _safe_version("sklearn"),
        "pandas": _safe_version("pandas"),
        "numpy": _safe_version("numpy"),
    }
    print("=" * 70)
    print(f"Notebook: {notebook_name}")
    print(f"Experiment: {config.get('experiment_name', '?')}")
    print(f"Seed: {seed}  |  Git: {_git_commit_short()}")
    print(f"Started at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z")
    print("Versi paket:")
    for name, ver in versions.items():
        print(f"  - {name}: {ver}")
    print("=" * 70)


@dataclass
class RunLog:
    notebook: str
    config_path: str
    start_at: float = field(default_factory=time.time)
    outputs: list[str] = field(default_factory=list)
    warnings_: list[str] = field(default_factory=list)

    def add_output(self, path: str | Path) -> None:
        self.outputs.append(str(path))

    def add_warning(self, msg: str) -> None:
        self.warnings_.append(msg)

    def save(self, log_path: str | Path = "reports/run_log.csv") -> None:
        end_at = time.time()
        duration = round(end_at - self.start_at, 2)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(
                    [
                        "notebook",
                        "config_path",
                        "start_at",
                        "end_at",
                        "duration_sec",
                        "outputs",
                        "warnings",
                        "git_commit",
                        "python_version",
                    ]
                )
            w.writerow(
                [
                    self.notebook,
                    self.config_path,
                    datetime.fromtimestamp(self.start_at, tz=timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    datetime.fromtimestamp(end_at, tz=timezone.utc).isoformat(timespec="seconds"),
                    duration,
                    " | ".join(self.outputs),
                    " | ".join(self.warnings_),
                    _git_commit_short(),
                    platform.python_version(),
                ]
            )
        print(f"[run_log] {self.notebook} → {duration}s, {len(self.outputs)} outputs, "
              f"{len(self.warnings_)} warnings → {log_path}")


def ensure_repo_root_on_sys_path() -> None:
    """Pastikan root repo ada di sys.path supaya `from src...` import berfungsi."""
    here = Path.cwd()
    for candidate in [here, *here.parents]:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    # Fallback: tambah cwd
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
