"""Helpers shared by the scoring scripts.

- Resumable JSONL output: `terminate_partial_line` and `read_done_keys` let a
  rerun skip rows already written and append the rest.
- `load_once`: one model per process when `--method a,b,c` scores several
  DreamBench++ methods in a row.
- `write_run_meta`: one JSON line per invocation in `<result>.meta.jsonl`
  (code commit, config hash, library versions, GPU, SLURM job, and the
  Hugging Face snapshot each model id resolved to), so every result file
  records which run produced it.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def hf_snapshot(model_id: str) -> str | None:
    """Commit hash the local HF cache resolves `model_id` to (refs/main)."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    ref = Path(hub) / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "numpy", "diffusers", "timm", "sam3",
                 "dreamsim", "spacy", "t2v_metrics", "ImageReward", "hpsv3"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:
            continue
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["cuda"] = torch.version.cuda
    except Exception:
        pass
    return out


def write_run_meta(result_path: Path, *, config: dict | None = None,
                   config_path: Path | None = None, model_ids: list[str] | None = None,
                   argv: list[str] | None = None, extra: dict | None = None) -> None:
    rec = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "code_commit": _git_commit(),
        "argv": argv,
        "config_path": str(config_path) if config_path else None,
        "config_sha256": (hashlib.sha256(config_path.read_bytes()).hexdigest()
                          if config_path and config_path.exists() else None),
        "config": config,
        "versions": _versions(),
        "hf_snapshots": {m: hf_snapshot(m) for m in (model_ids or []) if "/" in m},
        **(extra or {}),
    }
    meta = result_path.with_name(result_path.name + ".meta.jsonl")
    with meta.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# --------------------------------------------------------------------------
# Command-line and resume helpers

def method_list(value: str) -> list[str]:
    """argparse type for `--method a,b,c`."""
    return [m for m in (s.strip() for s in value.split(",")) if m]


class _BuilderDefault:
    """Sentinel for `--min-fg-fraction` not given: keep the sample builder's
    own threshold. Not a string, so argparse does not pass it through `type`."""

    def __repr__(self) -> str:
        return "the builder's default"


BUILDER_DEFAULT = _BuilderDefault()


def min_fg_fraction(value: str) -> float | None:
    """argparse type for `--min-fg-fraction`: a fraction, or `none` to keep
    every pair (the builder then records both masks' foreground fractions)."""
    return None if value.lower() == "none" else float(value)


_LOADED: dict = {}


def load_once(key: str, loader: Callable):
    """Return `loader()`, calling it at most once per `key` and process."""
    if key not in _LOADED:
        _LOADED[key] = loader()
    return _LOADED[key]


def terminate_partial_line(path: Path) -> None:
    """A job killed mid-write leaves a last line without its newline; close
    it so appended rows start on a fresh line (readers skip the broken one)."""
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
        if last != b"\n":
            with path.open("a") as f:
                f.write("\n")


def read_done_keys(path: Path, key: Callable[[dict], tuple]) -> set[tuple]:
    """Keys of the complete rows already in a JSONL result file."""
    done: set[tuple] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # a row cut by a killed job; it is scored again
    return done
