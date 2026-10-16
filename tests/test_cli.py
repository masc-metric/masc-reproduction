"""Command-line options of the scoring scripts (no models are loaded)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from repro_eval.runs import BUILDER_DEFAULT  # noqa: E402

SCRIPTS = ["compute_geometry", "compute_prompt_following", "compute_prompt_following_external"]


def _parser(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parser()


@pytest.mark.parametrize("name", SCRIPTS)
def test_defaults_keep_the_builder_threshold(name):
    args = _parser(name).parse_args(["--config", "c.yaml"])
    assert args.min_fg_fraction is BUILDER_DEFAULT
    assert args.method == [None]
    assert args.tag_suffix == "" and args.max_samples is None


@pytest.mark.parametrize("name", SCRIPTS)
def test_unfiltered_runs_and_method_lists(name):
    args = _parser(name).parse_args(["--config", "c.yaml", "--method", "emu2, dreambooth_sd",
                                     "--min-fg-fraction", "none", "--tag-suffix", "_nofilter"])
    assert args.min_fg_fraction is None
    assert args.method == ["emu2", "dreambooth_sd"]
    args = _parser(name).parse_args(["--config", "c.yaml", "--min-fg-fraction", "0.025"])
    assert args.min_fg_fraction == 0.025


def test_script_specific_options():
    assert _parser("compute_geometry").parse_args(
        ["--config", "c.yaml", "--dump-patch-arrays"]).dump_patch_arrays
    pf = _parser("compute_prompt_following")
    assert pf.parse_args(["--config", "c.yaml"]).stripper is None
    assert pf.parse_args(["--config", "c.yaml", "--stripper", "spacy"]).stripper == "spacy"
