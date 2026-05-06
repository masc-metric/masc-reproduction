"""Per-pair speed microbenchmark for every metric in CONTRIBUTIONS Tables 1-3.

Reports, per metric:
  - params (M)               : total parameters of the underlying model(s)
  - peak_mem (MiB)           : peak GPU memory observed during the timed loop
  - latency_ms (median)      : per-pair inference time in steady state
  - throughput (pairs/s)     : 1000 / latency_ms

Methodology:
  - Single fixed (ref, out, ref_mask, out_mask, prompt) DB++ pair.
  - Images / masks / prompt are pre-loaded once. The timed call is the
    final scoring call (`fg_match` for CP, `score` / `score_pair_*` for PF).
  - PIL is left un-decoded between calls so backends that need RGB conversion
    pay that cost on every call (matches how the eval pipeline runs).
  - 5 warmup calls (results discarded), then 20 timed calls. Median is
    reported; mean and stdev are also recorded for transparency.
  - `torch.cuda.synchronize()` is called before and after every timed call.
  - Each metric is loaded in a fresh process to keep peak-memory readings
    clean and to avoid cumulative GPU fragmentation across the lineup. We
    do this by re-execing this script once per metric in a worker mode.

Usage:
    # Run all metrics (each in its own worker process). Writes results
    # to results/benchmark_speed.jsonl, then prints the markdown table.
    python scripts/benchmark_speed.py

    # Run a single metric (worker mode — used internally).
    python scripts/benchmark_speed.py --only <metric_id>

    # Skip metrics by id (useful to skip slow ones during iteration).
    python scripts/benchmark_speed.py --skip dift_canonical,vqascore,hpsv3

    # Custom warmup / runs.
    python scripts/benchmark_speed.py --warmup 5 --runs 20
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

def _disable_torchcodec_before_transformers() -> None:
    """transformers 5.x audio_utils.py runs `is_torchcodec_available()` at
    import-time. The TorchCodec wheel needs FFmpeg shared libs (libavutil
    .so.{56,57,58,59}); on a host without ffmpeg this raises during
    `find_spec("torchcodec")`. Replace torchcodec in sys.modules with a
    stub module whose __spec__ resolves cleanly, AND override
    `is_torchcodec_available` to always return False so the audio path
    skips the codec-dependent branch entirely. Must run before any
    transformers import."""
    try:
        import sys as _sys
        import types as _types
        import importlib.machinery as _machinery
        # Pre-populate a stub torchcodec module with a valid spec so
        # find_spec("torchcodec") doesn't crash inside transformers.
        if "torchcodec" not in _sys.modules:
            stub = _types.ModuleType("torchcodec")
            stub.__spec__ = _machinery.ModuleSpec("torchcodec", loader=None)
            stub.__path__ = []  # mark as a package
            _sys.modules["torchcodec"] = stub
        if "torchcodec.decoders" not in _sys.modules:
            sub = _types.ModuleType("torchcodec.decoders")
            sub.__spec__ = _machinery.ModuleSpec("torchcodec.decoders", loader=None)
            class _VideoDecoder: pass
            sub.VideoDecoder = _VideoDecoder
            _sys.modules["torchcodec.decoders"] = sub
            _sys.modules["torchcodec"].decoders = sub
        # Override transformers' availability check.
        try:
            import transformers.utils.import_utils as _iu
            _iu.is_torchcodec_available = lambda: False
        except Exception:
            pass
    except Exception:
        pass


_disable_torchcodec_before_transformers()


# transformers 5.6+ refuses to call torch.load() unless torch >= 2.6 due to
# CVE-2025-32434, even though the affected code path is `weights_only=False`
# and we're loading first-party HF checkpoints (openai/, facebook/, ...). The
# repo's .venv ships torch 2.5.1, so we monkey-patch the gate off. This only
# affects the benchmarking process; production runs should upgrade torch.
def _patch_safetensors_gate() -> None:
    try:
        import transformers.utils.import_utils as _iu
        _iu.check_torch_load_is_safe = lambda *_a, **_kw: None
        import transformers.modeling_utils as _mu
        if hasattr(_mu, "check_torch_load_is_safe"):
            _mu.check_torch_load_is_safe = lambda *_a, **_kw: None
    except Exception:
        pass


def _patch_legacy_transformers_imports() -> None:
    """Older packages (ImageReward) `from transformers.modeling_utils import
    apply_chunking_to_forward`, which moved to `transformers.pytorch_utils`
    in transformers >= 4.x. Re-expose the legacy names. For symbols that
    have been removed entirely in the installed version (e.g.
    `find_pruneable_heads_and_indices`), expose a stub that raises only
    if the function is actually invoked — the import itself succeeds."""
    try:
        import transformers.modeling_utils as _mu
        from transformers import pytorch_utils as _pu
        for sym in (
            "apply_chunking_to_forward",
            "prune_linear_layer",
            "prune_layer",
            "Conv1D",
            "prune_conv1d_layer",
        ):
            if hasattr(_pu, sym) and not hasattr(_mu, sym):
                setattr(_mu, sym, getattr(_pu, sym))

        # `find_pruneable_heads_and_indices` was removed in transformers
        # 5.x; install a stub on BOTH modeling_utils and pytorch_utils
        # (lavis / t2v_metrics imports it from either path). Lets the
        # import succeed but raises if head-pruning is actually exercised
        # (not on the inference path we time).
        def _stub(*_a, **_kw):  # pragma: no cover
            raise NotImplementedError(
                "find_pruneable_heads_and_indices is removed in "
                "transformers >=5; benchmark only exercises the "
                "inference path, which does not need it."
            )
        if not hasattr(_mu, "find_pruneable_heads_and_indices"):
            _mu.find_pruneable_heads_and_indices = _stub
        if not hasattr(_pu, "find_pruneable_heads_and_indices"):
            _pu.find_pruneable_heads_and_indices = _stub
    except Exception:
        pass


def _patch_legacy_tokenizer() -> None:
    """transformers 5.x removed `additional_special_tokens_ids` (sequence
    of int ids). ImageReward's BLIP code reads
    `tokenizer.additional_special_tokens_ids[0]` once during model build
    to set `enc_token_id`. Re-expose the property by deriving ids from
    the still-present `additional_special_tokens` strings."""
    try:
        from transformers.tokenization_utils_base import (
            PreTrainedTokenizerBase as _Base,
        )
        if not hasattr(_Base, "additional_special_tokens_ids"):
            def _ids(self):
                toks = getattr(self, "additional_special_tokens", []) or []
                return [self.convert_tokens_to_ids(t) for t in toks]
            _Base.additional_special_tokens_ids = property(_ids)
    except Exception:
        pass


def _patch_legacy_transformers_modules() -> None:
    """transformers 5.x removed `transformers.utils.model_parallel_utils`
    entirely. lavis's modeling_t5.py imports `assert_device_map` and
    `get_device_map` from it. Inject a fake module with no-op stubs."""
    try:
        import sys as _sys
        import types as _types
        modname = "transformers.utils.model_parallel_utils"
        if modname not in _sys.modules:
            mod = _types.ModuleType(modname)
            def _stub(*_a, **_kw):  # pragma: no cover
                raise NotImplementedError(
                    f"{modname} is removed in transformers >=5; benchmark "
                    "exercises only inference, which does not need it."
                )
            mod.assert_device_map = _stub
            mod.get_device_map = _stub
            _sys.modules[modname] = mod
            try:
                import transformers.utils as _tu
                _tu.model_parallel_utils = mod
            except Exception:
                pass
    except Exception:
        pass


def _patch_legacy_processing_utils() -> None:
    """transformers 5.x removed `_validate_images_text_input_order` from
    `transformers.processing_utils`. tarsier_processor (in t2v_metrics
    lavis) imports it. Stub it as a pass-through (returns its inputs)."""
    try:
        import transformers.processing_utils as _pu
        if not hasattr(_pu, "_validate_images_text_input_order"):
            def _passthrough(images, text):  # pragma: no cover
                return images, text
            _pu._validate_images_text_input_order = _passthrough
    except Exception:
        pass


_patch_safetensors_gate()
_patch_legacy_transformers_imports()
_patch_legacy_tokenizer()
_patch_legacy_transformers_modules()
_patch_legacy_processing_utils()

from repro_eval.data import REPO_ROOT, RESULTS_DIR  # noqa: E402

OUT_PATH = RESULTS_DIR / "benchmark_speed.jsonl"

# ---- fixed sample ----------------------------------------------------------
# A real DB++ (ref, out, masks, prompt) pair. Both reference and generation
# are 512x512 (the DB++ canonical output size); the SAM3 reference mask
# covers the canonical object.
_SUBJECT = "object_00_motorcycle"
_METHOD_FULL = "dreambooth_sd_gs7_5_step100_seed42_torch_float16"
SAMPLE_PAIR = {
    "ref_image":  REPO_ROOT / "data/dreambench_plus/images/object/00.jpg",
    "ref_mask":   REPO_ROOT / "data/masks/dreambench_plus/refs/object/00.png",
    "out_image":  REPO_ROOT / f"data/dreambench_plus/samples/{_METHOD_FULL}/tgt_image/{_SUBJECT}/0_0.jpg",
    "out_mask":   REPO_ROOT / f"data/masks/dreambench_plus/samples/{_METHOD_FULL}/{_SUBJECT}/0_0.png",
    "prompt":     "A photograph of a motorcycle parked beside a bustling city street at night, illuminated by street lights",
    "object":     "motorcycle",
    "prompt_nosubj": "A photograph of parked beside a bustling city street at night, illuminated by street lights",
}


def _params_of(model) -> int:
    """Sum of `numel()` for every nn.Parameter we can find under `model`.

    DIFT-SDXL holds VAE + UNet + (already-deleted) text encoders in a
    Module wrapper, so `sum(p.numel() for p in model.parameters())` works.
    DreamSim's wrapper is also nn.Module-backed.
    """
    import torch.nn as nn
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters())
    # Fallback: introspect attributes that look like Modules.
    total = 0
    seen: set[int] = set()
    for v in vars(model).values() if hasattr(model, "__dict__") else []:
        if isinstance(v, nn.Module) and id(v) not in seen:
            total += sum(p.numel() for p in v.parameters())
            seen.add(id(v))
    return total


# ---- timing harness --------------------------------------------------------

@dataclass
class TimingResult:
    metric_id: str
    display_name: str
    table: str          # "CP" or "PF"
    kind: str           # "non-LLM" | "LLM judge" | "llm-judge-cost"
    params_M: float
    peak_mem_MiB: float
    latency_ms_median: float
    latency_ms_mean: float
    latency_ms_stdev: float
    throughput_pairs_per_s: float
    n_runs: int
    notes: str = ""


def _time_loop(call_fn, warmup: int, runs: int) -> tuple[list[float], float]:
    """Time `call_fn()` for `warmup + runs` iterations, returning (per-call ms, peak_MiB).

    Synchronizes around every call. `peak_MiB` is read after the timed loop
    via `torch.cuda.max_memory_allocated()`, which counts from the most
    recent reset.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(warmup):
        _ = call_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = call_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak = 0.0
    return timings, peak


# ---- metric loaders --------------------------------------------------------
# Each loader returns (call_fn, params, display_name, table, notes).
# `call_fn` is a no-arg callable that runs ONE per-pair scoring forward.

def _pil_load() -> dict:
    from PIL import Image
    import numpy as np
    out = {
        "ref_pil": Image.open(SAMPLE_PAIR["ref_image"]).convert("RGB"),
        "out_pil": Image.open(SAMPLE_PAIR["out_image"]).convert("RGB"),
        "ref_mask_np": np.array(Image.open(SAMPLE_PAIR["ref_mask"])),
        "out_mask_np": np.array(Image.open(SAMPLE_PAIR["out_mask"])),
        "out_mask_path": SAMPLE_PAIR["out_mask"],
        "prompt": SAMPLE_PAIR["prompt"],
        "prompt_nosubj": SAMPLE_PAIR["prompt_nosubj"],
    }
    return out


# ---- CP metrics ------------------------------------------------------------

def _load_clipi():
    from repro_eval.geometry.clipi_dbplus import load_matcher
    m = load_matcher()
    s = _pil_load()
    fn = lambda: m.fg_match(s["ref_pil"], s["ref_mask_np"], s["out_pil"], s["out_mask_np"])
    return fn, _params_of(m.model), "CLIP-I (DB++ shipped)", "CP", "openai/clip-vit-base-patch32"


def _load_dinoi():
    from repro_eval.geometry.dinoi_dbplus import load_matcher
    m = load_matcher()
    s = _pil_load()
    fn = lambda: m.fg_match(s["ref_pil"], s["ref_mask_np"], s["out_pil"], s["out_mask_np"])
    return fn, _params_of(m.model), "DINO-I (DB++ shipped)", "CP", "facebookresearch/dino:main / dino_vits8"


def _load_dinov3():
    from repro_eval.geometry.dinov3_dbplus import load_matcher
    m = load_matcher(model_id="facebook/dinov3-vitl16-pretrain-lvd1689m", input_size=224)
    s = _pil_load()
    fn = lambda: m.fg_match(s["ref_pil"], s["ref_mask_np"], s["out_pil"], s["out_mask_np"])
    return fn, _params_of(m.model), "DINOv3 ViT-L/16 CLS cosine", "CP", "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _load_dreamsim():
    from repro_eval.geometry.dreamsim import load_matcher
    m = load_matcher()
    s = _pil_load()
    fn = lambda: m.fg_match(s["ref_pil"], s["ref_mask_np"], s["out_pil"], s["out_mask_np"])
    return fn, _params_of(m.model), "DreamSim (NIGHTS-finetuned ensemble)", "CP", "DINO + CLIP + OpenCLIP ensemble"


def _load_radio_summary():
    from repro_eval.geometry.radio_summary import load_matcher
    m = load_matcher()
    s = _pil_load()
    fn = lambda: m.fg_match(s["ref_pil"], s["ref_mask_np"], s["out_pil"], s["out_mask_np"])
    return fn, _params_of(m.model), "AM-RADIO C-RADIOv4-SO400M summary cosine", "CP", "nvidia/C-RADIOv4-SO400M"


def _load_dift_canonical():
    from repro_eval.geometry.dift import load_matcher
    m = load_matcher(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        input_size=1024,
        timestep=261, up_ft_index=1, prompt="", seed=0,
        score_type="masked_maxcos", fg_threshold=0.5,
    )
    s = _pil_load()
    fn = lambda: m.fg_match(s["ref_pil"], s["ref_mask_np"], s["out_pil"], s["out_mask_np"])
    return fn, _params_of(m.model), "DIFT-SDXL canonical", "CP", "stabilityai/stable-diffusion-xl-base-1.0 (timestep=261, up_ft_index=1)"


def _load_siglip2_global():
    from repro_eval.geometry.siglip2_global import load_matcher
    m = load_matcher(model_id="google/siglip2-so400m-patch16-naflex", naflex_max_patches=1024)
    s = _pil_load()
    fn = lambda: m.fg_match(s["ref_pil"], s["ref_mask_np"], s["out_pil"], s["out_mask_np"])
    return fn, _params_of(m.model), "SigLIP2 SO400M-NaFlex global pool (same-backbone ablation)", "CP", "google/siglip2-so400m-patch16-naflex (1024 patches)"


def _load_ours_cp():
    from repro_eval.geometry.siglip2 import load_matcher
    m = load_matcher(
        model_id="google/siglip2-so400m-patch16-naflex",
        input_size=512, naflex_max_patches=1024,
        score_type="masked_maxcos", fg_threshold=0.5,
    )
    s = _pil_load()
    fn = lambda: m.fg_match(s["ref_pil"], s["ref_mask_np"], s["out_pil"], s["out_mask_np"])
    return fn, _params_of(m.model), "Ours: SigLIP2 SO400M-NaFlex masked-maxcos", "CP", "google/siglip2-so400m-patch16-naflex (1024 patches, masked-maxcos)"


# ---- PF metrics ------------------------------------------------------------

def _load_clipt():
    """CLIP-T baseline path from compute_prompt_following.py."""
    from transformers import CLIPModel, CLIPProcessor
    from scripts.compute_prompt_following import score_pair_clip
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).eval().to("cuda")
    s = _pil_load()
    fn = lambda: score_pair_clip(model, processor, s["out_pil"], s["prompt"], device="cuda")
    return fn, _params_of(model), "CLIP-T (DB++ shipped)", "PF", "openai/clip-vit-base-patch32"


def _load_siglip2t_global():
    """SigLIP2-T global-pool same-backbone ablation."""
    from transformers import AutoModel, AutoProcessor
    from scripts.compute_prompt_following import score_pair_siglip2
    model_id = "google/siglip2-so400m-patch16-naflex"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to("cuda")
    s = _pil_load()
    fn = lambda: score_pair_siglip2(model, processor, s["out_pil"], s["prompt"],
                                    naflex_max_patches=1024, device="cuda")
    return fn, _params_of(model), "SigLIP2 SO400M-NaFlex global pool (same-backbone ablation)", "PF", "google/siglip2-so400m-patch16-naflex (full prompt)"


def _load_ours_pf():
    """Ours: SigLIP2 BG-pool + nosubj.

    Mirrors the compute_prompt_following.py path with variant=global_bg
    and prompt subject-stripping.
    """
    from transformers import AutoModel, AutoProcessor
    from scripts.compute_prompt_following import score_pair_siglip2_masked_pool
    model_id = "google/siglip2-so400m-patch16-naflex"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to("cuda")
    s = _pil_load()
    fn = lambda: score_pair_siglip2_masked_pool(
        model, processor, s["out_pil"], s["out_mask_path"],
        s["prompt_nosubj"], naflex_max_patches=1024, device="cuda", use_bg=True)
    return fn, _params_of(model), "Ours: SigLIP2 SO400M-NaFlex BG-pool + nosubj", "PF", "google/siglip2-so400m-patch16-naflex (BG-pool, subject-stripped)"


def _load_vqascore():
    from repro_eval.pf.vqascore import load_scorer
    sc = load_scorer()
    s = _pil_load()
    fn = lambda: sc.score(s["out_pil"], s["prompt"])
    # t2v_metrics' VQAScore wraps a generative VLM (clip-flant5-xxl, ~3B).
    # Its `model` attribute is itself a wrapper; introspect its parameters.
    try:
        params = _params_of(sc.model.model)  # type: ignore[attr-defined]
    except Exception:
        params = 0
    return fn, params, "VQAScore (clip-flant5-xxl)", "PF", "clip-flant5-xxl (~3B)"


def _load_imagereward():
    from repro_eval.pf.imagereward import load_scorer
    sc = load_scorer()
    s = _pil_load()
    fn = lambda: sc.score(s["out_pil"], s["prompt"])
    try:
        params = _params_of(sc.model)
    except Exception:
        params = 0
    return fn, params, "ImageReward (BLIP fine-tune)", "PF", "ImageReward-v1.0 (BLIP)"


def _load_hpsv3():
    from repro_eval.pf.hpsv3 import load_scorer
    sc = load_scorer()
    s = _pil_load()
    fn = lambda: sc.score(s["out_pil"], s["prompt"])
    try:
        params = _params_of(sc.model.model)
    except Exception:
        try:
            params = _params_of(sc.model)
        except Exception:
            params = 0
    return fn, params, "HPSv3 (Qwen2-VL 7B)", "PF", "MizzenAI/HPSv3 (Qwen2-VL 7B)"


# ---- registry --------------------------------------------------------------

METRICS = [
    # (metric_id, loader, group)
    ("clipi",            _load_clipi,           "CP"),
    ("dinoi",            _load_dinoi,           "CP"),
    ("dinov3",           _load_dinov3,          "CP"),
    ("dreamsim",         _load_dreamsim,        "CP"),
    ("radio_summary",    _load_radio_summary,   "CP"),
    ("dift_canonical",   _load_dift_canonical,  "CP"),
    ("siglip2_global_cp",_load_siglip2_global,  "CP"),
    ("ours_cp",          _load_ours_cp,         "CP"),
    ("clipt",            _load_clipt,           "PF"),
    ("siglip2_global_pf",_load_siglip2t_global, "PF"),
    ("ours_pf",          _load_ours_pf,         "PF"),
    ("vqascore",         _load_vqascore,        "PF"),
    ("imagereward",      _load_imagereward,     "PF"),
    ("hpsv3",            _load_hpsv3,           "PF"),
]


def _bench_one(metric_id: str, warmup: int, runs: int) -> dict:
    loader = dict((mid, l) for mid, l, _ in METRICS)[metric_id]
    fn, params, display_name, table, notes = loader()
    timings_ms, peak_MiB = _time_loop(fn, warmup=warmup, runs=runs)

    median = statistics.median(timings_ms)
    mean = statistics.mean(timings_ms)
    stdev = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
    return {
        "metric_id": metric_id,
        "display_name": display_name,
        "table": table,
        "kind": "non-LLM",
        "params_M": params / 1e6,
        "peak_mem_MiB": peak_MiB,
        "latency_ms_median": median,
        "latency_ms_mean": mean,
        "latency_ms_stdev": stdev,
        "throughput_pairs_per_s": 1000.0 / median if median > 0 else 0.0,
        "n_runs": runs,
        "notes": notes,
    }


def _format_md_table(rows: list[dict]) -> str:
    """Markdown table sorted by table (CP first), then by latency."""
    sorted_rows = sorted(rows, key=lambda r: (r["table"] != "CP", r["latency_ms_median"]))
    lines = [
        "| metric | table | params (M) | peak GPU (MiB) | latency (ms / pair) | throughput (pairs/s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in sorted_rows:
        marker = "**" if "Ours" in r["display_name"] else ""
        name = f"{marker}{r['display_name']}{marker}"
        lat = f"{r['latency_ms_median']:.1f} ± {r['latency_ms_stdev']:.1f}"
        lines.append(
            f"| {name} | {r['table']} | "
            f"{r['params_M']:.0f} | {r['peak_mem_MiB']:.0f} | "
            f"{lat} | {r['throughput_pairs_per_s']:.1f} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="Run only this metric_id (worker mode).")
    ap.add_argument("--skip", type=str, default="",
                    help="Comma-separated list of metric_ids to skip.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    # Worker mode: bench one metric, print result as a single JSON line on
    # stdout, then exit.
    if args.only:
        # Validate sample paths exist
        for k in ("ref_image", "ref_mask", "out_image", "out_mask"):
            p = SAMPLE_PAIR[k]
            if not Path(p).exists():
                print(f"FATAL: sample {k} missing: {p}", file=sys.stderr)
                return 2
        try:
            row = _bench_one(args.only, warmup=args.warmup, runs=args.runs)
        except Exception as e:
            print(f"FAIL: {args.only}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        print("BENCH_RESULT_JSON " + json.dumps(row))
        return 0

    # Driver mode: re-exec self once per metric to get a clean process.
    skip = {m.strip() for m in args.skip.split(",") if m.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Truncate the output file at the start of a full run.
    if args.out.exists():
        args.out.unlink()

    rows: list[dict] = []
    for mid, _, _ in METRICS:
        if mid in skip:
            print(f"[skip] {mid}")
            continue
        print(f"[bench] {mid} ...", flush=True)
        env = os.environ.copy()
        # Avoid runaway tokenizer thread parallelism inside a worker.
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        cmd = [sys.executable, str(__file__), "--only", mid,
               "--warmup", str(args.warmup), "--runs", str(args.runs)]
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=1200)
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {mid}")
            continue
        if proc.returncode != 0:
            print(f"  [FAIL] {mid}\n  stderr: {proc.stderr.strip()[-1000:]}")
            continue
        row = None
        for line in proc.stdout.splitlines():
            if line.startswith("BENCH_RESULT_JSON "):
                row = json.loads(line[len("BENCH_RESULT_JSON "):])
                break
        if row is None:
            print(f"  [FAIL] {mid}: no BENCH_RESULT_JSON in stdout")
            print(f"  stdout tail: {proc.stdout.strip()[-500:]}")
            continue
        rows.append(row)
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"  -> latency = {row['latency_ms_median']:.1f} ms,  "
            f"throughput = {row['throughput_pairs_per_s']:.1f} pairs/s,  "
            f"peak = {row['peak_mem_MiB']:.0f} MiB"
        )

    print()
    print(_format_md_table(rows))
    print(f"\nWrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
