"""Latency bench for the three PF baselines that won't import in the main
`.venv` (VQAScore / ImageReward / HPSv3).

These wrap older HF stacks (lavis / image-reward / Qwen2-VL trainer) that
conflict with `transformers>=5.6` (required for SigLIP2 NaFlex). They run
in a sibling `.venv-pf-baselines` pinned to `transformers==4.45.2` plus
small import-time stubs for video-only deps that aren't on the per-pair
inference path.

Usage:
    # one-time env bootstrap (see top of this file for pip lines).
    .venv-pf-baselines/bin/python scripts/benchmark_speed_pf_baselines.py

Each row is written as a single line to results/benchmark_speed.jsonl,
matching the schema produced by scripts/benchmark_speed.py.

Bootstrap (run once, from a clean python3.11):
    python3 -m venv .venv-pf-baselines
    .venv-pf-baselines/bin/pip install --upgrade pip "setuptools<80"
    .venv-pf-baselines/bin/pip install torch==2.5.1 torchvision \
        --index-url https://download.pytorch.org/whl/cu124
    .venv-pf-baselines/bin/pip install --no-build-isolation "t2v-metrics==1.1"
    .venv-pf-baselines/bin/pip install --no-build-isolation "image-reward"
    .venv-pf-baselines/bin/pip install openai-clip ftfy
    .venv-pf-baselines/bin/pip install hpsv3 --no-deps
    .venv-pf-baselines/bin/pip install tensorboard imageio_ffmpeg
    .venv-pf-baselines/bin/pip uninstall -y apex torchcodec        # video / audio deps not on inference path
    .venv-pf-baselines/bin/pip install --no-deps "trl==0.9.6" \
        "transformers==4.45.2" "tokenizers==0.20.3" "fire>=0.7.0" \
        "peft>=0.8.0" "huggingface_hub<1.0"
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import types
from importlib.machinery import ModuleSpec


# ---- compat shims ----------------------------------------------------------
# t2v-metrics 1.1 imports a video-frame Flash-Attention kernel from
# `flash_attn`, which we don't have wheels for. The video models aren't on
# the inference path; install no-op packages so the package imports cleanly.

def _make_pkg(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__path__ = []
    spec = ModuleSpec(name, loader=None)
    spec.submodule_search_locations = []
    m.__spec__ = spec
    return m


def _stub_flash_attn() -> None:
    for n in ("flash_attn", "flash_attn.flash_attn_interface",
             "flash_attn.modules", "flash_attn.modules.mha",
             "flash_attn.bert_padding"):
        sys.modules.setdefault(n, _make_pkg(n))

    def _raise(*_a, **_kw):
        raise NotImplementedError("flash_attn is stubbed")

    sys.modules["flash_attn.flash_attn_interface"].flash_attn_varlen_qkvpacked_func = _raise
    sys.modules["flash_attn.modules.mha"].FlashSelfAttention = type("F", (), {})
    sys.modules["flash_attn.bert_padding"].pad_input = _raise
    sys.modules["flash_attn.bert_padding"].unpad_input = _raise


def _stub_torchcodec() -> None:
    if "torchcodec" not in sys.modules:
        sys.modules["torchcodec"] = _make_pkg("torchcodec")


def _ffmpeg_shim() -> None:
    """t2v-metrics asserts `which ffmpeg` at import time. The host has no
    ffmpeg, but `imageio-ffmpeg` ships a static binary; symlink it as
    `ffmpeg` on PATH."""
    if shutil.which("ffmpeg") is not None:
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        return
    ffbin = imageio_ffmpeg.get_ffmpeg_exe()
    shim_dir = os.path.join(tempfile.gettempdir(), "masc_repro_ffmpeg_shim")
    os.makedirs(shim_dir, exist_ok=True)
    shim = os.path.join(shim_dir, "ffmpeg")
    if not os.path.exists(shim):
        os.symlink(ffbin, shim)
    os.environ["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")


_stub_flash_attn()
_stub_torchcodec()
_ffmpeg_shim()


import torch
from PIL import Image


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_PAIR = {
    "ref_image": os.path.join(REPO, "data/dreambench_plus/images/object/00.jpg"),
    "out_image": os.path.join(
        REPO,
        "data/dreambench_plus/samples/dreambooth_sd_gs7_5_step100_seed42_torch_float16/"
        "tgt_image/object_00_motorcycle/0_0.jpg",
    ),
    "prompt": (
        "A photograph of a motorcycle parked beside a bustling city "
        "street at night, illuminated by street lights"
    ),
}
OUT_PATH = os.path.join(REPO, "results/benchmark_speed.jsonl")


def _params_total(obj) -> int:
    """Recursive sum of `numel()` over every nn.Module attached to `obj`."""
    seen: set[int] = set()
    total = 0

    def visit(o, depth=0):
        nonlocal total
        if depth > 4 or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, torch.nn.Module):
            total += sum(p.numel() for p in o.parameters())
            return
        if hasattr(o, "__dict__"):
            for v in vars(o).values():
                visit(v, depth + 1)

    visit(obj)
    return total


def _bench(call_fn, warmup: int = 5, runs: int = 20) -> tuple[float, float, float, float]:
    """Returns (median_ms, mean_ms, stdev_ms, peak_MiB)."""
    with torch.no_grad():
        for _ in range(warmup):
            _ = call_fn()
            torch.cuda.empty_cache()
            gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = call_fn()
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)
            torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return (
        statistics.median(timings),
        statistics.mean(timings),
        statistics.stdev(timings),
        peak,
    )


def _row(metric_id, display_name, params, lat_med, lat_mean, lat_std, peak, notes):
    return {
        "metric_id": metric_id,
        "display_name": display_name,
        "table": "PF",
        "kind": "non-LLM",
        "params_M": params / 1e6,
        "peak_mem_MiB": peak,
        "latency_ms_median": lat_med,
        "latency_ms_mean": lat_mean,
        "latency_ms_stdev": lat_std,
        "throughput_pairs_per_s": 1000.0 / lat_med if lat_med > 0 else 0.0,
        "n_runs": 20,
        "notes": notes,
    }


def bench_imagereward() -> dict:
    print("[imagereward] loading...")
    import ImageReward as RM
    m = RM.load("ImageReward-v1.0", device="cuda")
    img = Image.open(SAMPLE_PAIR["out_image"]).convert("RGB")

    def call_fn():
        return m.score(SAMPLE_PAIR["prompt"], img)

    params = sum(p.numel() for p in m.parameters())
    lat_med, lat_mean, lat_std, peak = _bench(call_fn)
    print(f"  median={lat_med:.1f} ms  peak={peak:.0f} MiB  params={params/1e6:.1f} M")
    return _row(
        "imagereward",
        "ImageReward (BLIP fine-tune)",
        params, lat_med, lat_mean, lat_std, peak,
        "ImageReward-v1.0 (BLIP); benched in .venv-pf-baselines (transformers==4.45.2)",
    )


def bench_vqascore() -> dict:
    print("[vqascore] loading clip-flant5-xxl (~11.5B, ~22 GiB)...")
    import t2v_metrics
    m = t2v_metrics.VQAScore(model="clip-flant5-xxl", device="cuda")

    def call_fn():
        return m(images=[SAMPLE_PAIR["out_image"]], texts=[SAMPLE_PAIR["prompt"]])

    params = _params_total(m.model)
    lat_med, lat_mean, lat_std, peak = _bench(call_fn)
    print(f"  median={lat_med:.1f} ms  peak={peak:.0f} MiB  params={params/1e6:.1f} M")
    return _row(
        "vqascore",
        "VQAScore (clip-flant5-xxl)",
        params, lat_med, lat_mean, lat_std, peak,
        "clip-flant5-xxl (~11.5B); benched in .venv-pf-baselines "
        "(transformers==4.45.2, t2v-metrics==1.1)",
    )


def bench_hpsv3() -> dict:
    print("[hpsv3] loading Qwen2-VL-7B reward model (~8.3B, ~16 GiB)...")
    from hpsv3 import HPSv3RewardInferencer
    m = HPSv3RewardInferencer(device="cuda")

    img = Image.open(SAMPLE_PAIR["out_image"]).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        img.save(tf, format="PNG")
        path = tf.name

    def call_fn():
        return m.reward([path], [SAMPLE_PAIR["prompt"]])

    try:
        params = _params_total(m)
        lat_med, lat_mean, lat_std, peak = _bench(call_fn)
    finally:
        os.unlink(path)
    print(f"  median={lat_med:.1f} ms  peak={peak:.0f} MiB  params={params/1e6:.1f} M")
    return _row(
        "hpsv3",
        "HPSv3 (Qwen2-VL 7B)",
        params, lat_med, lat_mean, lat_std, peak,
        "MizzenAI/HPSv3 (Qwen2-VL 7B); benched in .venv-pf-baselines "
        "(transformers==4.45.2)",
    )


def main() -> int:
    rows: list[dict] = []
    for fn in (bench_imagereward, bench_vqascore, bench_hpsv3):
        try:
            rows.append(fn())
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            continue

    # Append (don't overwrite — the main `.venv` produces 11 rows already).
    with open(OUT_PATH, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nAppended {len(rows)} rows to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
