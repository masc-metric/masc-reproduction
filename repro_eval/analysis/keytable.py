"""DreamBench++ per-key table read by every appendix analysis.

One row per (method, concept, prompt) pair, 9,450 in total: both human rating
groups for CP and PF, the foreground fraction of the reference and output
SAM3 masks, the scores of the unfiltered (`_nofilter`) runs, and the GPT-4o /
GPT-4V ratings shipped with DreamBench++. Every analysis selects its matched
key set from this table, so all metrics of a table are evaluated on identical
keys.

The table carries DreamBench++ human ratings and prompt text, which
DreamBench++ releases without a licence for redistribution. It is written
under the git-ignored `results/` folder and must not be published.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from repro_eval.data import (
    DREAMBENCHPLUS_DIR,
    DREAMBENCHPLUS_METHOD_ALIASES,
    MASKS_DIR,
    RESULTS_DIR,
    parse_dreambenchplus_subject,
)

METHODS = list(DREAMBENCHPLUS_METHOD_ALIASES)
RATINGS = DREAMBENCHPLUS_DIR / "data_human_rating" / "merged_data"
KEY_TABLE = RESULTS_DIR / "analysis" / "dbpp_keys.csv.gz"
SAM3_MASKS = MASKS_DIR / "dreambench_plus"

# column -> (model tag, run tag without the method suffix), as written by
# compute_geometry.py / compute_prompt_following.py with --tag-suffix _nofilter
CP_RUNS = {
    "masc": ("siglip2-so400m", "dreambenchplus_maskedmaxcos_naflex_nofilter"),
    "siglip2_global": ("siglip2-so400m", "dreambenchplus_global_naflex_nofilter"),
    "dreamsim": ("dreamsim-ensemble", "dreambenchplus_nofilter"),
    "dinov3": ("dinov3-vitl16", "dreambenchplus_dbplus_nofilter"),
}
PF_RUNS = {
    "pf_masc": (
        "siglip2-so400m",
        "dreambenchplus_global_bg_nosubj_nofilter",
    ),  # BG pool x regex strip
    "pf_full_full": ("siglip2-so400m", "dreambenchplus_naflex_nofilter"),  # full pool x full prompt
    "pf_bg_full": ("siglip2-so400m", "dreambenchplus_global_bg_nofilter"),  # BG pool x full prompt
    "pf_bg_spacy": (
        "siglip2-so400m",
        "dreambenchplus_global_bg_nosubj_nofilter_spacy",
    ),  # BG pool x spaCy
}
# column -> folder under data/dreambench_plus (ratings on the 0-4 rubric)
SHIPPED_CP = {
    "gpt4o": "data_gpt_rating/concept_preservation_wo_internal_thinking",
    "gpt4v": "data_gpt_rating/concept_preservation_gpt4v_full",
}


def key_of(concept_id: str, prompt_id) -> str:
    return f"{concept_id}-{prompt_id}"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut by a killed job
    return rows


def result_path(signal: str, model: str, tag: str, method: str) -> Path:
    return RESULTS_DIR / f"{signal}__{model}__{tag}_{method}.jsonl"


def _scores(signal: str, model: str, tag: str, method: str) -> dict[str, float]:
    path = result_path(signal, model, tag, method)
    if not path.exists():
        raise FileNotFoundError(f"{path} (see the README, 'Appendix: scoring runs')")
    out = {}
    for r in read_jsonl(path):
        s = r.get("score")
        out[key_of(r["concept_id"], r["prompt_id"])] = float("nan") if s is None else float(s)
    return out


def _shipped(folder: str, method: str) -> dict[str, float]:
    p = DREAMBENCHPLUS_DIR / folder / f"{method}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return {k: float(v) for k, v in json.load(p.open()).items()}


def _ratings(method: str, signal: str, group: str) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in json.load((RATINGS / group / f"{method}-{signal}.json").open()).items()
    }


def mask_fraction(path: Path) -> float:
    if not path.exists():
        return float("nan")
    m = np.asarray(Image.open(path))
    return float((m > 0).sum()) / float(m.size)


def skeleton() -> pd.DataFrame:
    """All 9,450 (method, concept, prompt) pairs with prompt text and the
    canonical object name (first caption line, as the PF scripts use)."""
    rows = []
    for method, full in DREAMBENCHPLUS_METHOD_ALIASES.items():
        tgt = DREAMBENCHPLUS_DIR / "samples" / full / "tgt_image"
        for subj in sorted(p for p in tgt.iterdir() if p.is_dir()):
            cat, idx, keyword = parse_dreambenchplus_subject(subj.name)
            cap = DREAMBENCHPLUS_DIR / "captions" / cat / f"{idx}.txt"
            obj = keyword.replace("_", " ")
            if cap.exists():
                first = cap.read_text().split("\n", 1)[0].strip()
                obj = first or obj
            for img in sorted(subj.iterdir()):
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                prompt_id, seed = img.stem.split("_", 1)
                txt = DREAMBENCHPLUS_DIR / "samples" / full / "text" / subj.name / f"{img.stem}.txt"
                rows.append(
                    {
                        "method": method,
                        "concept_id": subj.name,
                        "prompt_id": prompt_id,
                        "seed": int(seed),
                        "key": key_of(subj.name, prompt_id),
                        "category": str(cat),
                        "subject_idx": idx,
                        "keyword": keyword,
                        # One id per reference image: DreamBench++ names object 20
                        # "magic_cube" in six method folders and "rubik's_cube" in emu2.
                        "subject": f"{cat}/{idx}",
                        "object_name": obj,
                        "prompt": txt.read_text().strip() if txt.exists() else "",
                        "is_style": str(cat).startswith("style"),
                        "out_image": str(img.relative_to(DREAMBENCHPLUS_DIR)),
                        "ref_image": str(Path("images") / cat / f"{idx}.jpg"),
                    }
                )
    df = pd.DataFrame(rows)
    if len(df) != 9450 or df.duplicated(["method", "key"]).any():
        raise RuntimeError(f"expected 9,450 unique pairs, got {len(df)}")
    return df


def ref_mask_path(row) -> Path:
    cat = Path(row.ref_image).parent.relative_to("images")
    return SAM3_MASKS / "refs" / cat / f"{row.subject_idx}.png"


def out_mask_path(row) -> Path:
    full = DREAMBENCHPLUS_METHOD_ALIASES[row.method]
    return SAM3_MASKS / "samples" / full / row.concept_id / f"{row.prompt_id}_{row.seed}.png"


def add_mask_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Foreground pixel fraction of each pair's SAM3 masks (`fg_ref`, `fg_out`)."""
    ref_cache: dict[Path, float] = {}
    fr_ref, fr_out = [], []
    for r in df.itertuples():
        rp = ref_mask_path(r)
        if rp not in ref_cache:
            ref_cache[rp] = mask_fraction(rp)
        fr_ref.append(ref_cache[rp])
        fr_out.append(mask_fraction(out_mask_path(r)))
    df["fg_ref"] = fr_ref
    df["fg_out"] = fr_out
    return df


def add_ratings_and_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Join both human rating groups, the scoring runs and the shipped GPT
    ratings by (method, key)."""
    columns = ["cp_g1", "cp_g2", "pf_g1", "pf_g2", *CP_RUNS, *SHIPPED_CP, *PF_RUNS]
    for c in columns:
        df[c] = np.nan
    for method in METHODS:
        sel = df.method == method
        src = {
            "cp_g1": _ratings(method, "cp", "group1"),
            "cp_g2": _ratings(method, "cp", "group2"),
            "pf_g1": _ratings(method, "pf", "group1"),
            "pf_g2": _ratings(method, "pf", "group2"),
        }
        for c, (model, tag) in CP_RUNS.items():
            src[c] = _scores("geometry", model, tag, method)
        for c, (model, tag) in PF_RUNS.items():
            src[c] = _scores("prompt_following", model, tag, method)
        for c, folder in SHIPPED_CP.items():
            src[c] = _shipped(folder, method)
        for c, d in src.items():
            df.loc[sel, c] = df.loc[sel, "key"].map(d).astype(float).to_numpy()
        # MaSC CP is undefined when the reference mask has no foreground patch
        # on the SigLIP2 grid (the scorer then writes 0.0); keep the count.
        nfg = {
            key_of(r["concept_id"], r["prompt_id"]): r["extras"].get("n_ref_fg_only")
            for r in read_jsonl(result_path("geometry", *CP_RUNS["masc"], method))
        }
        df.loc[sel, "masc_n_ref_fg_patches"] = df.loc[sel, "key"].map(nfg).astype(float).to_numpy()
    return df


def build() -> pd.DataFrame:
    return add_ratings_and_scores(add_mask_fractions(skeleton()))


def read_keys(path: Path = KEY_TABLE) -> pd.DataFrame:
    """Read the per-key table with identifier columns kept as strings
    (subject_idx "00" must not become 0)."""
    if not path.exists():
        raise FileNotFoundError(f"{path}: run scripts/appendix/build_key_table.py first")
    return pd.read_csv(
        path,
        dtype={"subject_idx": str, "prompt_id": str, "concept_id": str, "key": str, "subject": str},
    )


def retained(df: pd.DataFrame, t: float) -> pd.Series:
    """The mask-area filter: both SAM3 masks cover more than `t` of their image
    (the scoring code drops a pair when either fraction is <= t)."""
    return (df.fg_ref > t) & (df.fg_out > t)


def load_patch_arrays(
    model: str = "siglip2-so400m", tag: str = "dreambenchplus_maskedmaxcos_naflex_nofilter"
) -> dict:
    """{(method, key): (nn_cos_per_ref float32[Hp*Wp], (Hp, Wp))} from the
    `--dump-patch-arrays` sidecar files."""
    out = {}
    for method in METHODS:
        p = RESULTS_DIR / "aux" / f"geometry__{model}__{tag}_{method}.patch_arrays.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"{p} (score MaSC with --dump-patch-arrays)")
        for r in read_jsonl(p):
            out[(method, key_of(r["concept_id"], r["prompt_id"]))] = (
                decode_arr(r),
                tuple(r["ref_grid"]),
            )
    return out


def decode_arr(r: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(r["nn_cos_per_ref_f32_b64"]), dtype="<f4")
