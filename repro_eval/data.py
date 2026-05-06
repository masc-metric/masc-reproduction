from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = DATA_DIR / "outputs"
MASKS_DIR = DATA_DIR / "masks"
RESULTS_DIR = REPO_ROOT / "results"

DREAMBENCHPLUS_DIR = DATA_DIR / "dreambench_plus"
ORIDA_DIR = DATA_DIR / "ORIDa"

# Short aliases → sample directory names under
# `data/dreambench_plus/samples/`. The short form matches the human
# rating file prefixes in `data_human_rating/merged_data/group*/`, so
# the same alias keys both the generations on disk and the ratings.
DREAMBENCHPLUS_METHOD_ALIASES = {
    "textual_inversion_sd":      "textual_inversion_sd_gs7_5_step100_seed42_torch_float16",
    "dreambooth_sd":             "dreambooth_sd_gs7_5_step100_seed42_torch_float16",
    "dreambooth_lora_sdxl":      "dreambooth_lora_sdxl_gs7_5_step100_seed42_torch_float16",
    "blip_diffusion":            "blip_diffusion_gs7_5_step100_seed42_torch_float16",
    "emu2":                      "emu2_gs3_step50_seed42_torch_bfloat16",
    "ip_adapter_plus_vit_h_sdxl": "ip_adapter_plus_vit_huge_sdxl_scale0_6_gs7_5_step100_seed42_torch_float16",
    "ip_adapter_vit_g_sdxl":     "ip_adapter_vit_giant_sdxl_scale0_6_gs7_5_step100_seed42_torch_float16",
}


def parse_dreambenchplus_subject(subject_dir: str) -> tuple[Path, str, str]:
    """`live_subject_animal_00_kitten` -> (Path("live_subject/animal"), "00", "kitten").

    Handles the 4 top-level categories explicitly: the `live_subject_*`
    ones have a 2-part category path, `object_*` and `style_*` have 1.
    """
    for prefix, cat_path in (
        ("live_subject_animal_", Path("live_subject/animal")),
        ("live_subject_human_",  Path("live_subject/human")),
        ("object_",              Path("object")),
        ("style_",               Path("style")),
    ):
        if subject_dir.startswith(prefix):
            rest = subject_dir[len(prefix):]
            idx, keyword = rest.split("_", 1)
            return cat_path, idx, keyword
    raise ValueError(f"Unrecognized DreamBench++ subject dir: {subject_dir!r}")


def mask_nonempty(mask_path: Path) -> bool:
    """True iff the PNG at `mask_path` contains at least one non-zero pixel."""
    return bool(np.asarray(Image.open(mask_path)).any())


def mask_fg_fraction(mask_path: Path) -> float:
    """Fraction of the mask's pixels that are foreground (non-zero)."""
    m = np.asarray(Image.open(mask_path))
    return float((m > 0).sum()) / float(m.size)


def build_dreambenchplus_samples(
    images_root: Path,
    masks_root: Path,
    image_exts: set[str],
    *,
    method: str | None = None,
    samples_root: Path | None = None,
    min_fg_fraction: float = 0.05,
) -> list[dict]:
    """Pair each DreamBench++ generation from `method` with its reference.

    Expected disk layout (from the DreamBench++ README):

        samples_root / <method_full_name> /
            src_image/<subject>/<j>_<k>.jpg       # ref duplicated
            tgt_image/<subject>/<j>_<k>.jpg       # generation
            text/<subject>/<j>_<k>.txt            # prompt

    where `<subject>` is e.g. `live_subject_animal_00_kitten` and
    `<j>_<k>` is `<prompt_idx>_<seed_idx>`. References come from
    `images_root/<category_path>/<idx>.jpg` (images_root points at
    `data/dreambench_plus/images/`). Reference masks are shared across
    methods under `masks_root` (e.g. `data/masks/dreambench_plus/refs/`);
    output masks live under the method-scoped
    `data/masks/dreambench_plus/samples/<method_full_name>/` tree, which
    is resolved here from `masks_root.parent / "samples" / <method>`.

    `method` is a short alias (e.g. `dreambooth_sd`) resolved through
    `DREAMBENCHPLUS_METHOD_ALIASES` — the short form is what the human
    rating filenames use, so the same key threads through to validation.
    """
    if method is None:
        raise ValueError("build_dreambenchplus_samples requires --method")
    if samples_root is None:
        samples_root = DREAMBENCHPLUS_DIR / "samples"

    method_full = DREAMBENCHPLUS_METHOD_ALIASES.get(method, method)
    method_dir = samples_root / method_full
    if not method_dir.exists():
        raise FileNotFoundError(
            f"Method directory not found: {method_dir}.\n"
            f"Known aliases: {sorted(DREAMBENCHPLUS_METHOD_ALIASES)}"
        )

    tgt_root = method_dir / "tgt_image"
    text_root = method_dir / "text"
    sample_masks_root = masks_root.parent / "samples" / method_full

    # Drop (ref, out) pairs where either mask covers <= `min_fg_fraction`
    # of the whole image. Tiny masks are usually SAM3 failures and
    # produce noise downstream. Ref fractions are cached since the same
    # ref is reused across all 9 prompts for a subject.
    ref_frac_cache: dict[Path, float] = {}
    def _ref_fg_frac(p: Path) -> float:
        if p not in ref_frac_cache:
            ref_frac_cache[p] = mask_fg_fraction(p)
        return ref_frac_cache[p]

    captions_root = DREAMBENCHPLUS_DIR / "captions"

    samples: list[dict] = []
    n_dropped_ref = n_dropped_out = 0
    for subject_dir in sorted(p for p in tgt_root.iterdir() if p.is_dir()):
        cat_path, idx, keyword = parse_dreambenchplus_subject(subject_dir.name)

        ref_img = images_root / cat_path / f"{idx}.jpg"
        ref_mask = (masks_root / cat_path / f"{idx}").with_suffix(".png")
        if not ref_img.exists() or not ref_mask.exists():
            continue
        if _ref_fg_frac(ref_mask) <= min_fg_fraction:
            n_dropped_ref += sum(1 for _ in subject_dir.iterdir())
            continue

        # DB++'s canonical human-readable object name lives on line 1 of
        # captions/<cat>/<idx>.txt — "piggy bank" not "piggy_bank",
        # "t-shirt" preserved, etc. Falls back to the folder-derived
        # keyword if the captions file is missing.
        object_name = keyword.replace("_", " ")
        captions_file = captions_root / cat_path / f"{idx}.txt"
        if captions_file.exists():
            first_line = captions_file.read_text().split("\n", 1)[0].strip()
            if first_line:
                object_name = first_line

        for out_img in sorted(subject_dir.iterdir()):
            if not out_img.is_file() or out_img.suffix.lower() not in image_exts:
                continue
            stem = out_img.stem  # "<prompt_idx>_<seed_idx>"
            try:
                prompt_idx, seed_idx = stem.split("_", 1)
            except ValueError:
                continue

            out_mask = (sample_masks_root / subject_dir.name / stem).with_suffix(".png")
            if not out_mask.exists():
                continue
            if mask_fg_fraction(out_mask) <= min_fg_fraction:
                n_dropped_out += 1
                continue

            prompt_text = ""
            text_file = text_root / subject_dir.name / f"{stem}.txt"
            if text_file.exists():
                prompt_text = text_file.read_text().strip()

            samples.append({
                "concept_id": subject_dir.name,
                "ref_image": ref_img,
                "ref_mask": ref_mask,
                "out_image": out_img,
                "out_mask": out_mask,
                "gen_method": method,
                "prompt_id": prompt_idx,
                "seed": int(seed_idx) if seed_idx.isdigit() else 0,
                "extras": {
                    "category": str(cat_path),
                    "subject_idx": idx,
                    "keyword": keyword,
                    "object_name": object_name,
                    "prompt_text": prompt_text,
                    "method_full": method_full,
                },
            })

    if n_dropped_ref or n_dropped_out:
        print(
            f"[dreambenchplus:{method}] dropped "
            f"{n_dropped_ref} pairs (ref mask <= {min_fg_fraction:.0%}), "
            f"{n_dropped_out} pairs (out mask <= {min_fg_fraction:.0%})"
        )
    return samples


def build_orida_samples(
    images_root: Path,
    masks_root: Path,  # unused: ORIDa ships paired masks under each subject
    image_exts: set[str],
    *,
    photos_per_subject: int = 8,
    cross_pair_count: int | None = None,
    seed: int = 0,
    min_fg_fraction: float = 0.005,
) -> list[dict]:
    """Within / cross subject pairs over ORIDa (https://arxiv.org/abs/2506.08964).

    `images_root` should point at a split directory (e.g.
    `data/ORIDa/ORIDa_v1.0/validation`) whose immediate children are
    subject folders named by integer (`1/`, `2/`, ...). Each subject
    contains:

      <subj>/factual_only/{images,annotations/masks}/<stem>{,_mask}.jpg
      <subj>/factual_counterfactual/<scene_id>/{images,annotations/masks}/...

    ORIDa ships its own segmentation masks (JPG, [0,255]) alongside each
    image, so `masks_root` is ignored — we resolve mask paths from the
    image path. Each `factual_counterfactual/<scene>` ships 5 images
    (`_0.jpg` ... `_4.jpg`); variant `_0` is the *counterfactual*
    (background only, no object, no shipped mask), so we take only the
    first factual variant (`_1`) per scene to keep each scene_id
    contributing one within-subject sample.

    Per subject we build a sorted, deduped pool of (image, mask) entries
    drawn from factual_only + factual_counterfactual, then pick
    `photos_per_subject` of them with even spread across the pool.
    Within-subject pairs = all C(K, 2) pairs of the picked photos.
    Cross-subject pairs are sampled uniformly across (subj_a < subj_b)
    x (k1 in pool_a) x (k2 in pool_b); count defaults to match within.
    """
    per_subject: list[tuple[str, list[tuple[Path, Path]]]] = []
    for subj_dir in sorted(
        (p for p in images_root.iterdir() if p.is_dir()),
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    ):
        entries: list[tuple[Path, Path]] = []

        fo_imgs = subj_dir / "factual_only" / "images"
        fo_masks = subj_dir / "factual_only" / "annotations" / "masks"
        if fo_imgs.is_dir():
            for img in sorted(fo_imgs.iterdir()):
                if not img.is_file() or img.suffix.lower() not in image_exts:
                    continue
                mask = fo_masks / f"{img.stem}_mask.jpg"
                if not mask.exists():
                    continue
                if mask_fg_fraction(mask) <= min_fg_fraction:
                    continue
                entries.append((img, mask))

        fc_root = subj_dir / "factual_counterfactual"
        if fc_root.is_dir():
            for scene_dir in sorted(p for p in fc_root.iterdir() if p.is_dir()):
                imgs_dir = scene_dir / "images"
                masks_dir = scene_dir / "annotations" / "masks"
                if not imgs_dir.is_dir() or not masks_dir.is_dir():
                    continue
                for img in sorted(imgs_dir.iterdir()):
                    if not img.is_file() or img.suffix.lower() not in image_exts:
                        continue
                    if not img.stem.endswith("_1"):
                        continue
                    mask = masks_dir / f"{img.stem}_mask.jpg"
                    if not mask.exists():
                        continue
                    if mask_fg_fraction(mask) <= min_fg_fraction:
                        continue
                    entries.append((img, mask))
                    break

        if not entries:
            continue

        if len(entries) <= photos_per_subject:
            picked = entries
        else:
            stride = len(entries) / photos_per_subject
            idxs = [int(stride * k) for k in range(photos_per_subject)]
            picked = [entries[i] for i in idxs]

        per_subject.append((subj_dir.name, picked))

    samples: list[dict] = []

    for name_a, pool_a in per_subject:
        for i in range(len(pool_a)):
            for j in range(i + 1, len(pool_a)):
                img_i, mask_i = pool_a[i]
                img_j, mask_j = pool_a[j]
                samples.append({
                    "concept_id": name_a,
                    "ref_image": img_i,
                    "ref_mask": mask_i,
                    "out_image": img_j,
                    "out_mask": mask_j,
                    "gen_method": "orida_within",
                    "prompt_id": "",
                    "seed": 0,
                    "extras": {"subject_a": name_a, "subject_b": name_a},
                })

    n_within = len(samples)
    n_cross_target = cross_pair_count if cross_pair_count is not None else n_within

    rng = np.random.default_rng(seed)
    if len(per_subject) >= 2 and n_cross_target > 0:
        cross_pool = []
        for ai in range(len(per_subject)):
            for bi in range(ai + 1, len(per_subject)):
                name_a, pool_a = per_subject[ai]
                name_b, pool_b = per_subject[bi]
                for ia in range(len(pool_a)):
                    for ib in range(len(pool_b)):
                        cross_pool.append((ai, bi, ia, ib))
        if cross_pool:
            n_pick = min(n_cross_target, len(cross_pool))
            chosen = rng.choice(len(cross_pool), size=n_pick, replace=False)
            for k in chosen:
                ai, bi, ia, ib = cross_pool[int(k)]
                name_a, pool_a = per_subject[ai]
                name_b, pool_b = per_subject[bi]
                img_i, mask_i = pool_a[ia]
                img_j, mask_j = pool_b[ib]
                samples.append({
                    "concept_id": f"{name_a}__vs__{name_b}",
                    "ref_image": img_i,
                    "ref_mask": mask_i,
                    "out_image": img_j,
                    "out_mask": mask_j,
                    "gen_method": "orida_cross",
                    "prompt_id": "",
                    "seed": 0,
                    "extras": {"subject_a": name_a, "subject_b": name_b},
                })

    return samples


def build_orida_diffbg_samples(
    images_root: Path,
    masks_root: Path,  # unused
    image_exts: set[str],
    *,
    backgrounds_per_subject: int | None = 10,
    subjects_max: int | None = 25,
    cross_pair_count: int | None = None,
    seed: int = 0,
    min_fg_fraction: float = 0.005,
    min_backgrounds: int = 2,
) -> list[dict]:
    """One photo per (subject, background) -> all-pairs within-bg-disjoint
    within-subject pairs, matched-count random cross-subject pairs.

    Variant of `build_orida_samples` for the proper "does this metric
    measure identity, not scene similarity" test. The first 6 chars of
    each ORIDa scene_id encode the **background** (a unique physical
    capture environment); the last char of scene_id is the camera-angle
    index. We take the alphabetically first cam angle per background and
    its alphabetically first factual placement (`_1` -- variant `_0` is
    the counterfactual / no-object).

    Used for the paper's ORIDa table: `subjects_max=50`,
    `backgrounds_per_subject=10`, `cross_pair_count=2250` (matches
    50 * C(10, 2) within-subject pairs).
    """
    per_subject: list[tuple[str, list[tuple[Path, Path, str]]]] = []

    for subj_dir in sorted(
        (p for p in images_root.iterdir() if p.is_dir()),
        key=lambda p: int(p.name) if p.name.isdigit() else 10**9,
    ):
        bg_to_entry: dict[str, tuple[Path, Path]] = {}

        fo_imgs = subj_dir / "factual_only" / "images"
        fo_masks = subj_dir / "factual_only" / "annotations" / "masks"
        if fo_imgs.is_dir():
            fo_by_bg: dict[str, list[Path]] = {}
            for img in fo_imgs.iterdir():
                if not img.is_file() or img.suffix.lower() not in image_exts:
                    continue
                parts = img.stem.split("_")
                if len(parts) < 3:
                    continue
                scene_id = parts[1]
                bg = scene_id[:-1]
                fo_by_bg.setdefault(bg, []).append(img)
            for bg, imgs in fo_by_bg.items():
                img = sorted(imgs)[0]
                mask = fo_masks / f"{img.stem}_mask.jpg"
                if not mask.exists():
                    continue
                if mask_fg_fraction(mask) <= min_fg_fraction:
                    continue
                bg_to_entry.setdefault(bg, (img, mask))

        fc_root = subj_dir / "factual_counterfactual"
        if fc_root.is_dir():
            fc_by_bg: dict[str, list[Path]] = {}
            for scene_dir in fc_root.iterdir():
                if not scene_dir.is_dir():
                    continue
                bg = scene_dir.name[:-1]
                fc_by_bg.setdefault(bg, []).append(scene_dir)
            for bg, scenes in fc_by_bg.items():
                if bg in bg_to_entry:
                    continue
                scene_dir = sorted(scenes, key=lambda p: p.name)[0]
                imgs_dir = scene_dir / "images"
                masks_dir = scene_dir / "annotations" / "masks"
                if not imgs_dir.is_dir() or not masks_dir.is_dir():
                    continue
                target_img = imgs_dir / f"00{subj_dir.name.zfill(3)}_{scene_dir.name}_1.jpg"
                if not target_img.exists():
                    target_img = next(
                        (p for p in sorted(imgs_dir.iterdir())
                         if p.suffix.lower() in image_exts and p.stem.endswith("_1")),
                        None,
                    )
                    if target_img is None:
                        continue
                mask = masks_dir / f"{target_img.stem}_mask.jpg"
                if not mask.exists():
                    continue
                if mask_fg_fraction(mask) <= min_fg_fraction:
                    continue
                bg_to_entry[bg] = (target_img, mask)

        required = backgrounds_per_subject if backgrounds_per_subject is not None else min_backgrounds
        if len(bg_to_entry) < required:
            continue

        cap = backgrounds_per_subject if backgrounds_per_subject is not None else len(bg_to_entry)
        chosen_bgs = sorted(bg_to_entry.keys())[:cap]
        pool = [(*bg_to_entry[bg], bg) for bg in chosen_bgs]
        per_subject.append((subj_dir.name, pool))

        if subjects_max is not None and len(per_subject) >= subjects_max:
            break

    samples: list[dict] = []

    for name_a, pool_a in per_subject:
        for i in range(len(pool_a)):
            for j in range(i + 1, len(pool_a)):
                img_i, mask_i, bg_i = pool_a[i]
                img_j, mask_j, bg_j = pool_a[j]
                samples.append({
                    "concept_id": name_a,
                    "ref_image": img_i,
                    "ref_mask": mask_i,
                    "out_image": img_j,
                    "out_mask": mask_j,
                    "gen_method": "orida_within",
                    "prompt_id": "",
                    "seed": 0,
                    "extras": {"subject_a": name_a, "subject_b": name_a,
                               "bg_a": bg_i, "bg_b": bg_j},
                })

    n_within = len(samples)
    n_cross_target = cross_pair_count if cross_pair_count is not None else n_within

    rng = np.random.default_rng(seed)
    if len(per_subject) >= 2 and n_cross_target > 0:
        cross_pool = []
        for ai in range(len(per_subject)):
            for bi in range(ai + 1, len(per_subject)):
                pool_a = per_subject[ai][1]
                pool_b = per_subject[bi][1]
                for ia in range(len(pool_a)):
                    for ib in range(len(pool_b)):
                        cross_pool.append((ai, bi, ia, ib))
        if cross_pool:
            n_pick = min(n_cross_target, len(cross_pool))
            chosen = rng.choice(len(cross_pool), size=n_pick, replace=False)
            for k in chosen:
                ai, bi, ia, ib = cross_pool[int(k)]
                name_a, pool_a = per_subject[ai]
                name_b, pool_b = per_subject[bi]
                img_i, mask_i, bg_i = pool_a[ia]
                img_j, mask_j, bg_j = pool_b[ib]
                samples.append({
                    "concept_id": f"{name_a}__vs__{name_b}",
                    "ref_image": img_i,
                    "ref_mask": mask_i,
                    "out_image": img_j,
                    "out_mask": mask_j,
                    "gen_method": "orida_cross",
                    "prompt_id": "",
                    "seed": 0,
                    "extras": {"subject_a": name_a, "subject_b": name_b,
                               "bg_a": bg_i, "bg_b": bg_j,
                               "same_bg": bg_i == bg_j},
                })

    return samples


SAMPLE_BUILDERS = {
    "dreambenchplus": build_dreambenchplus_samples,
    "orida": build_orida_samples,
    "orida_diffbg": build_orida_diffbg_samples,
}
