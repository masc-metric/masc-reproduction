---
title: "MaSC camera-ready - DreamBench++ comparison bundle (machine-readable)"
created: 2026-10-17
updated: 2026-10-17
tags: [masc, camera-ready, bootstrap, neurips-2026]
related:
  - "[[Projects/masc/camera-ready-ci-coverage-assessment]]"
  - "[[Projects/masc/submissions/neurips-2026]]"
---

# DreamBench++ comparison bundle

Input for the approved camera-ready revisions. Only aggregates and key ids: no ratings or prompt text. ORIDa is
excluded; its node-bootstrap results stay on hold (see the end of this README).

## Files

| File | Content |
|---|---|
| `dbpp_comparisons.json` | everything: `meta` (resampling, seeds, statistics, input and code sha256, originally printed values), `analyses`, `rows`, `comparisons`, `duplicate_groups` |
| `dbpp_comparisons.csv` | one row per paired difference (flat copy of `comparisons`) |
| `dbpp_rows.csv` | one row per metric and statistic: point, subject-cluster and key 95% intervals |
| `dbpp_analyses.csv` | one row per analysis: key set definition, size, subjects, seed labels, key-set sha256, manuscript target |
| `dbpp_keysets.json.gz` | `{keyset_sha256: sorted ["<method>|<concept_id>-<prompt_id>", ...]}` for every analysis |
| `code/` | `cluster_ci.py` (bootstrap run, SLURM 9155078), `export.py` (this bundle, SLURM 9171498), `render.py`; copies of the git-excluded originals in `release/camera_ready_analysis/cluster_ci/` |

## Rules for using it

- **Quote only records with `canonical: true`.** Records with identical key set, score columns and statistic are the
  same comparison computed in different bootstrap runs. Their intervals differ only by Monte Carlo error (for
  example, MaSC - DreamSim at 5%: [-0.008, +0.082] canonical vs [-0.006, +0.082] in the Table 1 replication).
  Canonical source priority: filter > strata > stripping > boundary > table1 > table3 > table4. Duplicate groups:
  MaSC - DreamSim (filter 5% = Table 1 replication), MaSC - full/full PF (strata "all" = Table 3 global pool =
  Table 4 full x full), Regex - none (stripping = Table 4 BG x full), dependency vs lexical "attribute" strata.
- **Primary interval: `subject_cluster_ci95`; secondary: `key_ci95`.** `*_resolved` = the interval excludes 0.
  Unresolved differences are worded "not resolved under subject-cluster resampling".
- **Intervals describe the replication** (regenerated masks and scores, key sets in `dbpp_keysets.json.gz`). They
  are never attached to the printed main-table values, which `meta.original_printed` keeps for reference with
  why no interval attaches (submission-time masks and key lists not retained; Table 4 FG x full -0.108 was a
  6-method value).
- Stripping and boundary records combine the published key interval (seed labels as printed in the appendix) with
  the subject-cluster interval from the cluster run. Their point estimates agree to 1e-9 (asserted).

## Resampling (exact; also in `meta`)

- B = 10,000; 95% percentile interval, `numpy.nanpercentile(..., [2.5, 97.5])`.
- RNG: `default_rng(SeedSequence([20261106, zlib.crc32(label.encode())]))`; cluster runs use `label + "|cluster"`.
- Subject cluster = one DreamBench++ reference image (`<category>/<index>`; 63 keys each). Per replicate:
  `draw = rng.integers(0, n_clusters, size=n_clusters)` over the sorted unique subjects; the replicate holds every
  key of each drawn subject once per draw.
- Key bootstrap: `rng.integers(0, n, size=(250, n))` index chunks.
- Paired: all rows of an analysis share each replicate's indices; differences are taken replicate by replicate.
- Kd_o alpha: mean over the two rating groups of interval Krippendorff alpha; continuous rows min-max normalised on
  the resampled rows in every replicate and compared with ratings / 4; GPT 0-4 rows raw. Spearman rho against the
  mean of both groups, average ranks, recomputed per replicate.
- Verified (self-tests in the cluster run): duplicated subjects keep their multiplicity; per-replicate alpha, rho
  and human ranks equal an independent recomputation on the materialised resample to 1e-9.
- Environment: release commit `5d52d99` (tracked code unmodified), Python 3.12, numpy 1.26.4, scipy 1.16.3.

## ORIDa (on hold, not in this bundle)

Node (two-way subject) bootstrap results are in `release/camera_ready_analysis/cluster_ci/out/cluster_ci.json`
under `orida` (pair-level, anchor-subject and node variants). Weights: within pair of subject s gets c_s, cross pair
(a, b) gets c_a * c_b, where c are the subject draw counts. Assumptions and limitations (objects are the first 50
ids, treated as a sample; cross-pair subsampling and background choice are held fixed; coverage not validated):
section H of [[Projects/masc/camera-ready-ci-coverage-assessment]]. No ORIDa significance claim until reviewed.
