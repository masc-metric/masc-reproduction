# Camera-ready statistical artifacts

`masc-camera-ready-stats-2026-10-17/` is a byte-preserved copy of the bundle
supplied by the authors. Its source ZIP SHA-256 was
`d574c5c181551040a5ca4c6d832b596ffb73bb1d8db519460a01dbd175e8f17d`
(277,675 bytes); the reported vault source commit was `ac39244`.
The ZIP itself is not duplicated here. Internal `SHA256SUMS` records remain
unchanged. The exported metadata identifies reproduction commit `5d52d99`
and hashes the statistical code and inputs.

## Verify without data, GPU, or API calls

From the repository root:

```bash
python scripts/verify_camera_ready_bundle.py
```

This checks the file manifest, code hashes against this checkout, record
counts, exact key-set hashes and sizes, subject counts, and the JSON/CSV
exports. It does not recompute the statistics or verify unavailable input
predictions. The bundle contains aggregate statistics and key IDs, not the
underlying human ratings, prompts, masks, predictions, or cluster logs.

## Interpretation

- Quote only records with `canonical: true`.
- Subject-cluster intervals are primary; individual-key intervals secondary.
- The intervals describe the supplementary evaluation with regenerated
  masks, not the original main-table estimates.
- An interval containing zero does not establish equivalence.
- ORIDa node-bootstrap results remain on hold and are not in this bundle.
- The original README's vault wikilinks are internal provenance references,
  not public dependencies. Its ORIDa discussion records unresolved work,
  not an approved inferential protocol.

## Source snapshots versus a complete rerun

`code/cluster_ci.py`, `export.py`, and `render.py` are unmodified snapshots
from the original `camera_ready_analysis/cluster_ci/` directory. They are
not standalone CLIs at their archived location: their repository-relative
paths expect that original layout. `render.py` reads the intermediate
`cluster_ci.json`, not the exported `dbpp_comparisons.json`. Do not invoke
these snapshots expecting a data-free rerun.

The existing `scripts/appendix/` instructions regenerate the earlier
appendix analyses, not the entire subject-cluster bundle. A complete bundle
rerun additionally requires the original unfiltered scoring runs (including
the extra Table 1/3/4 controls), patch arrays, the joined key table, and the
intermediate bootstrap outputs used by `export.py`. Those inputs are hashed
in the metadata but are not supplied in this archive. Restoring that runnable
workflow and obtaining permission to redistribute any restricted inputs are
still release tasks; checksums alone do not complete them.
