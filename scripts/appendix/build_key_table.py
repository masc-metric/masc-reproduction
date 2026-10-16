"""Build the DreamBench++ per-key table that the appendix analyses read.

    python scripts/appendix/build_key_table.py

Joins, for all 9,450 (method, concept, prompt) pairs: both human rating
groups, the SAM3 mask foreground fractions, the unfiltered scoring runs
(README, "Appendix: scoring runs") and the GPT-4o / GPT-4V ratings shipped with
DreamBench++. Writes results/analysis/dbpp_keys.csv.gz, which carries
DreamBench++ ratings and prompts and therefore stays local (results/ is
git-ignored).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repro_eval.analysis import keytable  # noqa: E402


def main() -> int:
    df = keytable.build()
    keytable.KEY_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(keytable.KEY_TABLE, index=False)
    print(f"Wrote {len(df):,} pairs to {keytable.KEY_TABLE}")
    cols = [
        *keytable.CP_RUNS,
        *keytable.SHIPPED_CP,
        *keytable.PF_RUNS,
        "cp_g1",
        "cp_g2",
        "pf_g1",
        "pf_g2",
        "fg_ref",
        "fg_out",
    ]
    for c in cols:
        print(f"  {c:16s} {int(df[c].notna().sum()):>5,} pairs with a value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
