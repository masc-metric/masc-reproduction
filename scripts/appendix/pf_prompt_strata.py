"""Appendix "Prompt Following by Prompt Type" (DreamBench++ PF).

    python scripts/appendix/pf_prompt_strata.py [--B 10000]

Compares MaSC PF (background pool, subject-stripped prompt) with the
same-backbone control (full-image pool, original prompt) within prompt
strata. Primary classification: a dependency parse (en_core_web_sm) labels a
prompt action/pose when the subject governs a verbal clause or is the subject
of a verb, and attribute when the subject has a modifier outside its canonical
name or an attached "with" phrase (repro_eval/analysis/strata.py).
Subject-dependent = either; scene-only = neither. Sensitivity: a lexical rule
labels action/pose whenever the prompt contains a non-auxiliary verb.

Each stratum uses the style-excluded keys retained by the 5% filter that fall
into it, with paired key-bootstrap and subject-cluster intervals. Needs
`python -m spacy download en_core_web_sm`. Writes
report/appendix/pf_prompt_strata.json; the per-prompt labels (identifiers
only, no prompt text) go to report/appendix/pf_prompt_strata_labels.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from masc.text import SpacySubjectStripper  # noqa: E402
from repro_eval.analysis.keytable import read_keys, retained  # noqa: E402
from repro_eval.analysis.stats import DEFAULT_B  # noqa: E402
from repro_eval.analysis.strata import classify, classify_lexical  # noqa: E402
from repro_eval.analysis.tables import (  # noqa: E402
    OUT_DIR,
    PaperCheck,
    Row,
    ci3,
    comparison,
    print_table,
    rated_table,
    write_json,
)

MASC, CONTROL = "MaSC", "Full image/full prompt"
ROWS = [Row(MASC, "pf_masc", False, "ours"), Row(CONTROL, "pf_full_full", False, "control")]
LABELS = ["action_pose", "attribute", "subject_dependent", "scene_only"]
STRATA = {  # name -> rows of the key table in the stratum
    "all": lambda d: slice(None),
    "subject_dependent": lambda d: d.subject_dependent,
    "action_pose": lambda d: d.action_pose,
    "attribute": lambda d: d.attribute,
    "both_action_and_attribute": lambda d: d.action_pose & d.attribute,
    "scene_only": lambda d: d.scene_only,
}
LEXICAL = ["subject_dependent", "action_pose", "attribute", "scene_only"]

# Camera-ready tables: stratum -> (keys, MaSC alpha, rho, control alpha, rho, delta rho, [CI])
PAPER = {
    "all": (6605, 0.355, 0.538, 0.326, 0.466, 0.072, (0.057, 0.088)),
    "subject_dependent": (4302, 0.377, 0.555, 0.343, 0.477, 0.078, (0.058, 0.097)),
    "action_pose": (3651, 0.391, 0.567, 0.355, 0.490, 0.077, (0.056, 0.098)),
    "attribute": (991, 0.366, 0.504, 0.302, 0.409, 0.095, (0.058, 0.132)),
    "both_action_and_attribute": (340, 0.436, 0.565, 0.358, 0.450, 0.116, (0.054, 0.178)),
    "scene_only": (2303, 0.310, 0.514, 0.307, 0.445, 0.069, (0.043, 0.096)),
}
PAPER_LEXICAL = {
    "subject_dependent": (5629, 0.371, 0.551, 0.331, 0.469, 0.081, (0.064, 0.098)),
    "action_pose": (5348, 0.379, 0.557, 0.337, 0.473, 0.084, (0.067, 0.101)),
    "attribute": (991, 0.366, 0.504, 0.302, 0.409, 0.095, (0.058, 0.133)),
    "scene_only": (976, 0.278, 0.479, 0.331, 0.443, 0.036, (-0.005, 0.077)),
}
PAPER_ALPHA_DIFFS = {  # (classification, stratum) -> delta alpha, [CI]
    ("dependency", "attribute"): (0.064, (0.009, 0.107)),
    ("dependency", "both_action_and_attribute"): (0.078, (0.023, 0.142)),
    ("lexical", "scene_only"): (-0.053, (-0.096, 0.033)),
}


def label_prompts(df: pd.DataFrame, rule) -> pd.DataFrame:
    stripper = SpacySubjectStripper.load()
    prompts = df[~df.is_style].drop_duplicates(["concept_id", "prompt_id"])
    return pd.DataFrame(
        [
            {
                "concept_id": r.concept_id,
                "prompt_id": r.prompt_id,
                **rule(stripper, r.prompt, r.object_name),
            }
            for r in prompts.itertuples()
        ]
    )


def run_strata(
    base: pd.DataFrame, labels: pd.DataFrame, strata: list[str], prefix: str, B: int
) -> dict:
    d = base.merge(labels[["concept_id", "prompt_id", *LABELS]], on=["concept_id", "prompt_id"])
    d = d[d.pf_masc.notna() & d.pf_full_full.notna()].reset_index(drop=True)
    out = {}
    for name in strata:
        sub = d if name == "all" else d[STRATA[name](d)].reset_index(drop=True)
        out[name] = rated_table(
            sub,
            ROWS,
            f"{prefix}{name}",
            g1="pf_g1",
            g2="pf_g2",
            comparisons=[(MASC, CONTROL)],
            human_row=False,
            B=B,
        )
    return out


def report(title: str, res: dict, paper: dict, check: PaperCheck, tag: str, B: int) -> None:
    lines = []
    for name, (n, ma, mr, ca, cr, dr, dr_ci) in paper.items():
        t = res[name]
        m, c = t["rows"][MASC], t["rows"][CONTROL]
        drho = comparison(t, MASC, CONTROL, "rho")
        da, da_c = (
            comparison(t, MASC, CONTROL, "alpha"),
            comparison(t, MASC, CONTROL, "alpha", True),
        )
        drho_c = comparison(t, MASC, CONTROL, "rho", True)
        lines.append(
            [
                name,
                f"{t['n_keys']:,}",
                f"{m['alpha']['point']:.3f}/{m['rho']['point']:.3f}",
                f"{c['alpha']['point']:.3f}/{c['rho']['point']:.3f}",
                f"{drho['diff_point']:.3f} {ci3(drho['ci95'])}",
                ci3(drho_c["ci95"]),
                f"{da['diff_point']:+.3f} {ci3(da['ci95'])}",
                ci3(da_c["ci95"]),
            ]
        )
        for k, ours, want in (
            ("keys", t["n_keys"], n),
            ("MaSC alpha", m["alpha"]["point"], ma),
            ("MaSC rho", m["rho"]["point"], mr),
            ("control alpha", c["alpha"]["point"], ca),
            ("control rho", c["rho"]["point"], cr),
            ("delta rho", drho["diff_point"], dr),
        ):
            check.add(f"{tag} {name} {k}", ours, want, 0 if k == "keys" else 3)
        check.add_ci(f"{tag} {name} delta rho CI", drho["ci95"], dr_ci)
        if (tag, name) in PAPER_ALPHA_DIFFS:
            want, want_ci = PAPER_ALPHA_DIFFS[(tag, name)]
            check.add(f"{tag} {name} delta alpha", da["diff_point"], want)
            check.add_ci(f"{tag} {name} delta alpha CI", da["ci95"], want_ci)
    print_table(
        f"{title}, B = {B:,}",
        [
            "Stratum",
            "Keys",
            "MaSC a/rho",
            "Control a/rho",
            "Δrho [key CI]",
            "[cluster CI]",
            "Δalpha [key CI]",
            "[cluster CI]",
        ],
        lines,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    args = ap.parse_args()
    df = read_keys()
    base = df[retained(df, 0.05) & ~df.is_style & df.pf_g1.notna() & df.pf_g2.notna()]

    dep = label_prompts(df, classify)
    lex = label_prompts(df, classify_lexical)
    res_dep = run_strata(base, dep, list(STRATA), "pf_stratum_", args.B)
    res_lex = run_strata(base, lex, LEXICAL, "pf_stratum_lexical_", args.B)
    path = write_json("pf_prompt_strata", {"dependency": res_dep, "lexical": res_lex})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = dep[["concept_id", "prompt_id", *LABELS]].merge(
        lex[["concept_id", "prompt_id", "action_pose", "subject_dependent", "scene_only"]],
        on=["concept_id", "prompt_id"],
        suffixes=("", "_lexical"),
    )
    labels.to_csv(OUT_DIR / "pf_prompt_strata_labels.csv", index=False)

    check = PaperCheck(intervals=args.B == DEFAULT_B)
    report("PF by dependency-based prompt stratum", res_dep, PAPER, check, "dependency", args.B)
    report("PF by lexical prompt stratum", res_lex, PAPER_LEXICAL, check, "lexical", args.B)
    print(f"\nWrote {path}")
    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())
