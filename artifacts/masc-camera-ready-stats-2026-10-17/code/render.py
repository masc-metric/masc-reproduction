"""Print the comparison table of cluster_ci.json (markdown)."""

import json
import sys
from pathlib import Path

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "out/cluster_ci.json"))


def ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def res(c):
    return "resolved" if c[0] > 0 or c[1] < 0 else "not resolved"


print("| Table | Comparison | stat | diff | subject-cluster 95% CI | key 95% CI | under subject clusters |")
print("|---|---|---|---:|---|---|---|")
for name in ("table1", "table3", "table4", "stripping", "boundary", "boundary_vs_gpt4o"):
    t = d[name]
    keyc = {(c["a"], c["b"], c["stat"]): c for c in t["comparisons"]}
    for c in t["cluster_bootstrap"]["comparisons"]:
        k = keyc[(c["a"], c["b"], c["stat"])]
        print(f"| {name} ({t['n_keys']:,} keys, {t['cluster_bootstrap']['n_clusters']} subj.) | {c['a']} - {c['b']} | "
              f"{c['stat']} | {c['diff_point']:+.3f} | {ci(c['ci95'])} | {ci(k['ci95'])} | {res(c['ci95'])} |")
o = d["orida"]
pair = {c["b"]: c for c in o["pair"]["comparisons"]}
anch = {c["b"]: c for c in o["anchor_subject"]["comparisons"]}
for c in o["two_way_subject"]["comparisons"]:
    print(f"| ORIDa ({o['n_within']}+{o['n_cross']} pairs, {o['n_subjects']} subj.) | MaSC - {c['b']} | AUC | "
          f"{c['diff_point']:+.3f} | {ci(c['ci95'])} (anchor {ci(anch[c['b']]['ci95'])}) | {ci(pair[c['b']]['ci95'])} (pairs) | {res(c['ci95'])} |")
print()
print("Per-metric subject-cluster intervals")
for name in ("table1", "table3"):
    for lab, r in d[name]["cluster_bootstrap"]["rows"].items():
        print(f"{name} | {lab} | alpha {r['alpha']['point']:+.3f} {ci(r['alpha']['ci95'])} | rho {r['rho']['point']:+.3f} {ci(r['rho']['ci95'])}")
for lab, r in o["two_way_subject"]["rows"].items():
    print(f"orida | {lab} | AUC {r['point']:.3f} {ci(r['ci95'])} | pairs {ci(o['pair']['rows'][lab]['ci95'])}")
