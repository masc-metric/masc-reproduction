"""Verify the frozen camera-ready export without statistical dependencies."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "masc-camera-ready-stats-2026-10-17"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (BUNDLE / name.lstrip("*")).resolve()
        assert path.is_relative_to(BUNDLE.resolve()), name
        assert digest(path) == expected, name
    data = json.loads((BUNDLE / "dbpp_comparisons.json").read_text())
    for name, expected in data["meta"]["code_sha256"].items():
        if name.startswith("camera_ready_analysis/cluster_ci/"):
            path = BUNDLE / "code" / Path(name).name
        else:
            path = ROOT / name
        assert digest(path) == expected, name
    with gzip.open(BUNDLE / "dbpp_keysets.json.gz", "rt") as stream:
        keysets = json.load(stream)
    for expected, keys in keysets.items():
        assert keys == sorted(set(keys)), expected
        assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == expected
    for analysis in data["analyses"]:
        keys = keysets[analysis["keyset_sha256"]]
        assert len(keys) == analysis["n_keys"], analysis["id"]
        subjects = {key.split("|", 1)[1].rsplit("-", 1)[0] for key in keys}
        assert len(subjects) == analysis["n_subjects"], analysis["id"]
    for section, filename, count in (
        ("comparisons", "dbpp_comparisons.csv", 82),
        ("rows", "dbpp_rows.csv", 206),
        ("analyses", "dbpp_analyses.csv", 23),
    ):
        with (BUNDLE / filename).open(newline="") as stream:
            exported = list(csv.DictReader(stream))
        assert len(data[section]) == len(exported) == count, section
        # Compare the scalar fields shared by each flat CSV and JSON record.
        for source, flat in zip(data[section], exported):
            for key, value in source.items():
                if key not in flat or isinstance(value, (list, dict)):
                    continue
                actual = flat[key]
                if value is None:
                    assert actual in ("", "None"), (section, key)
                elif isinstance(value, bool):
                    assert actual.lower() == str(value).lower(), (section, key)
                elif isinstance(value, (int, float)):
                    assert float(actual) == value, (section, key)
                else:
                    assert actual == value, (section, key)
    print("PASS: file/code hashes, key sets, analysis sizes, and 82/206/23 export records")
    print("Statistics and unavailable prediction inputs were not recomputed or verified.")


if __name__ == "__main__":
    main()
