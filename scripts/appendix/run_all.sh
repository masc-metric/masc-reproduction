#!/usr/bin/env bash
# Rebuild the Table 2 confidence intervals and every appendix table from the
# scoring runs (README, "Appendix"), and compare each with the paper.
# Extra arguments (e.g. --B 500 for a quick run) go to every analysis script.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}

$PY scripts/appendix/build_key_table.py || exit 1
failed=()
for script in orida_auc_ci mask_area_filtering mask_boundaries subject_stripping pf_prompt_strata; do
    echo; echo "=== $script"
    $PY "scripts/appendix/$script.py" "$@" || failed+=("$script")
done
echo
if ((${#failed[@]})); then
    echo "Differences from the paper in: ${failed[*]}"
    exit 1
fi
echo "Every table reproduced."
