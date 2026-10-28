#!/bin/bash
# Submitted by slurm/submit_all.sh when ENCODERS lists several sizes: one table comparing the evaluations of the
# experiments (one per encoder size), written to COMPARE_OUT ($ROOT/comparisons/) and printed here. Runs with
# whatever finished.
#SBATCH -J sr-jepa-compare
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
: "${COMPARE_EXPERIMENTS:?}" "${COMPARE_OUT:?}"
job_init "compare encoder sizes"
out="$COMPARE_OUT"
read -r -a exps <<< "${COMPARE_EXPERIMENTS//:/ }"
python -m sr_jepa.eval.compare "${exps[@]}" --out "$out"
if [[ -n "${CFS_COPY:-}" ]]; then
    mkdir -p "$CFS_COPY"
    cp "$out" "$CFS_COPY/"
    echo "copied to $CFS_COPY/$(basename "$out")"
fi
