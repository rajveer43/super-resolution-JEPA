#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Evaluates every trained variant (jepa, and e2e_jepa / e2e_scratch if they finished) against the truth,
# then packs the results (pack_results.sh). Runs even if the probe or an ablation failed.
#SBATCH -J sr-jepa-evaluate
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
job_init "evaluate"
run_stage evaluate    # prints the summary table
bash "$REPO/slurm/pack_results.sh"
