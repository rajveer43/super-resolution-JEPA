#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Ablation, one init per job (E2E_INIT): jepa = end-to-end fine-tuning of stages 1-3 (after the decoder);
# scratch = the same network from random weights (needs only the data, so it starts right after prepare).
# Resumes from last.pt.
#SBATCH -J sr-jepa-e2e
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
: "${E2E_INIT:?}"
job_init "train_e2e ($E2E_INIT)"
run_stage train_e2e "e2e.inits=[$E2E_INIT]"
