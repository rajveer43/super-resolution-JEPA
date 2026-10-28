#!/bin/bash
# Submitted by slurm/submit_all.sh (TUNE=1) as a job array: every task is one worker of the same Optuna study
# (TUNE_STUDY = pretrain | sr | decoder | probe | e2e_jepa | e2e_scratch), sharing its journal file, so the trials
# run in parallel on TUNE_WORKERS GPUs. A worker stops once the study has its completed trials; a requeued worker
# continues the study. The stage job after this array trains with the study's best values.
#SBATCH -J sr-jepa-tune
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
: "${TUNE_STUDY:?}"
job_init "tune_$TUNE_STUDY (worker ${SLURM_ARRAY_TASK_ID:-0})"
run_stage "tune_$TUNE_STUDY"
