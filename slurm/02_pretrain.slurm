#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Stage 1: V-JEPA 2 mask-denoising pretraining of the encoder (self-supervised).
# Resumes from last.pt; requeues itself before the time limit (see on_time_limit in common.sh).
#SBATCH -J sr-jepa-pretrain
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
job_init "pretrain"
run_stage pretrain
