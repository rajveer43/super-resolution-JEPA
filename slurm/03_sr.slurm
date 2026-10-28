#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Stage 2: latent SR world model on the frozen encoder (teacher forcing + rollout). Resumes from last.pt.
#SBATCH -J sr-jepa-sr
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
job_init "train_sr"
run_stage train_sr
