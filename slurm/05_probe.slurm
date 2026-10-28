#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Attentive probe on frozen latents (quark/gluon for jets, E_inc for showers), used by the evaluation.
# Needs only the pretrained encoder, so it runs in parallel with the SR and decoder stages.
#SBATCH -J sr-jepa-probe
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
job_init "train_probe"
run_stage train_probe
