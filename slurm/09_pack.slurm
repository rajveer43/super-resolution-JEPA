#!/bin/bash
# Submitted by slurm/submit_all.sh with --dependency=afterany on every other job of the experiment, so it runs
# whether they succeeded, failed or were cancelled: packs what exists (logs, histories, configs, figures and
# every checkpoint that was written), once without and once with the checkpoints. See pack_results.sh.
#SBATCH -J sr-jepa-pack
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
echo "pack: $(experiment_dir) on $(hostname), $(date '+%F %T')"
bash "$REPO/slurm/pack_results.sh"
bash "$REPO/slurm/pack_results.sh" --with-checkpoints
