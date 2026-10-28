#!/bin/bash
# Pack one experiment's results (metrics, summary, figures, histories, configs, logs) into a tarball to send back.
# Checkpoints (*.pt) are left out unless --with-checkpoints is given; then the pretrained encoder (last.pt) and
# every stage's best.pt are included, the other last.pt files are not. Copies the tarball to CFS_COPY if set.
#
#   bash slurm/pack_results.sh [<experiment> | --smoke] [--with-checkpoints]
# Runs automatically at the end of the evaluate job.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${REPO}/slurm/common.sh"

which_run=""
with_ckpt=0
for a in "$@"; do
    case "$a" in
        --with-checkpoints) with_ckpt=1 ;;
        *) which_run="$a" ;;
    esac
done
load_settings
[[ -n "${RUN_ENV:-}" && -n "${SLURM_JOB_ID:-}" ]] || load_experiment "$which_run"

exp="$(experiment_dir)"
name="${DATASET}_${EXPERIMENT}"
[[ -d "$exp" ]] || die "no results at $exp"
suffix=""
((with_ckpt)) && suffix="_with_checkpoints"
out="$ROOT/results_${name}${suffix}.tar.gz"
# tuning: best_params.json, trials.csv, plots and the study journal; not the per-trial folders
exclude=(--exclude='*.tmp' --exclude="$name/tuning/*/trial_*")
if ((with_ckpt)); then
    for s in sr decoder probe e2e_jepa e2e_scratch; do exclude+=(--exclude="$name/$s/last.pt"); done
else
    exclude+=(--exclude='*.pt')
fi

# experiment folder + the job logs, under one top-level directory named after the experiment
stage_dir="$(mktemp -d)"
trap 'rm -rf "$stage_dir"' EXIT
ln -s "$exp" "$stage_dir/$name"
logdir="$(dirname "$RUN_ENV")"
[[ -d "$logdir" ]] && ln -s "$logdir" "$stage_dir/${name}_slurm-logs"
tar -czhf "$out" "${exclude[@]}" -C "$stage_dir" .
echo "packed $(du -h "$out" | cut -f1)  $out"

if [[ -n "${CFS_COPY:-}" ]]; then
    mkdir -p "$CFS_COPY"
    cp "$out" "$CFS_COPY/"
    echo "copied to $CFS_COPY/$(basename "$out")"
fi
