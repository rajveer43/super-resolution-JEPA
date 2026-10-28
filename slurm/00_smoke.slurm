#!/bin/bash
# Submitted by "bash slurm/submit_all.sh --smoke" (debug QOS, 1 GPU, 30 min).
# Every stage on a few hundred synthetic events with tiny models, under $ROOT (= <ROOT>/smoke), then checks
# that the checkpoints and the evaluation were written. SMOKE_TUNE=1 (TUNE=1 in settings.sh) also runs every
# Optuna study with a few trials and checks that each stage trained with its tuned values.
# Prints "SMOKE OK" on success.
#SBATCH -J sr-jepa-smoke
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh --smoke}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
job_init "smoke test"
CONFIG="smoke_$DATASET"
small=()
for s in pretrain sr decoder e2e; do small+=("$s.batch_size=32"); done
tune=()
((${SMOKE_TUNE:-0})) && tune=(tune.enabled=true)
run_stage all device=cuda data.max_events=400 eval.max_events=64 "${small[@]}" ${tune[@]+"${tune[@]}"}

exp="$(experiment_dir)"
missing=0
check() {
    if compgen -G "$1" > /dev/null; then
        echo "  ok      $1"
    else
        echo "  MISSING $1"
        missing=1
    fi
}
echo "checking outputs in $exp"
for f in pretrain/last.pt sr/best.pt decoder/best.pt probe/best.pt e2e_jepa/best.pt e2e_scratch/best.pt \
         pretrain/history.csv evaluation/metrics.json evaluation/summary.md evaluation/examples.png; do
    check "$exp/$f"
done
if ((${SMOKE_TUNE:-0})); then
    for s in pretrain sr decoder probe e2e_jepa e2e_scratch; do
        check "$exp/tuning/$s/best_params.json"
        # each stage folder is named like its study; its config.json records what was applied
        if python -c "import json,sys; c=json.load(open(sys.argv[1])); sys.exit(sys.argv[2] not in c['_tuned']['applied'])" \
            "$exp/$s/config.json" "$s" 2> /dev/null; then
            echo "  ok      $s/config.json records the tuned values"
        else
            echo "  MISSING tuned values in $s/config.json"
            missing=1
        fi
    done
fi
((missing == 0)) || die "smoke test finished but outputs are missing (see above)"
grep -q "events_per_s" "$exp/pretrain/history.csv" && echo "pretraining throughput (events/s, tiny model):" \
    && python -c "import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); print(' ', r[-1]['events_per_s'])" \
        "$exp/pretrain/history.csv"
echo "SMOKE OK"
