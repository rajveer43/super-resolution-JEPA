#!/bin/bash
# Submit the whole SR-JEPA pipeline as a chain of Slurm jobs (run on a login node).
#
#   bash slurm/submit_all.sh                  everything, from download + prepare
#   bash slurm/submit_all.sh --from sr        restart from a stage (earlier stages are not submitted)
#   bash slurm/submit_all.sh --smoke          the 30-minute check on synthetic data (debug QOS)
#   bash slurm/submit_all.sh --dry-run        print the sbatch commands, submit nothing
#
# Stages: prepare -> pretrain -> sr -> decoder ----------------> e2e (jepa) --\
#                       \-> probe ------------------------------------------------> evaluate + pack
#            \-> e2e (scratch) ----------------------------------------------------/
# Every training stage resumes from last.pt, so re-submitting after a failure or timeout is safe.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"

STAGES=(prepare pretrain sr decoder probe e2e evaluate)
FROM=prepare
DRY=0
SMOKE=0
while (($#)); do
    case "$1" in
        --from) FROM="${2:?--from needs a stage}"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        -h | --help) sed -n '2,12p' "$0"; exit 0 ;;
        *) die "unknown option $1" ;;
    esac
done

stage_index() {
    local i
    for i in "${!STAGES[@]}"; do [[ "${STAGES[$i]}" == "$1" ]] && { echo "$i"; return; }; done
    die "unknown stage '$1' (choose from: ${STAGES[*]})"
}
FROM_I=$(stage_index "$FROM")
wanted() { (($(stage_index "$1") >= FROM_I)); }

load_settings
unset RUN_ENV
if ((!SMOKE)) && [[ "$FROM" != prepare && -z "$RUN_DATE" ]]; then
    # a restart continues the latest experiment with this tag, even on a later day
    RUN_DATE="$(latest_experiment_date)"
    [[ -n "$RUN_DATE" ]] && echo "continuing the experiment from $RUN_DATE (set RUN_DATE in settings.sh to pick another)"
fi
if ((SMOKE)); then
    ROOT="$ROOT/smoke"
    resolve_experiment -smoke
    EXTRA_SET=""
    MAX_EVENTS=""
else
    resolve_experiment
fi
if [[ "$ACCOUNT" == mXXXX* ]]; then
    ((DRY)) || die "set ACCOUNT in slurm/settings.sh"
    echo "(dry run: ACCOUNT is still the placeholder '$ACCOUNT')"
fi

LOGDIR="$(log_dir)"
ENV_FILE="$LOGDIR/run.env"
echo "experiment : $(experiment_dir)"
echo "cache      : $(cache_dir)"
echo "logs       : $LOGDIR"

# ---- freeze the settings for the jobs ------------------------------------------------------------
write_env() {
    local v
    for v in REPO ROOT EXPERIMENT ACCOUNT CPU_ACCOUNT QOS GPU_MEM DATASET RUN_TAG RUN_DATE MAX_EVENTS BATCH PATIENCE \
             E2E_INITS EXTRA_SET CFS_COPY MAX_REQUEUE; do
        printf 'export %s=%q\n' "$v" "${!v}"
    done
}
if ((DRY)); then
    echo "--- run.env would be:"
    write_env
else
    mkdir -p "$LOGDIR"
    [[ -f "$ENV_FILE" ]] && cp "$ENV_FILE" "$ENV_FILE.$(date +%Y%m%d-%H%M%S)"
    write_env > "$ENV_FILE"
fi

# ---- sbatch wrapper ------------------------------------------------------------------------------
# sub <stage-name> <time> <script> <dependency or ""> [extra exports]
sub() {
    local name="$1" time="$2" script="$3" dep="$4" exports="${5:-}"
    local -a args=(--parsable -J "jepa_${RUN_TAG}_${name}" -t "$time" --kill-on-invalid-dep=yes
                   --export="ALL,RUN_ENV=$ENV_FILE${exports:+,$exports}" -o "$LOGDIR/${name}-%j.out")
    if [[ "$name" == prepare && -n "$CPU_ACCOUNT" ]]; then
        args+=(-A "$CPU_ACCOUNT" -C cpu -q shared -c 32)
    elif ((SMOKE)); then
        args+=(-A "$ACCOUNT" -C gpu -q debug --gpus=1 -c 32)
    else
        local constraint=gpu
        [[ "$GPU_MEM" == 80 ]] && constraint="gpu&hbm80g"
        args+=(-A "$ACCOUNT" -C "$constraint" -q "$QOS" --gpus=1 -c 32)
        [[ "$QOS" == regular ]] && args+=(-N 1)
        # training stages: requeue on preemption and, via USR1 10 min before the limit, on timeout
        [[ "$name" != prepare && "$name" != evaluate ]] && args+=(--requeue --signal=B:USR1@600)
    fi
    [[ -n "$dep" ]] && args+=(--dependency="$dep")
    args+=("$REPO/slurm/$script")
    if ((DRY)); then
        echo "sbatch ${args[*]}" >&2
        echo "<$name>"
    else
        local id
        id=$(sbatch "${args[@]}")
        id="${id%%;*}"
        printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$id" >> "$LOGDIR/jobs.tsv"
        echo "  submitted $name: $id" >&2
        echo "$id"
    fi
}

# dependency string from "type:id" pairs whose id is not empty, e.g. afterok:123,afterany:456
dep() {
    local out="" p
    for p in "$@"; do
        [[ -n "${p#*:}" ]] && out+="${out:+,}$p"
    done
    echo "$out"
}

if ((SMOKE)); then
    sub smoke 00:30:00 00_smoke.slurm "" > /dev/null
    echo "smoke job submitted; wait for 'SMOKE OK' at the end of $LOGDIR/smoke-<jobid>.out"
    exit 0
fi

has_init() { [[ " ${E2E_LIST[*]} " == *" $1 "* ]]; }
P="" PT="" SR="" DEC="" PR="" EJ="" ES=""
wanted prepare && P=$(sub prepare "$TIME_PREPARE" 01_prepare.slurm "")
wanted pretrain && PT=$(sub pretrain "$TIME_PRETRAIN" 02_pretrain.slurm "$(dep "afterok:$P")")
wanted sr && SR=$(sub sr "$TIME_SR" 03_sr.slurm "$(dep "afterok:$PT")")
wanted decoder && DEC=$(sub decoder "$TIME_DECODER" 04_decoder.slurm "$(dep "afterok:$SR")")
wanted probe && PR=$(sub probe "$TIME_PROBE" 05_probe.slurm "$(dep "afterok:$PT")")
if wanted e2e; then
    has_init scratch && ES=$(sub e2e-scratch "$TIME_E2E" 06_e2e.slurm "$(dep "afterok:$P")" E2E_INIT=scratch)
    has_init jepa && EJ=$(sub e2e-jepa "$TIME_E2E" 06_e2e.slurm "$(dep "afterok:$DEC")" E2E_INIT=jepa)
fi
# the decoder is required; the probe and the ablations are optional (afterany: evaluate uses what finished)
wanted evaluate && sub evaluate "$TIME_EVAL" 07_evaluate.slurm \
    "$(dep "afterok:$DEC" "afterany:$PR" "afterany:$EJ" "afterany:$ES")" > /dev/null

echo
if ((DRY)); then
    echo "dry run: nothing submitted (the commands above would be)"
else
    echo "submitted. Follow with:  bash slurm/status.sh $EXPERIMENT   (job list: $LOGDIR/jobs.tsv)"
    echo "squeue -u \$USER -o '%.18i %.40j %.10T %.10M %.20R'"
fi
