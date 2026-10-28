#!/bin/bash
# Submit the whole SR-JEPA pipeline as a chain of Slurm jobs (run on a login node).
#
#   bash slurm/submit_all.sh                  everything, from download + prepare
#   bash slurm/submit_all.sh --from sr        restart from a stage (earlier stages are not submitted)
#   bash slurm/submit_all.sh --smoke          the 30-minute check on synthetic data (debug QOS; TUNE=1 adds tiny studies)
#   bash slurm/submit_all.sh --dry-run        print the sbatch commands, submit nothing
#
# Stages: prepare -> pretrain -> sr -> decoder ----------------> e2e (jepa) --\
#                       \-> probe ------------------------------------------------> evaluate + pack
#            \-> e2e (scratch) ----------------------------------------------------/
# TUNE=1: an Optuna study (job array of TUNE_WORKERS workers) runs right before each stage in TUNE_STAGES; the
# stage trains with its best values. ENCODERS="S B": one such chain per encoder size (only the first is tuned,
# the others reuse its values), then a job that compares them.
# Every training stage resumes from last.pt and every study from its journal, so re-submitting is safe.
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
        -h | --help) sed -n '2,15p' "$0"; exit 0 ;;
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
    TUNED_FROM=""
    ENCODER_LIST=(S)
else
    resolve_experiment
fi
if [[ "$ACCOUNT" == mXXXX* ]]; then
    ((DRY)) || die "set ACCOUNT in slurm/settings.sh"
    echo "(dry run: ACCOUNT is still the placeholder '$ACCOUNT')"
fi

# ---- tuning: studies to run here, or tuned values to reuse --------------------------------------
STUDIES="$(tuned_studies)"
if [[ -n "$TUNED_FROM" ]]; then
    ((TUNE)) && echo "TUNED_FROM is set: reusing its tuned values instead of tuning (TUNE=1 ignored)"
    TUNE=0
    for s in $STUDIES; do
        [[ -f "$TUNED_FROM/tuning/$s/best_params.json" ]] \
            || die "TUNED_FROM=$TUNED_FROM has no tuning/$s/best_params.json (study $s in TUNE_STAGES)"
    done
fi
((TUNE)) && [[ -z "$STUDIES" ]] && die "TUNE=1 but TUNE_STAGES lists no study"

# ---- sbatch wrapper ------------------------------------------------------------------------------
# sub <stage-name> <time> <script> <dependency or ""> [extra exports] [array range]; uses LOGDIR / ENV_FILE
sub() {
    local name="$1" time="$2" script="$3" dep="$4" exports="${5:-}" array="${6:-}"
    local -a args=(--parsable -J "jepa_${RUN_TAG}_${name}" -t "$time" --kill-on-invalid-dep=yes
                   --export="ALL,RUN_ENV=$ENV_FILE${exports:+,$exports}")
    if [[ -n "$array" ]]; then
        args+=(--array="$array" -o "$LOGDIR/${name}-%A_%a.out")
    else
        args+=(-o "$LOGDIR/${name}-%j.out")
    fi
    if [[ "$name" == prepare && -n "$CPU_ACCOUNT" ]]; then
        args+=(-A "$CPU_ACCOUNT" -C cpu -q shared -c 32)
    elif ((SMOKE)); then
        args+=(-A "$ACCOUNT" -C gpu -q debug --gpus=1 -c 32)
    else
        local constraint=gpu
        [[ "$GPU_MEM" == 80 ]] && constraint="gpu&hbm80g"
        args+=(-A "$ACCOUNT" -C "$constraint" -q "$QOS" --gpus=1 -c 32)
        [[ "$QOS" == regular ]] && args+=(-N 1)
        # training and tuning: requeue on preemption and, via USR1 10 min before the limit, on timeout
        [[ "$name" != prepare && "$name" != evaluate && "$name" != compare ]] && args+=(--requeue --signal=B:USR1@600)
    fi
    [[ -n "$dep" ]] && args+=(--dependency="$dep")
    args+=("$REPO/slurm/$script")
    if ((DRY)); then
        echo "sbatch ${args[*]}" >&2
        echo "<${RUN_TAG}:$name>"
    else
        local id
        id=$(sbatch "${args[@]}")
        id="${id%%;*}"
        printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$id" "${array:--}" >> "$LOGDIR/jobs.tsv"
        echo "  submitted $name: $id${array:+ (array $array)}" >&2
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

# freeze the settings of the current experiment for its jobs
write_env() {
    local v
    for v in REPO ROOT EXPERIMENT ACCOUNT CPU_ACCOUNT QOS GPU_MEM DATASET RUN_TAG RUN_DATE MAX_EVENTS BATCH PATIENCE \
             E2E_INITS EXTRA_SET CFS_COPY MAX_REQUEUE ENCODER_SIZE TUNE_THIS TUNED_FROM_EXP TUNE_STAGES TUNE_WORKERS \
             N_TRIALS; do
        printf 'export %s=%q\n' "$v" "${!v}"
    done
}
start_experiment() {
    LOGDIR="$(log_dir)"
    ENV_FILE="$LOGDIR/run.env"
    echo
    echo "experiment : $(experiment_dir)   (encoder $ENCODER_SIZE)"
    echo "cache      : $(cache_dir)"
    echo "logs       : $LOGDIR"
    if ((TUNE_THIS)); then
        echo "tuning     : studies '$STUDIES', $TUNE_WORKERS workers each"
    elif [[ -n "$TUNED_FROM_EXP" ]]; then
        echo "tuning     : values of $TUNED_FROM_EXP"
    fi
    if ((DRY)); then
        echo "--- run.env would be:"
        write_env
    else
        mkdir -p "$LOGDIR"
        [[ -f "$ENV_FILE" ]] && cp "$ENV_FILE" "$ENV_FILE.$(date +%Y%m%d-%H%M%S)"
        write_env > "$ENV_FILE"
    fi
}

if ((SMOKE)); then
    ENCODER_SIZE=S TUNE_THIS=$TUNE TUNED_FROM_EXP=""
    start_experiment
    sub smoke 00:30:00 00_smoke.slurm "" "SMOKE_TUNE=$TUNE" > /dev/null
    echo "smoke job submitted; wait for 'SMOKE OK' at the end of $LOGDIR/smoke-<jobid>.out"
    exit 0
fi

# ---- one chain per encoder size ------------------------------------------------------------------
BASE_TAG="$RUN_TAG"
# TJ_<study> = tune job id of the first size's study; the other sizes wait for it (plain variables, no bash 4 maps)
tj() { local v="TJ_$1"; echo "${!v:-}"; }
REF_EXP="" FIRST_ENV=""
EVALS=() EXPS=()
P=""
has_init() { [[ " ${E2E_LIST[*]} " == *" $1 "* ]]; }
has_study() { [[ " $STUDIES " == *" $1 "* ]]; }
for idx in "${!ENCODER_LIST[@]}"; do
    ENCODER_SIZE="${ENCODER_LIST[$idx]}"
    RUN_TAG="$BASE_TAG"
    ((${#ENCODER_LIST[@]} > 1)) && RUN_TAG="${BASE_TAG}-vit${ENCODER_SIZE}"
    EXPERIMENT="${RUN_DATE}_${RUN_TAG}"
    if ((idx == 0)); then
        TUNE_THIS=$TUNE
        TUNED_FROM_EXP="$TUNED_FROM"
        REF_EXP="$(experiment_dir)"
    else
        TUNE_THIS=0
        TUNED_FROM_EXP="$TUNED_FROM"
        ((TUNE)) && TUNED_FROM_EXP="$REF_EXP"
    fi
    start_experiment
    ((idx == 0)) && FIRST_ENV="$ENV_FILE"

    # tune <study> <dependency>: submit the study's job array (first size only); tj <study> -> its job id
    tune() {
        if ((TUNE_THIS)) && has_study "$1"; then
            printf -v "TJ_$1" '%s' "$(sub "tune-$1" "$TIME_TUNE" tune.slurm "$2" "TUNE_STUDY=$1" \
                "0-$((TUNE_WORKERS - 1))")"
        fi
    }
    T="" PT="" SR="" DEC="" PR="" EJ="" ES="" EV=""
    if ((idx == 0)) && wanted prepare; then P=$(sub prepare "$TIME_PREPARE" 01_prepare.slurm ""); fi
    # a study's workers may fail one by one; the stage then still runs if the study has a result (tune.require
    # checks it), hence afterany on the tune arrays
    if wanted pretrain; then
        tune pretrain "$(dep "afterok:$P")"; T="$(tj pretrain)"
        PT=$(sub pretrain "$TIME_PRETRAIN" 02_pretrain.slurm "$(dep "afterok:$P" "afterany:$T")")
    fi
    if wanted sr; then
        tune sr "$(dep "afterok:$PT")"; T="$(tj sr)"
        SR=$(sub sr "$TIME_SR" 03_sr.slurm "$(dep "afterok:$PT" "afterany:$T")")
    fi
    if wanted decoder; then
        tune decoder "$(dep "afterok:$SR")"; T="$(tj decoder)"
        DEC=$(sub decoder "$TIME_DECODER" 04_decoder.slurm "$(dep "afterok:$SR" "afterany:$T")")
    fi
    if wanted probe; then
        tune probe "$(dep "afterok:$PT")"; T="$(tj probe)"
        PR=$(sub probe "$TIME_PROBE" 05_probe.slurm "$(dep "afterok:$PT" "afterany:$T")")
    fi
    if wanted e2e; then
        if has_init scratch; then
            tune e2e_scratch "$(dep "afterok:$P")"; T="$(tj e2e_scratch)"
            ES=$(sub e2e-scratch "$TIME_E2E" 06_e2e.slurm "$(dep "afterok:$P" "afterany:$T")" E2E_INIT=scratch)
        fi
        if has_init jepa; then
            tune e2e_jepa "$(dep "afterok:$DEC")"; T="$(tj e2e_jepa)"
            EJ=$(sub e2e-jepa "$TIME_E2E" 06_e2e.slurm "$(dep "afterok:$DEC" "afterany:$T")" E2E_INIT=jepa)
        fi
    fi
    # the decoder is required; the probe and the ablations are optional (afterany: evaluate uses what finished)
    if wanted evaluate; then
        EV=$(sub evaluate "$TIME_EVAL" 07_evaluate.slurm \
            "$(dep "afterok:$DEC" "afterany:$PR" "afterany:$EJ" "afterany:$ES")")
    fi
    EVALS+=("$EV")
    EXPS+=("$(experiment_dir)")
done

if ((${#ENCODER_LIST[@]} > 1)); then
    RUN_TAG="$BASE_TAG"
    EXPERIMENT="${RUN_DATE}_${BASE_TAG}-compare"
    LOGDIR="$ROOT/logs/$DATASET/$EXPERIMENT"
    ENV_FILE="$FIRST_ENV"    # REPO, ROOT, account settings; the comparison's inputs come through the exports
    deps=()
    for e in "${EVALS[@]}"; do deps+=("afterany:$e"); done
    joined="$(IFS=:; echo "${EXPS[*]}")"
    compare_out="$ROOT/comparisons/${DATASET}_${RUN_DATE}_${BASE_TAG}.md"
    ((DRY)) || mkdir -p "$LOGDIR"
    echo
    echo "comparison : $compare_out"
    sub compare "$TIME_COMPARE" 08_compare.slurm "$(dep "${deps[@]}")" \
        "COMPARE_EXPERIMENTS=$joined,COMPARE_OUT=$compare_out" > /dev/null
fi

echo
if ((DRY)); then
    echo "dry run: nothing submitted (the commands above would be)"
else
    echo "submitted. Follow with:  bash slurm/status.sh <experiment>   (job lists: \$ROOT/logs/$DATASET/<experiment>/jobs.tsv)"
    echo "squeue -u \$USER -o '%.18i %.40j %.10T %.10M %.20R'"
fi
