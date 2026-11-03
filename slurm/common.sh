# Shared by every job and by the submit / status / pack scripts. Not run directly.
# Jobs get RUN_ENV (the frozen settings written by submit_all.sh) through sbatch --export.

# ---------------------------------------------------------------- settings
# REPO and the settings: from the frozen run.env inside a job, from settings.sh otherwise.
load_settings() {
    if [[ -n "${RUN_ENV:-}" ]]; then
        # shellcheck disable=SC1090
        source "$RUN_ENV"
    else
        REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        # shellcheck source=settings.sh
        source "$REPO/slurm/settings.sh"
    fi
    [[ -n "${ROOT:-}" && "$ROOT" != "/sr_jepa" ]] || die "ROOT is empty: is \$SCRATCH set? (see settings.sh)"
    [[ "$DATASET" == qg || "$DATASET" == calo ]] || die "DATASET must be qg or calo, not '$DATASET'"
    read -r -a E2E_LIST <<< "${E2E_INITS:-}"
    read -r -a ENCODER_LIST <<< "${ENCODERS:-S}"
    local e
    for e in "${ENCODER_LIST[@]}"; do [[ "$e" =~ ^[SBL]$ ]] || die "ENCODERS may only list S, B and L, not '$e'"; done
}

# Optuna studies of TUNE_STAGES (e2e -> one study per init in E2E_INITS), e.g. "pretrain sr e2e_jepa".
tuned_studies() {
    local s i out=()
    for s in ${TUNE_STAGES:-}; do
        if [[ "$s" == e2e ]]; then
            for i in "${E2E_LIST[@]}"; do out+=("e2e_$i"); done
        else
            out+=("$s")
        fi
    done
    echo "${out[*]}"
}

# Encoder presets of the size comparison: "dim depth heads batch-cap" (batch caps are memory estimates for a
# 40 GB A100 at 128 x 128 jets; the learning rates follow the batch through lr_ref_batch).
encoder_preset() {
    case "$1" in
        S) echo "384 12 6 1024" ;;
        B) echo "768 12 12 128" ;;
        L) echo "1024 24 16 64" ;;
        *) die "unknown encoder size '$1'" ;;
    esac
}

die() { echo "ERROR: $*" >&2; exit 1; }

# Largest batch PER GPU of each stage on a 40 GB A100 (x GPU_MEM / 40 for 80 GB), for the S encoder at 128 x 128
# jets. Estimates from the stages' memory profiles, measured against the one failure we have (SR and e2e at 256
# on one GPU: out of memory in the first step, while pretraining at 256 was fine): pretraining sees 64 tokens per
# event and only the unmasked ones; SR attends over 2 x 257 tokens twice with gradients; e2e adds the encoder and
# the decoder to that. The training split (11.6 GB in fp16) also lives on the GPU unless it is sharded over GPUs.
stage_batch_cap() {
    case "$1" in
        pretrain) echo 256 ;;
        sr) echo 64 ;;
        decoder) echo 128 ;;
        e2e) echo 32 ;;
        *) die "no batch cap for stage '$1'" ;;
    esac
}

# For scripts run by hand after submission: load the frozen settings of one experiment.
#   load_experiment ""         the most recently submitted one
#   load_experiment <name>     that one, e.g. 2026-10-28_vjepa2-baseline
#   load_experiment --smoke    the most recent smoke test
load_experiment() {
    local logs="$ROOT/logs" id="${1:-}" env_file=""
    if [[ "$id" == --smoke ]]; then
        logs="$ROOT/smoke/logs"
        id=""
    fi
    env_file="$(newest_env_file "$logs" "${id:-*}")"
    [[ -f "$env_file" ]] || die "no submitted $DATASET experiment found under $logs${id:+ for $id}"
    RUN_ENV="$env_file"
    load_settings
}

# Newest run.env of a $DATASET experiment whose name matches a pattern (empty if none).
newest_env_file() {
    local logs="$1" pattern="$2" f
    local -a files=()
    # shellcheck disable=SC2086
    for f in "$logs/$DATASET/"$pattern/run.env; do
        [[ -f "$f" ]] && files+=("$f")
    done
    ((${#files[@]})) || return 0
    # shellcheck disable=SC2012
    ls -t "${files[@]}" | head -1 || true
}

# Most recent experiment with this dataset and tag -> its date (empty if none).
latest_experiment_date() {
    local f
    f="$(newest_env_file "$ROOT/logs" "*_${RUN_TAG}")"
    [[ -n "$f" ]] || f="$(newest_env_file "$ROOT/logs" "*_${RUN_TAG}-vit?")"
    [[ -n "$f" ]] && basename "$(dirname "$f")" | cut -c1-10
    return 0
}

# Fix the experiment name once (submit_all.sh): EXPERIMENT = <date>_<tag>, RUN_DATE = today if empty.
resolve_experiment() {
    RUN_DATE="${RUN_DATE:-$(TZ=Asia/Kolkata date +%Y-%m-%d)}"
    RUN_TAG="${RUN_TAG}${1:-}"
    [[ "$RUN_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "RUN_DATE must look like 2026-10-28, not '$RUN_DATE'"
    [[ "$RUN_TAG" =~ ^[A-Za-z0-9.-]+$ ]] || die "RUN_TAG may only use letters, digits, '-' and '.': '$RUN_TAG'"
    EXPERIMENT="${RUN_DATE}_${RUN_TAG}"
}

experiment_dir() { echo "$ROOT/experiments/$DATASET/$EXPERIMENT"; }
log_dir() { echo "$ROOT/logs/$DATASET/$EXPERIMENT"; }
raw_dir() { echo "$ROOT/raw/$DATASET"; }        # shared by every experiment of this dataset
# one cache per dataset and event count: prepare skips an existing cache, so a different count needs its own
cache_dir() { echo "$ROOT/cache/${DATASET}_${MAX_EVENTS:-all}"; }

# ---------------------------------------------------------------- environment
setup_python() {
    if ! type module > /dev/null 2>&1 && [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    fi
    if type module > /dev/null 2>&1; then
        set +u    # Lmod's shell code reads unset variables
        module load pytorch
        set -u
    fi
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True    # less fragmentation (PyTorch's own OOM advice)
    export MPLBACKEND=Agg
    local cpus="${SLURM_CPUS_PER_TASK:-4}"
    PHYS_CORES=$((cpus > 1 ? cpus / 2 : 1))   # Perlmutter: 2 hardware threads per core
    export OMP_NUM_THREADS="$PHYS_CORES"
}

job_header() {
    echo "==================================================================="
    echo "stage      : $1"
    echo "experiment : $(experiment_dir)"
    echo "job        : ${SLURM_JOB_ID:-local} (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname)"
    echo "started    : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "code       : $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2> /dev/null) $(git -C "$REPO" rev-parse --short HEAD 2> /dev/null)"
    echo "settings   : dataset=$DATASET max_events=${MAX_EVENTS:-all} batch=$BATCH gpus=${NGPUS:-1} patience=$PATIENCE qos=$QOS extra='$EXTRA_SET'"
    echo "encoder    : ${ENCODER_SIZE:-S} | tuning: $( ((${TUNE_THIS:-0})) && echo "studies '$(tuned_studies)' in this experiment" \
        || { [[ -n "${TUNED_FROM_EXP:-}" ]] && echo "values from $TUNED_FROM_EXP" || echo off; })"
    python -c 'import sys, torch; print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'
    # CPU nodes (prepare) may have nvidia-smi on PATH but no driver: it then fails, and under
    # set -e that failure would end the job here, before python starts
    if command -v nvidia-smi > /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2> /dev/null || echo "(no GPU on this node)"
    fi
    echo "==================================================================="
}

# Overrides that put every path of this experiment under $ROOT and apply the settings.
pipeline_sets() {
    local workers=$((PHYS_CORES > 2 ? PHYS_CORES / 2 : 1)) s
    SETS=("run_name=$EXPERIMENT" "paths.raw=$(raw_dir)" "paths.cache=$(cache_dir)" "paths.out=$(experiment_dir)"
          "num_workers=$workers" "data.max_events=${MAX_EVENTS:-null}")
    local dim depth heads cap batch="${BATCH:-}" gpus="${NGPUS:-1}" b c
    read -r dim depth heads cap <<< "$(encoder_preset "${ENCODER_SIZE:-S}")"
    if [[ "${ENCODER_SIZE:-S}" != S ]]; then  # S keeps the config's encoder (ViT-S unless EXTRA_SET says otherwise)
        SETS+=("encoder.dim=$dim" "encoder.depth=$depth" "encoder.heads=$heads")
    else
        cap=1024
    fi
    if [[ -n "$batch" ]]; then
        # the global batch of each stage: BATCH, capped by what fits per GPU, and a multiple of the GPU count
        for s in pretrain sr decoder e2e; do
            c=$(($(stage_batch_cap "$s") * ${GPU_MEM:-40} / 40))
            ((c > cap)) && c=$cap
            b=$batch
            ((b > c * gpus)) && b=$((c * gpus))
            b=$((b / gpus * gpus))
            ((b > 0)) || die "batch of $s: $batch events cannot be split over $gpus GPUs"
            [[ "$b" != "$batch" ]] && echo "batch $s: $batch -> $b ($((b / gpus)) per GPU, the most that fits)"
            SETS+=("$s.batch_size=$b")
        done
    fi
    # tuning: every stage whose study ran (here or in TUNED_FROM_EXP) must find its best values, or it refuses to
    # train (tune.require); TUNED_FROM_EXP is where they are read from when this experiment did not tune
    local studies
    studies="$(tuned_studies)"
    if [[ -n "$studies" ]] && { ((${TUNE_THIS:-0})) || [[ -n "${TUNED_FROM_EXP:-}" ]]; }; then
        SETS+=("tune.require=[${studies// /,}]")
    fi
    [[ -n "${TUNED_FROM_EXP:-}" ]] && SETS+=("tune.from_experiment=$TUNED_FROM_EXP")
    local kv
    for kv in ${N_TRIALS:-}; do SETS+=("tune.studies.${kv%%=*}.n_trials=${kv#*=}"); done
    for s in sr decoder probe e2e; do SETS+=("$s.patience=${PATIENCE:-null}"); done
    # jet energies fit fp16, so the whole train split stays on the GPU (falls back to fp32 if a value does not)
    [[ "$DATASET" == qg ]] && SETS+=("data.preload_dtype=float16")
    local -a extra=()
    [[ -n "$EXTRA_SET" ]] && read -r -a extra <<< "$EXTRA_SET"
    SETS+=(${extra[@]+"${extra[@]}"})
}

# Near the time limit Slurm sends USR1 (sbatch --signal=B:USR1@600): requeue the job, which restarts the stage
# from its last.pt. Dependent jobs keep waiting, because a requeued job is pending again, not finished.
on_time_limit() {
    local n="${SLURM_RESTART_COUNT:-0}"
    if ((n < ${MAX_REQUEUE:-5})); then
        echo "== time limit near: requeueing job $SLURM_JOB_ID (restart $((n + 1)) of ${MAX_REQUEUE:-5})"
        scontrol requeue "$SLURM_JOB_ID"
    else
        echo "== time limit near and MAX_REQUEUE=${MAX_REQUEUE:-5} restarts used: raise TIME_* in settings.sh," \
             "then re-submit with submit_all.sh --from <stage>"
    fi
}

# run_stage "<stage> [<stage> ...]" [extra key=value ...]
# python -m sr_jepa with this experiment's paths, in the batch step itself (an srun step would not inherit -c
# from sbatch on current Slurm versions). Python runs in the background so the USR1 trap can fire.
run_stage() {
    local -a stages
    read -r -a stages <<< "$1"
    shift
    pipeline_sets
    local -a launch=(python -u -m sr_jepa)
    local gpus="${NGPUS:-1}"
    case "${stages[0]}" in
        pretrain | train_sr | train_decoder | train_probe | train_e2e)    # data parallel: one process per GPU
            if ((gpus > 1)); then
                launch=(python -u -m torch.distributed.run --standalone --nproc_per_node="$gpus" -m sr_jepa)
                export OMP_NUM_THREADS=$((PHYS_CORES > gpus ? PHYS_CORES / gpus : 1))
            fi ;;
    esac
    local -a cmd=("${launch[@]}" "${stages[@]}" -c "$REPO/configs/${CONFIG:-$DATASET}.yaml"
                  --set "${SETS[@]}" "$@")
    echo "+ ${cmd[*]}"
    local t0=$SECONDS rc pid
    "${cmd[@]}" &
    pid=$!
    trap on_time_limit USR1
    set +e
    while true; do
        wait "$pid"
        rc=$?
        kill -0 "$pid" 2> /dev/null || break    # wait was interrupted by the signal: keep waiting
    done
    set -e
    trap - USR1
    echo "-- ${stages[*]} finished in $(((SECONDS - t0) / 60)) min (exit $rc)"
    return "$rc"
}

# The standard start of every job script.
job_init() {
    : "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
    load_settings
    setup_python
    job_header "$1"
}
