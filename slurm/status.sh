#!/bin/bash
# Progress of one experiment: queued / running jobs, what is finished, the last epoch of each stage, and the end
# of the latest logs.
#
#   bash slurm/status.sh                 the most recently submitted experiment
#   bash slurm/status.sh <experiment>    a given one, e.g. 2026-10-28_vjepa2-baseline
#   bash slurm/status.sh --smoke         the latest smoke test
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings

load_experiment "${1:-}"
exp="$(experiment_dir)"
logdir="$(dirname "$RUN_ENV")"

echo "experiment $EXPERIMENT  ($exp)"
echo
echo "---- jobs in the queue"
ids="$(cut -f3 "$logdir/jobs.tsv" 2> /dev/null | paste -sd, -)"
if [[ -z "$ids" ]]; then
    echo "(no jobs submitted yet)"
elif command -v squeue > /dev/null; then
    squeue -j "$ids" -o '%.14i %.36j %.9T %.10M %.10l %.24R' 2> /dev/null || echo "(none of this experiment's jobs are queued)"
else
    echo "(squeue not available)"
fi

mark() { if compgen -G "$1" > /dev/null; then echo "done   "; else echo "-      "; fi; }
# last row of a history.csv as "epoch N | key=value ..." for the columns that matter
last_epoch() {
    [[ -f "$1" ]] || return 0
    python - "$1" << 'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if rows:
    r = rows[-1]
    keys = [k for k in r if k.startswith("val_") or k in ("train_loss", "events_per_s")]
    print("        epoch", r["epoch"], "|", " ".join(f"{k}={float(r[k]):.4g}" for k in keys))
EOF
}

echo
echo "---- finished work"
echo "$(mark "$(cache_dir)/meta.json") prepare   $(cache_dir)"
for s in pretrain sr decoder probe e2e_jepa e2e_scratch; do
    f=best.pt
    [[ "$s" == pretrain ]] && f=last.pt   # later stages use the final EMA encoder
    echo "$(mark "$exp/$s/$f") $s"
    last_epoch "$exp/$s/history.csv"
done
echo "$(mark "$exp/evaluation/summary.md") evaluate  evaluation/summary.md"

echo
echo "---- tuning (studies of this experiment, or the values it reuses)"
python3 - "$exp" "${TUNED_FROM_EXP:-}" "$(tuned_studies)" << 'EOF'
import csv, json, os, sys
exp, reused, studies = sys.argv[1], sys.argv[2], sys.argv[3].split()
if not studies:
    print("(TUNE_STAGES is empty)")
for s in studies:
    d = os.path.join(reused or exp, "tuning", s)
    states = {}
    if os.path.exists(os.path.join(d, "trials.csv")):
        for r in csv.DictReader(open(os.path.join(d, "trials.csv"))):
            states[r["state"].lower()] = states.get(r["state"].lower(), 0) + 1
    best = os.path.join(d, "best_params.json")
    line = f"{s:12s} trials {states or '-'}"
    if os.path.exists(best):
        b = json.load(open(best))
        line += f" | best {b['value']:.4g} (trial {b['trial']})"
    print(line + ("" if not reused else f"  [from {reused}]"))
    if os.path.exists(best):
        print("             " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in b["params"].items() if k != "masks"))
    cfg = os.path.join(exp, s, "config.json")
    if os.path.exists(cfg):
        applied = json.load(open(cfg)).get("_tuned", {}).get("applied", {})
        print("             final stage trained with the tuned values" if s in applied
              else "             !! final stage trained WITHOUT this study's values")
EOF
compgen -G "$ROOT/results_${DATASET}_${EXPERIMENT}*.tar.gz" > /dev/null \
    && echo "packed: $(ls "$ROOT"/results_"${DATASET}"_"${EXPERIMENT}"*.tar.gz)"

echo
echo "---- end of the latest log of each stage ($logdir)"
for stage in prepare tune-pretrain pretrain tune-sr sr tune-decoder decoder tune-probe probe tune-e2e_scratch \
    e2e-scratch tune-e2e_jepa e2e-jepa evaluate smoke; do
    # shellcheck disable=SC2012
    f="$(ls -t "$logdir/$stage"-*.out 2> /dev/null | head -1)"
    [[ -n "$f" ]] || continue
    echo "== $(basename "$f")"
    tail -n 3 "$f"
done
grep -l -i -E "traceback|error:|out of memory|DUE TO TIME LIMIT" "$logdir"/*.out 2> /dev/null \
    | sed 's/^/!! problem reported in /'
exit 0
