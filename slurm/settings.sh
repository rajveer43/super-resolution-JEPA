# Settings for running SR-JEPA on Perlmutter. This is the ONLY file to edit.
# submit_all.sh freezes these values into $ROOT/logs/<dataset>/<experiment>/run.env at submit time, so editing this
# file later does not change jobs that are already queued.

# ---- account and resources ----------------------------------------------------------------------
ACCOUNT=mXXXX                 # NERSC project with GPU hours (some GPU allocations are named mXXXX_g)
CPU_ACCOUNT=                  # optional project with CPU hours for download + prepare; empty = use a GPU job
QOS=shared                    # shared (1 GPU, charged per GPU: recommended) | regular (whole node) | preempt
GPU_MEM=40                    # 40 | 80 (80 GB A100s are fewer, so they may queue longer)

# ---- what to run ---------------------------------------------------------------------------------
DATASET=qg                    # qg (CMS quark/gluon jets) | calo (CaloChallenge DS2)
RUN_TAG=vjepa2-baseline       # what this experiment tests, in words joined by '-' (letters, digits, '-', '.')
RUN_DATE=                     # empty = today (Asia/Kolkata) -> new experiment; e.g. 2026-10-28 to resume one
                              # -> experiment folder <RUN_DATE>_<RUN_TAG>, e.g. 2026-10-28_vjepa2-baseline
MAX_EVENTS=                   # empty = all (qg: 139k jets; calo: 100k showers from dataset_2_1)
BATCH=256                     # training batch size of every stage on one A100; learning rates scale with sqrt(batch)
PATIENCE=5                    # early stopping (epochs) for SR, decoder, probe and e2e; pretraining runs its schedule
E2E_INITS="jepa scratch"      # ablations: end-to-end fine-tuning from JEPA and training from scratch; "" = none
EXTRA_SET=""                  # extra config overrides, space separated key=value (YAML values), e.g.
                              # "pretrain.epochs=200 encoder.dim=768 encoder.heads=12"

# ---- hyperparameter tuning (Optuna; sr_jepa/tune.py) ---------------------------------------------
TUNE=0                        # 1 = an Optuna study before each stage in TUNE_STAGES; every later run of the experiment
                              # uses its best values, and a stage refuses to train if its study has no result
TUNE_STAGES="pretrain sr decoder probe e2e"   # e2e = one study per init in E2E_INITS (same budget each)
TUNE_WORKERS=4                # parallel workers per study: a job array sharing one study, one GPU each
N_TRIALS=""                   # optional, e.g. "sr=60 decoder=20"; defaults (and budgets) in sr_jepa/config.py TUNE
TUNED_FROM=                   # reuse the tuned values of an earlier experiment instead of tuning again, e.g.
                              # $SCRATCH/sr_jepa/experiments/qg/2026-10-28_vjepa2-baseline (checked at submit time)

# ---- encoder size (explicit comparison of full runs, not part of the search) ----------------------
ENCODERS="S"                  # one full pipeline per size: S = ViT-S (dim 384, depth 12) | B = ViT-B (768, 12) |
                              # L = ViT-L (1024, 24). E.g. "S B". With several, each experiment gets the suffix -vit<X>;
                              # only the first size is tuned, the others reuse its tuned values, and a final job
                              # compares them. B and L train with batch min(BATCH, 128 / 64) (memory, estimated).

# ---- where things go -----------------------------------------------------------------------------
ROOT="$SCRATCH/sr_jepa"       # raw/ cache/ experiments/ logs/ live here ($SCRATCH: fast, purged after 8 weeks unused)
CFS_COPY=                     # optional, e.g. /global/cfs/cdirs/mXXXX/sr_jepa: results are copied here at the end

# ---- wall-time limits (HH:MM:SS, max 48:00:00) ---------------------------------------------------
# Training jobs requeue themselves 10 min before the limit and resume from last.pt (at most MAX_REQUEUE times),
# so a limit that is too short costs queue time, not work.
TIME_PREPARE=03:00:00
TIME_PRETRAIN=12:00:00
TIME_SR=08:00:00
TIME_DECODER=06:00:00
TIME_PROBE=02:00:00
TIME_E2E=08:00:00
TIME_EVAL=02:00:00
TIME_TUNE=12:00:00            # per tuning worker
TIME_COMPARE=00:30:00
MAX_REQUEUE=5
