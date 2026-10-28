# Running SR-JEPA on Perlmutter (NERSC)

This guide runs the whole SR-JEPA pipeline on Perlmutter GPU nodes and sends the results back. You need a NERSC account in a project that has GPU hours. Everything is set in **one file**, `slurm/settings.sh`, and you run **three commands**. Every job can be resubmitted safely, because each training stage resumes from its last checkpoint.

What runs (one A100 per job; independent jobs run in parallel):

```
download + prepare ──> pretrain ──> latent SR ──> decoder ──> e2e (jepa init) ──┐
        │                  └──> probe ───────────────────────────────────────────┼──> evaluate + pack
        └──> e2e (scratch init) ─────────────────────────────────────────────────┘
```

## 1. Get the code (login node)

```bash
cd $HOME            # the code is small; data and results go to $SCRATCH
git clone https://github.com/rajveer43/super-resolution-JEPA.git
cd super-resolution-JEPA
```

To pick up later fixes, run `git pull`, preferably while none of your jobs are queued. A job uses the code that is in this folder when it starts. Its settings stay as they were when you submitted it.

## 2. Edit `slurm/settings.sh`

Usually you only change these:

| Setting | What to put |
|---|---|
| `ACCOUNT` | The NERSC project to charge. Run `iris` or see https://iris.nersc.gov for yours; some GPU allocations are named `mXXXX_g`. |
| `CPU_ACCOUNT` | Optional: a project with CPU hours, so download and prepare don't use a GPU. |
| `DATASET` | `qg` (CMS jets) or `calo` (CaloChallenge DS2). |
| `RUN_TAG` | A short name for what this experiment tests. Use a new one for each new experiment. |
| `CFS_COPY` | Optional: a CFS directory the result tarball is copied to. |

The defaults:
- all events: 139k jets or 100k showers;
- batch 256 on one A100, with learning rates scaled automatically;
- early stopping after 5 epochs without improvement;
- both ablations (`e2e_jepa` and `e2e_scratch`).

Use `EXTRA_SET` for other config overrides, e.g. `"pretrain.epochs=200 encoder.dim=768 encoder.heads=12"`. Data, caches, results and logs go to `ROOT=$SCRATCH/sr_jepa`. The raw data is downloaded once per dataset, and there is one cache per dataset and event count.

## 3. One-time setup (login node)

```bash
bash slurm/setup_env.sh
```

This loads the NERSC `pytorch` module, installs any missing packages into that module's user site, and checks that the code imports.

## 4. Smoke test, then the real run

```bash
bash slurm/submit_all.sh --smoke     # debug QOS, about 10 minutes; the log ends with "SMOKE OK"
bash slurm/submit_all.sh             # the whole chain
bash slurm/submit_all.sh --dry-run   # print the sbatch commands without submitting
```

## 5. Follow it

```bash
bash slurm/status.sh                 # queue, finished stages with their last epoch, end of each log
squeue -u $USER
```

For each stage, `status.sh` prints the last row of `history.csv`. Watch these columns:

| Stage | Column | What it should do |
|---|---|---|
| pretrain | `val_feat_std` | Stay well above 0. Falling towards 0 means the representation has collapsed. |
| any | `events_per_s` | Be in the thousands on an A100. |
| sr | `val_loss` | Fall clearly below `val_identity`, the latent error of not super-resolving. |
| decoder | `val_loss` | Approach `val_autoencode`, the decoder's ceiling. |

## Time limits, timeouts and failures

- **Time limits:** training jobs ask Slurm for a signal 10 minutes before their limit. They then requeue themselves, at most `MAX_REQUEUE` times, and resume from `last.pt`. Jobs that depend on them keep waiting. A `TIME_*` that is too short costs queue time, not work.
- **Failures:** if a job fails, fix the cause and restart from that stage. Earlier stages are not resubmitted, and the run continues the latest experiment with this `RUN_TAG`:
  ```bash
  bash slurm/submit_all.sh --from sr   # stages: prepare pretrain sr decoder probe e2e evaluate
  ```
- **Evaluation needs only the decoder.** The probe and the ablations are optional: if one fails, evaluation still runs and leaves it out.

## 6. Results

The evaluate job prints the summary table and packs `$ROOT/results_<dataset>_<experiment>.tar.gz`, which contains:
- metrics, the summary, figures, every `history.csv`, the configs and the Slurm logs;
- no checkpoints. To include the pretrained encoder and every `best.pt`, run:
  ```bash
  bash slurm/pack_results.sh <experiment> --with-checkpoints
  ```

It is copied to `CFS_COPY` if that is set. Send the tarball back.
