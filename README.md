# SR-JEPA: self-supervised world models for calorimeter super-resolution

Super-resolution of calorimeter images with the recipe of **V-JEPA 2** (Assran et al., 2025, arXiv:2506.09985). It has two stages. Self-supervised mask-denoising pretraining happens in representation space. It is followed by a world model that predicts in the frozen latent space. Here, resolution plays the role that time plays in the paper.

The method is applied to two datasets:

| Dataset | Content | Native grid | Measured LR | Ladder (all levels have truth) | Model layout |
|---|---|---|---|---|---|
| `qg` | CMS Open Data quark/gluon jet images (ML4Sci): tracks, ECAL, HCAL | 3 x 125 x 125, padded to 128 x 128 | 32 x 32 | 32 → 64 → 128 | image, 3 channels; 8x8 patches → 16x16 tokens |
| `calo` | CaloChallenge 2022 Dataset 2, electron showers | 45 layers x 16 angle x 9 radius | 4 x 3 per layer | 4x3 → 8x9 → 16x9 | **video** with the layers as time; 5x2x3 tubelets → 9x8x3 tokens |

**Measurement model.** The measured LR is the HR image sum-pooled, which is what a coarser detector would record. Gaussian electronic noise is then added, followed by ATLAS-like 4-2-0 topological clustering. The data pipeline and its cache format are shared with `superresolution-difflense`.

## From the paper to super-resolution

| V-JEPA 2 | SR-JEPA |
|---|---|
| Video clip, tubelets 2x16x16 | Jet image (D = 1), or shower with the 45 layers as the time axis (tubelets over layer x angle x radius) |
| 3D-RoPE over (T, H, W) | 3D-RoPE over (D, H, W). The CaloChallenge angle uses **periodic** frequencies 2πk/n, so the first and last angular bins are neighbours |
| Mask denoising, eq. 1: `‖P(Δy, E(x)) − sg(Ē(y))‖₁` on masked tokens, EMA target, multiblock masks | The same, with layer-normed targets, fixed EMA momentum and multiblock masks (short-range 8 x 0.15, long-range 2 x 0.7). For showers the short-range blocks span only part of the depth, to learn the longitudinal development |
| VideoMix22M data mix (images + videos with sampling weights) | `view_weights` mixes the noisy LR measurement, the intermediate grid and the truth. The frozen encoder must embed measurement and truth in **one** space |
| Warmup-constant-decay schedule; progressive resolution (low-res until the cooldown) | The same. `qg` pretrains on 64x64 inputs (8x8 tokens) and switches to 128x128 in the cooldown. RoPE makes this work without changing the model |
| V-JEPA 2-AC: frozen encoder, block-causal predictor over frames, with action/state tokens | Frozen encoder; block-causal predictor over **ladder levels** (LR → … → HR). Each level has a condition token: the level embedding, plus E_inc for showers, the analogue of action/state |
| Teacher forcing (eq. 2) + rollout loss (eq. 3), gradient through one step | The same. Rollout runs from the measurement alone, which is the inference path |
| Attentive probe on frozen features | Quark/gluon AUC or E_inc resolution, from latents of the truth, the measurement and the SR prediction |
| (planning with CEM) | not needed: SR is the forward rollout |

Pixels need one extra stage that the paper has no equivalent for. A **decoder** maps latent tokens, together with the LR measurement, back to HR energies. It is trained on frozen latents. Half the events are decoded from the predictor's own rollout latent, so the decoder sees the latent distribution it will meet at inference.

### Stages

1. `pretrain`: the encoder and mask predictor learn with no labels and no LR→HR pairing.
2. `train_sr`: the latent SR world model, on the frozen EMA encoder.
3. `train_decoder`: latent → HR energies. Loss: L1 on log energy densities plus the relative event-energy error.
4. `train_probe`: an attentive probe on frozen HR latents, used as an evaluation tool.
5. `train_e2e` (ablation): the same encoder → predictor → decoder trained end to end on pixels. `e2e_jepa` starts from stages 1–3; `e2e_scratch` starts from random weights. Comparing `jepa`, `e2e_jepa` and `e2e_scratch` shows what self-supervised pretraining contributes.
6. `evaluate`: every variant against the truth, with the measurement spread uniformly onto the native grid (`lr_up`, energy-preserving) as the baseline.

### Metrics

| Group | Metric |
|---|---|
| Energy | Event-energy residual: median bias and IQR/1.349, in total and per plane (jet) or layer group (shower) |
| Physics | W1/σ_truth, KS statistic and per-event Pearson r. Jets: mass, pT, girth, ΔR spread, C2/C3/D2. Showers: depth and radial centroids/widths, layer fractions, E_dep/E_inc |
| Cells | Hit precision, recall and F1; MAE and MSE of log energy densities |
| Latent | L1 to E(HR), for the predicted latent and for the re-encoded output |
| Probe | Quark/gluon AUC, or E_inc bias and resolution, for each latent |

## Usage

### Colab (recommended)

Open [`notebooks/sr_jepa_colab.ipynb`](notebooks/sr_jepa_colab.ipynb) in Colab (*File → Open notebook → GitHub*) and choose a GPU runtime. Run all cells, first with `SMOKE = True` and then with `SMOKE = False`.
- The notebook clones the repo and downloads the data once to Drive. Raw files that the DiffLense notebook already downloaded are reused.
- It runs every stage and writes logs, checkpoints, metrics and figures to `MyDrive/GSoC_SR_JEPA/experiments/<dataset>/<date>_<run-tag>/`, plus a zip in `exports/`.
- Every stage resumes after a disconnect.

### Command line

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m sr_jepa download prepare -c configs/qg.yaml     # CERNBox (3.7 GB parquet); calo: Zenodo record 6366271
# or reuse the difflense cache (same data section, same seed):
#   --set paths.cache=../superresolution-difflense/data/cache/qg

python -m sr_jepa all -c configs/qg.yaml                 # download → prepare → pretrain → … → evaluate
python -m sr_jepa all -c configs/calo.yaml

# single stages and overrides; every stage resumes from its last.pt
python -m sr_jepa pretrain -c configs/qg.yaml --set pretrain.epochs=200 encoder.dim=768 encoder.heads=12
python -m sr_jepa evaluate -c configs/calo.yaml --set eval.project_to_lr=true

# tests and tiny synthetic end-to-end runs (CPU, a few seconds)
pytest -q tests
python -m sr_jepa all -c configs/smoke_qg.yaml
python -m sr_jepa all -c configs/smoke_calo.yaml
```

Outputs go to `runs/<dataset>/<run_name>/`. Each stage writes its own directory (`pretrain/`, `sr/`, `decoder/`, `probe/`, `e2e_jepa/`, `e2e_scratch/`) holding `best.pt`, `last.pt`, `history.csv` and `config.json`. The evaluation goes to `evaluation/`: `metrics.json`, `summary.md`, `examples.png` and `observables.png`.

**What to watch in `history.csv`:**
- `pretrain`: `val_feat_std` must stay well above 0; if it falls towards 0, the representation has collapsed.
- `sr`: `val_loss` (the rollout from the LR) must fall clearly below `val_identity` (the latent error of doing nothing).
- `decoder`: `val_autoencode` is the ceiling that `val_loss` can reach.

## Layout

```
sr_jepa/
  config.py            defaults + YAML + --set overrides
  geometry.py          physical coordinates of planes and cells (for observables)
  data/                download, prepare (cache), ops (pooling, noise, topo-clustering), datasets + Layout
  masks.py             3D multiblock masks
  models/rope.py       3D-RoPE (with a periodic axis)
  models/vit.py        Encoder, MaskPredictor, LatentSRPredictor (block-causal), Decoder, AttentiveProbe
  pipeline.py          latent space, SR pipeline, checkpoints of the variants
  train/               pretrain, sr, decoder, probe, e2e, common (WCD schedule, EMA, resumable trainer)
  eval/                observables, evaluate
configs/               qg, calo, synthetic smoke runs
tests/                 invariants: energy closure, RoPE shift-invariance and periodicity, masks,
                       block-causality, patch order
```

## Status

- Unit tests pass, and every stage runs end to end on synthetic jets and showers.
- A longer synthetic jet run (2k events, 20 epochs per stage, CPU) checks that the pipeline learns. On held-out events, frozen-JEPA SR beats the `lr_up` baseline:
  - pixel MAE: 0.022 vs 0.050
  - jet-mass W1/σ: 0.016 vs 0.12
  - girth W1/σ: 0.10 vs 0.44
  - hit F1: 0.91 vs 0.89

  The latent rollout error is 0.048, against 0.164 for not super-resolving. `lr_up` keeps the better event-energy resolution, because it conserves energy exactly; `eval.project_to_lr` restores closure for SR.
- Training on the real datasets is next. The default model sizes (ViT-S encoder) are chosen for one GPU.

## Reference

M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985, 2025. Code: https://github.com/facebookresearch/vjepa2
