"""Configuration: generic defaults, merged with a dataset YAML and with --set overrides.

Everything is a plain nested dict so it serialises to JSON next to every result.
Dotted overrides: --set pretrain.lr=5e-4 model.patch='[1,8,8]'
"""
import copy
import json
import os

import yaml

# shared optimisation defaults of every training stage (warmup-constant-decay schedule, V-JEPA 2 section 2.4)
_OPT = {
    "batch_size": 64,
    "epochs": 50,
    "lr": 5e-4,              # peak learning rate
    "start_lr_frac": 0.1,    # warmup starts at lr * start_lr_frac
    "final_lr_frac": 0.01,   # cooldown ends at lr * final_lr_frac
    "warmup_frac": 0.05,     # of all iterations
    "cooldown_frac": 0.1,    # last fraction of iterations: linear decay (and full resolution, see pretrain)
    "weight_decay": 0.04,    # fixed, no ramp (V-JEPA 2)
    "grad_clip": 1.0,
    "max_train": None,
    "max_val": 2000,
    "patience": None,        # early stopping on the validation loss (epochs); None = off
    # lr is the peak at this batch size; another batch size scales it by sqrt(batch_size / lr_ref_batch)
    "lr_ref_batch": None,
}

BASE = {
    "run_name": "default",
    "seed": 42,
    "device": "auto",            # auto | cuda | mps | cpu
    "amp": True,                 # bf16 autocast on CUDA only
    "num_workers": 2,
    "paths": {
        "raw": "data/raw/{dataset}",
        # A cache written by superresolution-difflense (same format) can be reused as is, as long as
        # ladder.lr_factor and noise match: --set paths.cache=../superresolution-difflense/data/cache/qg
        "cache": "data/cache/{dataset}",
        "out": "runs/{dataset}/{run_name}",
    },
    "data": {
        "kind": "jet",           # jet (CMS quark/gluon images) | shower (CaloChallenge DS2); set by the YAML
        "source": "qg",          # qg | calo | synthetic
        "max_events": None,
        "split": [0.8, 0.1, 0.1],
        "import_cache": None,
        "augment": True,         # symmetry augmentation of training batches (Layout.augment)
        "preload": "auto",       # auto | gpu | cpu | none: where a split lives during training (see make_loader)
        "preload_dtype": "float32",  # float16 halves the memory (jets); falls back to float32 if values overflow
    },
    # The measured LR is HR sum-pooled by lr_factor, plus noise and topo-clustering. The ladder
    # LR -> ... -> native grid is the "time" axis of the SR world model: every level has truth.
    "ladder": {
        "lr_factor": [4, 4],
        "stages": [[2, 2], [2, 2]],
    },
    "noise": {
        "enabled": True,
        "rel_sigma": 0.1,        # per plane (list) or one value; sigma = rel_sigma * mean non-zero HR cell energy
        "topo": [4.0, 2.0, 0.0], # seed / grow / boundary thresholds in units of sigma (ATLAS-like 4-2-0)
    },
    # Model layout: (C, D, H, W). jet: C = 3 channels, D = 1. shower: C = 1, D = 45 layers (the
    # "video time" axis of V-JEPA), H x W = angle x radius. Tokens are tubelets of size `patch` (D, H, W).
    "model": {
        "patch": [1, 8, 8],
        "rope_base": 10000.0,
    },
    "encoder": {"dim": 384, "depth": 12, "heads": 6, "mlp_ratio": 4.0},
    # stage-1 mask-denoising predictor: narrow ViT, like the ViT-small predictor of V-JEPA 2
    "predictor": {"dim": 192, "depth": 6, "heads": 6, "mlp_ratio": 4.0},
    # Multiblock masking (V-JEPA): each generator yields its own context/target split per sample.
    # spatial_scale: fraction of the H x W token plane per block; depth_scale: fraction of D per block
    # (1.0 = tubes through every layer, as in V-JEPA).
    "masks": [
        {"n_blocks": 8, "spatial_scale": [0.15, 0.15], "aspect": [0.75, 1.5], "depth_scale": [1.0, 1.0]},
        {"n_blocks": 2, "spatial_scale": [0.7, 0.7], "aspect": [0.75, 1.5], "depth_scale": [1.0, 1.0]},
    ],
    "pretrain": {
        **_OPT,
        "epochs": 100,
        "lr_ref_batch": 128,
        "cooldown_frac": 0.2,    # the cooldown is also the full-resolution phase: give it a fifth of training
        "ema": 0.999,            # fixed teacher momentum (V-JEPA 2: no ramp)
        # sampling weight of each ladder level as pretraining input, LR first (cf. the VM22M data mix):
        # the frozen encoder must embed the measurement as well as the truth
        "view_weights": [0.2, 0.2, 0.6],
        # progressive resolution (V-JEPA 2 section 2.4): inputs sum-pooled by main_pool (D, H, W) during
        # warmup + constant phase, native resolution during the cooldown
        "main_pool": [1, 1, 1],
    },
    # stage 2: latent SR world model (the V-JEPA 2-AC analogue). Predicts the frozen-encoder latent of
    # ladder level k+1 from levels <= k with block-causal attention; teacher forcing + rollout loss.
    "sr": {
        **_OPT,
        "dim": 384, "depth": 8, "heads": 6, "mlp_ratio": 4.0,
        "lr_ref_batch": 64,
        "patience": 5,           # the predictor memorises after ~15 epochs on 48k jets
        "residual": True,        # predict z_{k+1} = z_k + delta
        "rollout_weight": 1.0,
    },
    # stage 3: pixel decoder latent -> HR energies, trained on frozen latents
    "decoder": {
        **_OPT,
        "dim": 256, "depth": 4, "heads": 8, "mlp_ratio": 4.0,
        "lr_ref_batch": 64,
        "patience": 5,
        "lr_skip": True,         # also sees the patch-embedded LR measurement
        "pred_latent_prob": 0.5, # share of events decoded from the SR predictor's latent instead of E(HR)
        "energy_weight": 0.1,    # relative event-energy error term
    },
    # ablation: the same encoder -> predictor -> decoder trained end to end on pixels from LR only
    "e2e": {
        **_OPT,
        "inits": ["jepa", "scratch"],
        "lr_ref_batch": 64,
        "patience": 5,
        "encoder_lr_scale": 0.1, # for the jepa init: fine-tune the pretrained encoder slowly
        "energy_weight": 0.1,
    },
    # frozen-representation check (V-JEPA attentive probe): quark/gluon (jet) or E_inc (shower) from latents
    "probe": {**_OPT, "epochs": 20, "lr": 1e-3, "weight_decay": 0.01, "heads": 4, "depth": 1, "patience": 5},
    "eval": {
        "max_events": 5000,
        "batch_size": 128,
        "ecf_top_k": 32,
        "hist_bins": 50,
        "n_examples": 4,
        "project_to_lr": False,  # rescale SR so each LR cell's energy matches the (noisy) measurement
    },
}


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def set_dotted(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse_value(s):
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def load_config(path=None, overrides=()):
    cfg = copy.deepcopy(BASE)
    if path:
        with open(path) as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    for item in overrides:
        k, _, v = item.partition("=")
        set_dotted(cfg, k.strip(), parse_value(v))
    if "dataset" not in cfg:
        raise ValueError("config needs a top-level 'dataset' name")
    for k, v in cfg["paths"].items():
        if v is not None:
            cfg["paths"][k] = v.format(dataset=cfg["dataset"], run_name=cfg["run_name"])
    return cfg


def save_config(cfg, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
