"""python -m sr_jepa <stage> [<stage> ...] -c configs/qg.yaml [--set key=value ...] [--force]

Stages, in pipeline order:
  download       raw data (qg: CERNBox share of the ML4Sci jet images; calo: CaloChallenge DS2 from Zenodo)
  prepare        raw -> cache of HR truth, noisy LR measurement and targets (train-only statistics)
  pretrain       stage 1: V-JEPA 2 mask-denoising pretraining of the encoder (self-supervised)
  train_sr       stage 2: latent SR world model on the frozen encoder (teacher forcing + rollout)
  train_decoder  stage 3: pixel decoder latent -> HR energies
  train_probe    attentive probe on frozen latents (quark/gluon or E_inc), for the evaluation
  train_e2e      ablation: end-to-end fine-tuning (init jepa) and training from scratch (init scratch)
  evaluate       physics + latent metrics of every trained variant on the test split
  all            every stage above, in order; with tune.enabled=true each preceded by its Optuna study

Tuning (sr_jepa/tune.py): tune_pretrain, tune_sr, tune_decoder, tune_probe, tune_e2e_jepa, tune_e2e_scratch,
tune_e2e (the studies of every init in e2e.inits). Each writes {out}/tuning/<study>/best_params.json, which every
later stage of the experiment uses (see load_config).
"""
import argparse

from .config import TUNE_STUDIES, load_config

ORDER = ["download", "prepare", "pretrain", "train_sr", "train_decoder", "train_probe", "train_e2e", "evaluate"]
TUNE_BEFORE = {"pretrain": ["tune_pretrain"], "train_sr": ["tune_sr"], "train_decoder": ["tune_decoder"],
               "train_probe": ["tune_probe"], "train_e2e": ["tune_e2e"]}


def run_stage(stage, cfg, force=False):
    if stage == "download":
        from .data.download import download
        download(cfg)
    elif stage == "prepare":
        from .data.prepare import prepare
        prepare(cfg, force=force)
    elif stage == "pretrain":
        from .train import pretrain
        pretrain.run(cfg)
    elif stage == "train_sr":
        from .train import sr
        sr.run(cfg)
    elif stage == "train_decoder":
        from .train import decoder
        decoder.run(cfg)
    elif stage == "train_probe":
        from .train import probe
        probe.run(cfg)
    elif stage == "train_e2e":
        from .train import e2e
        for init in cfg["e2e"]["inits"]:
            e2e.run(cfg, init)
    elif stage == "evaluate":
        from .eval import evaluate
        evaluate.run(cfg)
    elif stage == "tune_e2e":
        from . import tune
        for init in cfg["e2e"]["inits"]:
            tune.run(cfg, f"e2e_{init}")
    elif stage.startswith("tune_") and stage[len("tune_"):] in TUNE_STUDIES:
        from . import tune
        tune.run(cfg, stage[len("tune_"):])
    else:
        raise ValueError(f"unknown stage {stage}; choose from {ORDER + ['all', 'tune_e2e']} or "
                         f"{['tune_' + s for s in TUNE_STUDIES]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sr_jepa", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stages", nargs="+")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--force", action="store_true", help="rebuild the data cache")
    a = ap.parse_args(argv)
    stages = a.stages
    if stages == ["all"]:
        tuned = load_config(a.config, a.set)["tune"]["enabled"]
        stages = [t for s in ORDER for t in (TUNE_BEFORE.get(s, []) if tuned else []) + [s]]
    for s in stages:
        print(f"\n===== {s} =====")
        # reloaded per stage: a study that just finished must reach the stages after it
        run_stage(s, load_config(a.config, a.set), a.force)


if __name__ == "__main__":
    main()
