"""Tuning: tuned values reach every stage, with provenance; required studies are enforced; architecture mismatches
fail clearly; every point of the default search spaces builds a working model."""
import json
import math
import os

import pytest
import torch

from sr_jepa.config import best_params_path, load_config, tuned_report
from sr_jepa.masks import sample_masks
from sr_jepa.models.build import build_decoder, build_encoder, build_mask_predictor, build_sr_predictor
from sr_jepa.train.common import arch_signature, load_modules, save_modules
from test_core import _layout


def _write_best(out, study, params):
    path = best_params_path(str(out), study)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"params": params}, f)
    return path


def test_precedence_yaml_tuned_set(tmp_path, capsys):
    _write_best(tmp_path, "sr", {"sr.lr": 1e-4, "sr.dim": 512, "sr.heads": 8})
    _write_best(tmp_path, "e2e_jepa", {"e2e.per_init.jepa.lr": 2e-5})
    cfg = load_config("configs/qg.yaml", [f"paths.out={tmp_path}", "sr.dim=256"])
    assert cfg["sr"]["lr"] == 1e-4                       # tuned beats the YAML (3e-4)
    assert cfg["sr"]["dim"] == 256 and cfg["sr"]["heads"] == 8   # --set beats tuned
    assert cfg["_tuned"]["overridden"]["sr.dim"] == {"tuned": 512, "set": 256, "study": "sr"}
    assert "WARNING: --set overrides the tuned sr.dim" in capsys.readouterr().out
    assert cfg["_tuned"]["applied"]["sr"]["source"].endswith("tuning/sr/best_params.json")
    from sr_jepa.train.e2e import section
    assert section(cfg, "jepa")["lr"] == 2e-5 and section(cfg, "scratch")["lr"] == cfg["e2e"]["lr"]
    assert load_config("configs/qg.yaml", [f"paths.out={tmp_path}"], apply_tuned=False)["sr"]["lr"] == 3e-4


def test_from_experiment_and_provenance_in_saved_config(tmp_path):
    other = tmp_path / "other"
    _write_best(other, "decoder", {"decoder.depth": 6})
    cfg = load_config("configs/qg.yaml", [f"paths.out={tmp_path / 'new'}", f"tune.from_experiment={other}"])
    assert cfg["decoder"]["depth"] == 6
    assert cfg["_tuned"]["applied"]["decoder"]["source"] == best_params_path(str(other), "decoder")
    json.dumps(cfg)  # the provenance block is saved inside every stage's config.json


def test_require_refuses_untuned_stage(tmp_path):
    cfg = load_config("configs/qg.yaml", [f"paths.out={tmp_path}", "tune.require=[sr]"])
    with pytest.raises(RuntimeError, match="tune.require lists 'sr'"):
        tuned_report(cfg, "sr")
    tuned_report(cfg, "decoder")  # not required: falls back to the config values
    _write_best(tmp_path, "sr", {"sr.lr": 1e-4})
    tuned_report(load_config("configs/qg.yaml", [f"paths.out={tmp_path}", "tune.require=[sr]"]), "sr")


def test_checkpoint_architecture_mismatch_is_explicit(tmp_path):
    cfg, L = _layout("smoke_qg")
    P = build_sr_predictor(cfg, L)
    path = str(tmp_path / "best.pt")
    save_modules(path, {"predictor": P}, arch_signature(cfg))
    load_modules(path, {"predictor": build_sr_predictor(cfg, L)}, cfg=cfg, sections=("encoder", "sr"))
    cfg2 = load_config("configs/smoke_qg.yaml", ["sr.dim=64", "sr.heads=1"])
    with pytest.raises(ValueError, match=r"sr\.dim: checkpoint 48, config 64"):
        load_modules(path, {"predictor": build_sr_predictor(cfg2, L)}, cfg=cfg2, sections=("encoder", "sr"))
    with pytest.raises(RuntimeError, match="does not match the checkpoint"):  # old checkpoints: no arch saved
        torch.save({"modules": {"predictor": P.state_dict()}}, path)
        load_modules(path, {"predictor": build_sr_predictor(cfg2, L)}, cfg=cfg2, sections=("sr",))


@pytest.mark.parametrize("name", ["qg", "calo"])
def test_default_search_spaces_build_working_models(name):
    optuna = pytest.importorskip("optuna")
    from sr_jepa.tune import sample, space_of, to_config, trial_config
    cfg, L = _layout(name)
    cfg["tune"]["studies"]["pretrain"]["search_encoder"] = True
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    g = torch.Generator().manual_seed(0)
    for s in cfg["tune"]["studies"]:
        for _ in range(6):
            t = study.ask()
            params = to_config(cfg, s, sample(t, space_of(cfg, s)))
            c = trial_config(cfg, s, params, "/tmp/unused")
            for sec in ("encoder", "predictor", "sr", "decoder"):
                d, h = c[sec]["dim"], c[sec]["heads"]
                assert d % h == 0 and d // h >= 6, (s, sec, d, h)
            if s == "pretrain":
                assert math.isclose(sum(c["pretrain"]["view_weights"]), 1.0)
                build_mask_predictor(c, L)
                enc = build_encoder(c, L)
                for spec in c["masks"]:
                    ctx, tgt = sample_masks(2, L.grid(), spec, g)
                    assert ctx.shape[1] > 0 and tgt.shape[1] > 0
            elif s == "sr":
                P = build_sr_predictor(c, L).eval()
                z = torch.randn(1, math.prod(L.grid()), c["encoder"]["dim"])
                assert P.rollout(z, L.cond(torch.tensor([1e4])), L.grid(), 2)[-1].shape == z.shape
            elif s == "decoder":
                D = build_decoder(c, L)
                z = torch.randn(1, math.prod(L.grid()), c["encoder"]["dim"])
                x = torch.zeros(1, L.C, L.D, L.H, L.W)
                assert D(z, L.grid(), x).shape == x.shape
            elif s.startswith("e2e_"):
                init = s[len("e2e_"):]
                assert set(c["e2e"]["per_init"][init]) <= {"lr", "weight_decay", "energy_weight", "encoder_lr_scale"}
            study.tell(t, 0.0)
    assert enc is not None
