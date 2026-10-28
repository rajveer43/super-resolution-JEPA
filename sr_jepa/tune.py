"""Optuna hyperparameter tuning: one study per stage (CLI stages tune_pretrain, tune_sr, tune_decoder, tune_probe,
tune_e2e_jepa, tune_e2e_scratch; tune_e2e runs the e2e studies of every init in e2e.inits).

A trial is a short run of the stage's own training code (`epochs` epochs on at most `max_train` events, budgets in
config TUNE["studies"]) in {out}/tuning/<study>/trial_<n>/, trained on top of the experiment's frozen upstream
stages (paths.upstream). It reports its validation loss after every epoch, and the median pruner stops trials that
fall behind. Objectives use the validation split only; the test split is read only by `evaluate`.

    sr, decoder, probe, e2e_*   lowest validation loss of the trial (sr: the rollout from the measurement)
    pretrain                    the JEPA loss cannot rank encoders, and latent L1 values of different encoders are
                                not comparable. A trial therefore pretrains briefly, trains a short SR predictor on
                                its frozen encoder, and returns min(rollout / identity validation loss): the share
                                of the latent error left after super-resolving, which is scale-free. Trials whose
                                features collapse (val_feat_std < tune.collapse_std) are pruned.

The study lives in an Optuna journal file, which is safe for several processes on Lustre: Slurm array tasks share
one study, and a requeued job continues it. Workers stop once `n_trials` trials have COMPLETED. Pruned and failed
trials do not count; a study also stops after n_trials FAILED trials (a bug or out-of-memory, not bad luck) or
5 x n_trials finished trials in all. A trial killed by a requeue stays RUNNING in the journal and is ignored.

Outputs in {out}/tuning/<study>/: best_params.json (dotted config keys; load_config merges it into every later run
of the experiment), trials.csv, optimization_history.png, param_importances.png.
"""
import copy
import csv
import datetime
import glob
import json
import math
import os
import warnings

import torch

from .config import best_params_path, set_dotted

STAGE = {"pretrain": "pretrain", "sr": "sr", "decoder": "decoder", "probe": "probe", "e2e_jepa": "e2e",
         "e2e_scratch": "e2e"}


# ------------------------------------------------------------------ search space
def sample(trial, space):
    """Optuna suggestions for a search space (list of {key, type, ...}) -> {key: value}."""
    raw = {}
    for p in space:
        k = p["key"]
        if p["type"] == "float":
            raw[k] = trial.suggest_float(k, p["low"], p["high"], log=p.get("log", False))
        elif p["type"] == "int":
            raw[k] = trial.suggest_int(k, p["low"], p["high"], log=p.get("log", False))
        elif p["type"] == "categorical":
            raw[k] = trial.suggest_categorical(k, p["choices"])
        else:
            raise ValueError(f"unknown search-space type {p['type']!r} for {k}")
    return raw


def to_config(cfg, study, raw):
    """Sampled values -> dotted config keys. Derived keys: <section>.heads = dim // 64 (head dim 64, which RoPE
    needs >= 6), the block scale of the first / last mask generator, view_weights from the LR share."""
    out = {}
    for k, v in raw.items():
        if k in ("masks.short_scale", "masks.long_scale"):
            masks = copy.deepcopy(out.get("masks", cfg["masks"]))
            masks[0 if k == "masks.short_scale" else -1]["spatial_scale"] = [v, v]
            out["masks"] = masks
        elif k == "pretrain.lr_view_share":
            w = cfg["pretrain"]["view_weights"]
            rest = sum(w[1:]) or 1.0
            out["pretrain.view_weights"] = [v] + [x / rest * (1 - v) for x in w[1:]]
        else:
            out[k] = v
            sec, _, name = k.rpartition(".")
            if name == "dim":
                out[f"{sec}.heads"] = max(1, int(v) // 64)
    if study.startswith("e2e_"):  # each init has its own e2e settings
        init = study[len("e2e_"):]
        out = {(f"e2e.per_init.{init}." + k[len("e2e."):]) if k.startswith("e2e.") else k: v for k, v in out.items()}
    return out


def space_of(cfg, study):
    b = cfg["tune"]["studies"][study]
    space = list(b["space"])
    if study == "pretrain" and b.get("search_encoder"):
        space += b["encoder_space"]
    return space


def trial_config(cfg, study, params, trial_dir):
    """The experiment's config with the trial's values and the study's short budget; output in trial_dir,
    upstream checkpoints from the experiment (pretraining trials need none)."""
    c = copy.deepcopy(cfg)
    for k, v in params.items():
        set_dotted(c, k, copy.deepcopy(v))
    c["paths"]["upstream"] = None if study == "pretrain" else (cfg["paths"].get("upstream") or cfg["paths"]["out"])
    c["paths"]["out"] = trial_dir
    c["tune"]["require"] = []  # a trial trains with the values it samples, tuned or not
    c["_tuned"]["trial"] = {"study": study, "params": params}
    b = cfg["tune"]["studies"][study]
    budget = {"epochs": b["epochs"], "max_train": b["max_train"], "max_val": b["max_val"], "patience": None}
    c[STAGE[study]].update(budget)
    if study == "pretrain":
        c["sr"].update(budget, epochs=b["proxy_sr_epochs"])
    return c


# ------------------------------------------------------------------ objective
def _history(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _reporter(trial, value=None, collapse=None):
    """Trainer on_epoch hook: report value(val) to Optuna and prune; prune collapsed pretraining."""
    import optuna

    def hook(epoch, val):
        if collapse is not None and val.get("feat_std", math.inf) < collapse:
            raise optuna.TrialPruned(f"features collapsed: val_feat_std={val['feat_std']:.3g} < {collapse}")
        if value is None:
            return
        trial.report(value(val), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return hook


def _loss(val):
    return float(val["loss"])


def _ratio(val):
    return float(val["loss"]) / max(float(val["identity"]), 1e-12)


def run_trial(c, study, trial, trial_dir):
    from .train import decoder, e2e, pretrain, probe, sr
    if study == "pretrain":
        pretrain.run(c, _reporter(trial, collapse=c["tune"]["collapse_std"]))
        sr.run(c, _reporter(trial, _ratio))
        h = _history(os.path.join(trial_dir, "sr", "history.csv"))
        return min(float(r["val_loss"]) / max(float(r["val_identity"]), 1e-12) for r in h)
    if study.startswith("e2e_"):
        init = study[len("e2e_"):]
        e2e.run(c, init, _reporter(trial, _loss))
        folder = f"e2e_{init}"
    else:
        {"sr": sr, "decoder": decoder, "probe": probe}[study].run(c, _reporter(trial, _loss))
        folder = study
    return min(float(r["val_loss"]) for r in _history(os.path.join(trial_dir, folder, "history.csv")))


def _drop_checkpoints(trial_dir):
    for f in glob.glob(os.path.join(trial_dir, "**", "*.pt"), recursive=True):
        os.remove(f)


def make_objective(cfg, study, study_dir):
    space = space_of(cfg, study)

    def objective(trial):
        params = to_config(cfg, study, sample(trial, space))
        trial.set_user_attr("config_params", params)
        trial_dir = os.path.join(study_dir, f"trial_{trial.number:04d}")
        print(f"\n[tune {study}] trial {trial.number}: {params}")
        try:
            return run_trial(trial_config(cfg, study, params, trial_dir), study, trial, trial_dir)
        finally:
            _drop_checkpoints(trial_dir)  # keep config.json + history.csv only
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return objective


# ------------------------------------------------------------------ study
def storage(path):
    """Journal-file storage with an open()-based lock (works on Lustre / NFS); Optuna >= 4 and 3.x APIs."""
    import optuna
    try:
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
        backend = JournalFileBackend(path, lock_obj=JournalFileOpenLock(path))
    except ImportError:
        from optuna.storages import JournalFileOpenLock, JournalFileStorage
        backend = JournalFileStorage(path, lock_obj=JournalFileOpenLock(path))
    return optuna.storages.JournalStorage(backend)


def open_study(cfg, study):
    import optuna
    d = os.path.join(cfg["paths"]["out"], "tuning", study)
    os.makedirs(d, exist_ok=True)
    store = storage(os.path.join(d, "study.journal"))
    name = f"{cfg['dataset']}_{study}"
    try:
        done = len(optuna.load_study(study_name=name, storage=store).get_trials(deepcopy=False))
    except KeyError:
        done = 0
    worker = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0") or 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        # seeded per worker and by the trials already run: parallel workers, and a worker restarted after a
        # requeue, would otherwise propose the same first points again
        sampler = optuna.samplers.TPESampler(seed=int(cfg["seed"]) + 1000 * worker + done, multivariate=True)
    pr = cfg["tune"]["pruner"]
    pruner = optuna.pruners.MedianPruner(n_startup_trials=pr["startup_trials"], n_warmup_steps=pr["warmup_epochs"])
    st = optuna.create_study(study_name=name, storage=store, direction="minimize", sampler=sampler, pruner=pruner,
                             load_if_exists=True)
    return st, d


def counts(st):
    from optuna.trial import TrialState
    states = [t.state for t in st.get_trials(deepcopy=False)]
    return {s.name.lower(): states.count(s) for s in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL,
                                                       TrialState.RUNNING)}


def run(cfg, study):
    import optuna
    if study not in cfg["tune"]["studies"]:
        raise ValueError(f"unknown study {study}; choose from {list(cfg['tune']['studies'])}")
    n = int(cfg["tune"]["studies"][study]["n_trials"])
    st, d = open_study(cfg, study)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def finished(c):
        return c["complete"] >= n or c["fail"] >= n or c["complete"] + c["pruned"] + c["fail"] >= 5 * n

    def stop(s, _):
        if finished(counts(s)):
            s.stop()

    c = counts(st)
    print(f"[tune {study}] study {d}: {c} (target {n} completed trials)")
    if not finished(c):
        st.optimize(make_objective(cfg, study, d), callbacks=[stop], catch=(RuntimeError,), gc_after_trial=True)
    return write_results(cfg, study, st, d)


def write_results(cfg, study, st, d):
    """best_params.json, trials.csv and plots from the study's current state (every worker rewrites them)."""
    from optuna.trial import TrialState
    trials = st.get_trials(deepcopy=False)
    keys = sorted({k for t in trials for k in t.params})
    tmp = os.path.join(d, f"trials.csv.{os.getpid()}.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "state", "value", "minutes"] + keys)
        for t in trials:
            dur = (t.datetime_complete - t.datetime_start).total_seconds() / 60 if t.datetime_complete else ""
            w.writerow([t.number, t.state.name, t.value, dur] + [t.params.get(k, "") for k in keys])
    os.replace(tmp, os.path.join(d, "trials.csv"))
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    c = counts(st)
    if not complete:
        print(f"[tune {study}] no completed trial yet ({c}): best_params.json not written")
        return None
    best = min(complete, key=lambda t: t.value)
    payload = {"study": study, "value": best.value, "trial": best.number, "params": best.user_attrs["config_params"],
               "optuna_params": best.params, "trials": c, "budget": {k: v for k, v in
                                                                       cfg["tune"]["studies"][study].items()
                                                                       if k not in ("space", "encoder_space")},
               "written": datetime.datetime.now().isoformat(timespec="seconds")}
    path = best_params_path(cfg["paths"]["out"], study)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    print(f"[tune {study}] best trial {best.number}: value {best.value:.5g} -> {path}\n  {payload['params']}")
    _plots(st, d, len(complete))
    return payload


def _plots(st, d, n_complete):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import optuna.visualization.matplotlib as vis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = vis.plot_optimization_history(st)
            ax.figure.savefig(os.path.join(d, "optimization_history.png"), dpi=120, bbox_inches="tight")
            plt.close(ax.figure)
            if n_complete >= 3:
                ax = vis.plot_param_importances(st)
                ax.figure.savefig(os.path.join(d, "param_importances.png"), dpi=120, bbox_inches="tight")
                plt.close(ax.figure)
    except Exception as e:  # plots are a convenience; never fail a study over them
        print(f"(plots skipped: {e})")
