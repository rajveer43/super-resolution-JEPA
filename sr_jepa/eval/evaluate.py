"""Evaluation on the test split: every available SR variant against the truth, with the measurement spread
uniformly onto the native grid (lr_up, energy-preserving, no learning) as the baseline.

Metrics (streamed over batches; only per-event scalars and a few example images are kept):
  energy      event energy residual (E - E_true) / E_true: median bias and IQR/1.349 resolution, total and per
              plane (jet) or layer group (shower)
  pixels      MAE / MSE of log energy densities (the training space)
  hits        cell precision / recall / F1 above the readout threshold
  physics     W1 / sigma_truth, KS statistic and per-event Pearson r of jet / shower observables
  latent      L1 to E(HR) in the frozen JEPA latent space: predicted latent (jepa) and re-encoded output
  probe       the frozen-feature probe applied to each latent: quark/gluon AUC (jet), E_inc bias /
              resolution (shower)
Outputs: {out}/evaluation/metrics.json, summary.md, examples.png, observables.png
"""
import json
import os

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

from ..data.datasets import make_loader
from ..data.ops import uniform_upsample
from ..geometry import Geometry
from ..models.build import build_probe
from ..pipeline import VARIANTS, available, frozen_encoder, latents, load_pipeline, super_resolve
from ..train.common import load_modules, setup, stage_dir
from .observables import dense_to_cells, observables


def _resolution(r):
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if not len(r):
        return {"bias": float("nan"), "resolution": float("nan")}
    q25, q50, q75 = np.percentile(r, [25, 50, 75])
    return {"bias": float(q50), "resolution": float((q75 - q25) / 1.349)}


def _groups(L, geom):
    if L.kind == "jet":
        return {name: [p] for p, name in enumerate(geom.planes)}
    return {f"layers_{a}_{b}": list(range(a, b)) for a, b in geom.layer_groups}


@torch.no_grad()
def run(cfg):
    device, meta, L = setup(cfg)
    ev = cfg["eval"]
    geom = Geometry(cfg)
    thr = np.full(L.P, L.hit_threshold)
    out = stage_dir(cfg, "evaluation")
    variants = [v for v in VARIANTS if available(cfg, v)]
    if not variants:
        raise FileNotFoundError("no trained SR variant found: run pretrain, train_sr, train_decoder (or train_e2e)")
    pipes = {v: load_pipeline(cfg, L, device, v) for v in variants}
    enc = frozen_encoder(cfg, L, device)
    probe = None
    probe_ck = os.path.join(cfg["paths"]["out"], "probe", "best.pt")
    if os.path.exists(probe_ck):
        probe = build_probe(cfg, 1)
        load_modules(probe_ck, {"probe": probe})
        probe = probe.to(device).eval()
    methods = ["lr_up"] + variants
    print(f"evaluating {methods} on up to {ev['max_events']} test events")

    groups = _groups(L, geom)
    rec = {m: {"E": [], "planes": [], "mae": [], "mse": [], "tp": 0, "fp": 0, "fn": 0, "latent": [],
               "latent_pred": [], "probe": [], "obs": []} for m in methods}
    truth = {"E": [], "planes": [], "obs": [], "target": [], "probe": []}
    examples = None
    loader = make_loader(cfg, "test", ev["batch_size"], False, ev["max_events"])
    for _, hr, lr, t in loader:
        hr, lr, t = hr.to(device), lr.to(device), t.to(device)
        x_hr = L.normalize(hr)
        z_hr = latents(enc, x_hr)
        outs = {"lr_up": (uniform_upsample(lr, L._factor(L.levels[0])), None)}
        for v, pipe in pipes.items():
            outs[v] = super_resolve(pipe, lr, t, ev["project_to_lr"])
        hr_np = hr.cpu().numpy()
        truth["E"].append(hr_np.sum((1, 2, 3)))
        truth["planes"].append(hr_np.sum((2, 3)))
        truth["target"].append(t.cpu().numpy())
        truth["obs"].append(observables(dense_to_cells(hr_np), len(hr_np), geom, thr, t.cpu().numpy(),
                                        ev["ecf_top_k"]))
        if probe is not None:
            truth["probe"].append(probe(z_hr)[:, 0].float().cpu().numpy())
        hit_true = hr > L.hit_threshold
        for m, (E, z_pred) in outs.items():
            r = rec[m]
            E_np = E.cpu().numpy()
            r["E"].append(E_np.sum((1, 2, 3)))
            r["planes"].append(E_np.sum((2, 3)))
            d = L.normalize(E) - x_hr
            r["mae"].append(d.abs().mean(dim=(1, 2, 3, 4)).cpu().numpy())
            r["mse"].append((d ** 2).mean(dim=(1, 2, 3, 4)).cpu().numpy())
            hit = E > L.hit_threshold
            r["tp"] += int((hit & hit_true).sum())
            r["fp"] += int((hit & ~hit_true).sum())
            r["fn"] += int((~hit & hit_true).sum())
            z_re = latents(enc, L.normalize(E))
            r["latent"].append((z_re - z_hr).abs().mean(dim=(1, 2)).cpu().numpy())
            z_for_probe = z_re
            if z_pred is not None and m == "jepa":  # same latent space as the frozen encoder
                r["latent_pred"].append((z_pred - z_hr).abs().mean(dim=(1, 2)).cpu().numpy())
                z_for_probe = z_pred
            if probe is not None:
                r["probe"].append(probe(z_for_probe)[:, 0].float().cpu().numpy())
            r["obs"].append(observables(dense_to_cells(E_np), len(E_np), geom, thr, t.cpu().numpy(),
                                        ev["ecf_top_k"]))
        if examples is None:
            k = min(ev["n_examples"], len(hr_np))
            examples = {"hr": hr_np[:k], **{m: outs[m][0][:k].cpu().numpy() for m in methods}}

    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    cat_obs = lambda xs: {k: np.concatenate([x[k] for x in xs]) for k in xs[0]}
    tE, tP, tobs, target = cat(truth["E"]), cat(truth["planes"]), cat_obs(truth["obs"]), cat(truth["target"])
    metrics = {"n_events": int(len(tE)), "methods": {}}
    for m in methods:
        r = rec[m]
        E, Pl, obs = cat(r["E"]), cat(r["planes"]), cat_obs(r["obs"])
        res = {"energy": {"total": _resolution((E - tE) / np.maximum(tE, 1e-12))}}
        for name, idx in groups.items():
            e_t, e_m = tP[:, idx].sum(1), Pl[:, idx].sum(1)
            ok = e_t > 0
            res["energy"][name] = _resolution((e_m[ok] - e_t[ok]) / e_t[ok])
        res["pixels"] = {"mae_log": float(cat(r["mae"]).mean()), "mse_log": float(cat(r["mse"]).mean())}
        p = r["tp"] / max(r["tp"] + r["fp"], 1)
        rc = r["tp"] / max(r["tp"] + r["fn"], 1)
        res["hits"] = {"precision": p, "recall": rc, "f1": 2 * p * rc / max(p + rc, 1e-12)}
        res["latent"] = {"reencoded_l1": float(cat(r["latent"]).mean())}
        if r["latent_pred"]:
            res["latent"]["predicted_l1"] = float(cat(r["latent_pred"]).mean())
        res["observables"] = {}
        for k in tobs:
            a, b = obs[k], tobs[k]
            ok = np.isfinite(a) & np.isfinite(b)
            sd = max(float(np.std(b[ok])), 1e-12)
            res["observables"][k] = {
                "w1_over_sigma": float(wasserstein_distance(a[ok], b[ok]) / sd),
                "ks": float(ks_2samp(a[ok], b[ok]).statistic),
                "pearson_r": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and np.std(a[ok]) > 0 else 0.0}
        if probe is not None:
            res["probe"] = _probe_metrics(L, cat(r["probe"]), target)
        metrics["methods"][m] = res
    if probe is not None:
        metrics["probe_on_truth"] = _probe_metrics(L, cat(truth["probe"]), target)

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(summary_markdown(metrics, methods))
    _figures(out, L, geom, examples, {m: cat_obs(rec[m]["obs"]) for m in methods}, tobs, ev["hist_bins"])
    print(summary_markdown(metrics, methods))
    return metrics


def _probe_metrics(L, pred, target):
    if L.kind == "jet":
        from sklearn.metrics import roc_auc_score
        return {"auc": float(roc_auc_score(target, pred))} if len(np.unique(target)) == 2 else {}
    e_pred = 10 ** ((pred + 1.0) * 1.5 + 3.0)  # inverse of Layout.cond
    return {f"e_inc_{k}": v for k, v in _resolution((e_pred - target) / target).items()}


def summary_markdown(metrics, methods):
    M = metrics["methods"]
    lines = [f"# SR-JEPA evaluation ({metrics['n_events']} test events)", "",
             "| metric | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]

    def row(name, get):
        vals = []
        for m in methods:
            try:
                vals.append(f"{get(M[m]):.4g}")
            except (KeyError, TypeError):
                vals.append("-")
        lines.append(f"| {name} | " + " | ".join(vals) + " |")

    row("energy bias (median)", lambda r: r["energy"]["total"]["bias"])
    row("energy resolution (IQR/1.349)", lambda r: r["energy"]["total"]["resolution"])
    row("pixel MAE (log density)", lambda r: r["pixels"]["mae_log"])
    row("hit F1", lambda r: r["hits"]["f1"])
    row("latent L1 to E(HR), re-encoded", lambda r: r["latent"]["reencoded_l1"])
    row("latent L1 to E(HR), predicted", lambda r: r["latent"]["predicted_l1"])
    obs = list(M[methods[0]]["observables"])
    row("mean W1/sigma over observables", lambda r: float(np.mean([v["w1_over_sigma"]
                                                                    for v in r["observables"].values()])))
    if "probe_on_truth" in metrics:
        key = next(iter(metrics["probe_on_truth"]), None)
        if key:
            row(f"probe {key} (truth: {metrics['probe_on_truth'][key]:.4g})", lambda r: r["probe"][key])
    lines += ["", "## Observables: W1 / sigma_truth (KS) [Pearson r]", "",
              "| observable | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for k in obs:
        lines.append(f"| {k} | " + " | ".join(
            "{w1_over_sigma:.3f} ({ks:.3f}) [{pearson_r:.3f}]".format(**M[m]["observables"][k]) for m in methods) + " |")
    return "\n".join(lines) + "\n"


def _figures(out, L, geom, examples, obs, tobs, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [m for m in examples if m != "hr"]
    cols = methods + ["hr"]
    k = len(examples["hr"])
    plane = 1 if L.kind == "jet" and L.P > 1 else None   # ECAL for jets; layer sum for showers
    fig, axes = plt.subplots(k, len(cols), figsize=(2.6 * len(cols), 2.6 * k), squeeze=False)
    for i in range(k):
        ref = examples["hr"][i][plane] if plane is not None else examples["hr"][i].sum(0)
        vmax = max(float(ref.max()), 1e-6)
        for j, c in enumerate(cols):
            img = examples[c][i][plane] if plane is not None else examples[c][i].sum(0)
            axes[i, j].imshow(np.log10(np.maximum(img, vmax * 1e-4)), vmin=np.log10(vmax * 1e-4),
                              vmax=np.log10(vmax), cmap="inferno", aspect="auto")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(c)
    fig.suptitle(f"{'ECAL' if plane is not None else 'energy summed over layers'} (log10)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "examples.png"), dpi=120)
    plt.close(fig)

    keys = list(tobs)
    nc = 4
    nr = (len(keys) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.7 * nr), squeeze=False)
    for a, key in zip(axes.flat, keys):
        b = tobs[key][np.isfinite(tobs[key])]
        lo, hi = np.percentile(b, [0.5, 99.5]) if len(b) else (0, 1)
        edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
        a.hist(b, edges, histtype="stepfilled", alpha=0.3, color="k", label="hr")
        for m in methods:
            a.hist(obs[m][key][np.isfinite(obs[m][key])], edges, histtype="step", label=m)
        a.set_title(key, fontsize=9)
    for a in axes.flat[len(keys):]:
        a.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "observables.png"), dpi=120)
    plt.close(fig)
