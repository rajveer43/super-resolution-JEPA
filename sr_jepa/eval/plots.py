"""Evaluation figures, drawn from the per-event numbers that `evaluate` saves (evaluation/per_event.npz), so they
can be redrawn without evaluating again:

    python -m sr_jepa.eval.plots <experiment folder>

examples.png            a few test events, one per row, every method next to the truth (jets: channels summed;
                        showers: radius x layer, summed over angle; LR as energy per fine pixel)
channels_event<i>.png   one test event: rows truth / LR / each method, columns tracks / ECAL / HCAL (jets) or
                        early / middle / late layers (showers); each column shares one colour scale
observables.png         distribution of every observable, each method against the truth (overview grid)
hist/hist_<obs>.png     one observable: normalised distributions + ratio to the truth, % outside the range
scatter/scatter_<obs>.png  one observable per event, method against truth; bias, resolution and r per panel
scatter_observables.png per-event SR against truth for the key observables, one panel per method
profile_<name>.png      mean energy fraction against radius (jets: dR from the jet axis; showers: r) and, for
                        showers, against layer; with the ratio to the truth
w1_heatmap.png          W1 / sigma_truth for every observable x method (lower is better)
energy_residuals.png    (E - E_true) / E_true per event, in total and per plane / layer group
training_curves.png     train and validation loss per epoch of every stage (from each history.csv)
probe_roc.png           jets: quark/gluon ROC of the frozen-feature probe on every method's latent
energy_vs_einc.png      showers: bias and resolution of the event energy against the incident energy

Colours follow the method in every figure (validated palette: the three learned methods use hues that stay
distinguishable for colour-blind readers in every pair; the baseline is dashed grey, the truth a grey fill).
"""
import argparse
import csv
import json
import os

import numpy as np

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a988f", "#e6e5e0"
STYLE = {
    "truth": {"color": "#b5b3ab", "label": "truth (HR)"},
    "lr_up": {"color": INK2, "ls": "--", "label": "LR spread (baseline)"},
    "jepa": {"color": "#2a78d6", "ls": "-", "label": "JEPA (frozen encoder)"},
    "e2e_jepa": {"color": "#eb6834", "ls": "-", "label": "JEPA, fine-tuned end to end"},
    "e2e_scratch": {"color": "#1baf7a", "ls": "-", "label": "same network, from scratch"},
}
KEY_OBSERVABLES = {"jet": ["jet_mass", "jet_pt", "girth"], "shower": ["depth_centroid", "r_centroid", "frac_early"]}
STAGES = [("pretrain", "pretraining (JEPA)"), ("sr", "latent SR predictor"), ("decoder", "decoder"),
          ("probe", "probe"), ("e2e_jepa", "e2e from JEPA"), ("e2e_scratch", "e2e from scratch")]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 2.0, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "legend.fontsize": 8})
    return plt


def _save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _range(arrays, lo=1, hi=99):
    v = np.concatenate([a[np.isfinite(a)] for a in arrays if len(a)])
    if not len(v):
        return 0.0, 1.0
    a, b = np.percentile(v, [lo, hi])
    return (a, b) if b > a else (a - 0.5, a + 0.5)


def _resolution(r):
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return np.nan, np.nan
    q25, q50, q75 = np.percentile(r, [25, 50, 75])
    return q50, (q75 - q25) / 1.349


# ------------------------------------------------------------------ data
def save_per_event(out, data):
    """data: {"kind", "planes", "groups", "methods", "truth": {...}, "m": {method: {...}}, "examples": {...}}."""
    flat = {}
    for who, d in [("truth", data["truth"])] + [(m, data["m"][m]) for m in data["methods"]]:
        for k, v in d.items():
            if isinstance(v, dict):  # obs, profiles
                for o, a in v.items():
                    flat[f"{who}/{k}/{o}"] = a
            elif v is not None and len(v):
                flat[f"{who}/{k}"] = v
    for k, v in data["examples"].items():
        flat[f"examples/{k}"] = v
    np.savez_compressed(os.path.join(out, "per_event.npz"), **flat)
    with open(os.path.join(out, "per_event_meta.json"), "w") as f:
        json.dump({k: data[k] for k in ("kind", "planes", "groups", "methods", "profile_axes") if k in data}, f,
                  indent=2)


def load_per_event(out):
    with open(os.path.join(out, "per_event_meta.json")) as f:
        data = json.load(f)
    z = np.load(os.path.join(out, "per_event.npz"))
    data["truth"], data["m"], data["examples"] = {"obs": {}}, {m: {"obs": {}} for m in data["methods"]}, {}
    for key in z.files:
        parts = key.split("/")
        if parts[0] == "examples":
            data["examples"][parts[1]] = z[key]
            continue
        d = data["truth"] if parts[0] == "truth" else data["m"][parts[0]]
        if len(parts) == 3:
            d.setdefault(parts[1], {})[parts[2]] = z[key]
        else:
            d[parts[1]] = z[key]
    return data


def profile_axes(kind, geom):
    """The radial / longitudinal binning of the per-event energy profiles (stored with the per-event data)."""
    if kind == "jet":
        edges = np.linspace(0.0, 0.8, 17)
        return {"radial_profile": {"edges": edges.tolist(), "centres": (0.5 * (edges[1:] + edges[:-1])).tolist(),
                                   "label": "dR from the jet axis"}}
    U, V = geom.native
    return {"radial_profile": {"centres": geom.r_centers(V).tolist(), "label": "r [mm]"},
            "longitudinal_profile": {"centres": list(range(geom.P)), "label": "layer"}}


def event_profiles(kind, geom, axes, E):
    """Energy fraction per radial (and longitudinal) bin of every event; E: (B, P, U, V) numpy, linear energies."""
    tot = np.maximum(E.sum((1, 2, 3)), 1e-30)[:, None]
    if kind == "jet":
        U, V = geom.native
        u, v = np.meshgrid(np.arange(U), np.arange(V), indexing="ij")
        dr = np.hypot(u - geom.center, v - geom.center) * geom.pixel
        edges = np.asarray(axes["radial_profile"]["edges"])
        idx = np.digitize(dr.ravel(), edges) - 1
        ok = (idx >= 0) & (idx < len(edges) - 1)
        M = np.zeros((U * V, len(edges) - 1))
        M[np.arange(U * V)[ok], idx[ok]] = 1.0
        return {"radial_profile": E.sum(1).reshape(len(E), -1) @ M / tot}
    return {"radial_profile": E.sum((1, 2)) / tot, "longitudinal_profile": E.sum((2, 3)) / tot}


# ------------------------------------------------------------------ figures
def _label(m):
    return STYLE["truth" if m == "hr" else m]["label"]


def examples(plt, out, data):
    """The Superresolution repo's views: jets with the channels summed (eta x phi); showers summed over angle
    (radius x layer). Every panel of a row shares one log colour scale; the LR measurement is spread uniformly,
    i.e. shown as energy per fine pixel, so merged cells do not look brighter than the truth."""
    from matplotlib.colors import LogNorm
    ex = data["examples"]
    cols = ["hr"] + data["methods"]
    k = len(ex["hr"])
    jet = data["kind"] == "jet"

    def view(E):
        return E.sum(0) if jet else E.sum(1).T   # jet: (U, V); shower (P, U, V) -> (V radius, P layer)
    fig, axes = plt.subplots(k, len(cols), figsize=(2.4 * len(cols) + 0.8, 2.4 * k), squeeze=False)
    for i in range(k):
        vmax = max(float(view(ex["hr"][i]).max()), 1e-6)
        norm = LogNorm(vmax * 1e-4, vmax)
        for j, c in enumerate(cols):
            a = axes[i, j]
            im = a.imshow(np.clip(view(ex[c][i]), vmax * 1e-4, None), norm=norm, cmap="Blues", aspect="auto",
                          origin="lower", interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            if i == 0:
                a.set_title(_label(c), fontsize=8)
            if j == 0 and not jet:
                a.set_ylabel("radius", fontsize=7)
            if i == k - 1 and not jet:
                a.set_xlabel("layer", fontsize=7)
        fig.colorbar(im, ax=list(axes[i]), fraction=0.025, pad=0.01).set_label("energy per pixel", fontsize=7)
    fig.suptitle(("Jet images, channels summed" if jet else "Showers summed over angle (radius x layer)")
                 + "; one event per row, each row on one colour scale; LR shown per fine pixel", x=0.02, ha="left",
                 fontsize=9)
    _save(fig, out, "examples.png")


def channel_panels(plt, out, data):
    """One figure per example event: rows truth / LR / each method, columns tracks / ECAL / HCAL (jets) or the
    early / middle / late layer groups (showers, summed; 45 single layers would not be readable). Each column
    shares one log colour scale, because every channel has its own energy range."""
    from matplotlib.colors import LogNorm
    ex = data["examples"]
    rows = ["hr"] + data["methods"]
    jet = data["kind"] == "jet"
    cols = ([(name.upper() if name != "tracks" else "Tracks", [p]) for p, name in enumerate(data["planes"])] if jet
            else [(name.replace("_", " "), idx) for name, idx in data["groups"].items()])
    for i in range(len(ex["hr"])):
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.0 * len(cols) + 0.6, 2.7 * len(rows)),
                                 squeeze=False, layout="constrained")
        for c, (title, idx) in enumerate(cols):
            vmax = max(float(ex["hr"][i][idx].sum(0).max()), 1e-6)
            norm = LogNorm(vmax * 1e-3, vmax)
            for r, m in enumerate(rows):
                a = axes[r, c]
                im = a.imshow(np.clip(ex[m][i][idx].sum(0), vmax * 1e-3, None), norm=norm, cmap="Blues",
                              origin="lower", interpolation="nearest", aspect="auto")
                a.set_xticks([])
                a.set_yticks([])
                a.grid(False)
                if r == 0:
                    a.set_title(title, fontsize=11, fontweight="bold")
                if c == 0:
                    a.set_ylabel(_label(m), fontsize=8)
            fig.colorbar(im, ax=list(axes[:, c]), fraction=0.04, pad=0.02, location="bottom").set_label(
                f"{title} energy per pixel (log)", fontsize=8)
        fig.suptitle(f"Test event {i}: every method against the truth, per "
                     + ("channel" if jet else "layer group (angle x radius)") + "; LR shown per fine pixel",
                     fontsize=10)
        _save(fig, out, f"channels_event{i}.png")


def observables(plt, out, data, bins=50):
    keys = list(data["truth"]["obs"])
    nc = 4
    nr = -(-len(keys) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.6 * nr), squeeze=False)
    from matplotlib.ticker import MaxNLocator
    for a, key in zip(axes.flat, keys):
        t = data["truth"]["obs"][key]
        # range from truth and every method: a method outside the truth's range must show, not vanish
        lo, hi = _range([t] + [data["m"][m]["obs"][key] for m in data["methods"]], 0.5, 99.5)
        edges = np.linspace(lo, hi, bins + 1)
        a.xaxis.set_major_locator(MaxNLocator(5))
        a.hist(t[np.isfinite(t)], edges, histtype="stepfilled", color=STYLE["truth"]["color"],
               label=STYLE["truth"]["label"])
        for m in data["methods"]:
            v = data["m"][m]["obs"][key]
            s = STYLE[m]
            a.hist(v[np.isfinite(v)], edges, histtype="step", color=s["color"], ls=s["ls"], lw=1.6, label=s["label"])
        a.set_title(key)
        a.set_yticks([])
    for a in axes.flat[len(keys):]:
        a.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out, "observables.png")


def _ratio_limits(ratios):
    v = np.concatenate([r[np.isfinite(r) & (r > 0)] for r in ratios]) if ratios else np.zeros(0)
    if not len(v):
        return 0.5, 1.5, False
    lo, hi = np.percentile(v, [2, 98])
    lo, hi = min(lo * 0.9, 0.8), max(hi * 1.1, 1.25)
    log = hi / lo > 4
    return (max(lo, 1e-2), min(hi, 1e2), True) if log else (max(lo, 0.0), hi, False)


def hist_ratio(plt, out, data, bins=40, min_ref=10):
    """hist/hist_<observable>.png: densities normalised to all events (events outside the range are counted in
    the legend), and the ratio of each method to the truth in bins with >= min_ref truth events. Ratios beyond
    the panel are drawn on its edge, so a large shift stays visible."""
    d = os.path.join(out, "hist")
    os.makedirs(d, exist_ok=True)
    for key, t in data["truth"]["obs"].items():
        t = t[np.isfinite(t)]
        lo, hi = _range([t] + [data["m"][m]["obs"][key] for m in data["methods"]], 0.5, 99.5)
        edges = np.linspace(lo, hi, bins + 1)
        fig, (ax, rx) = plt.subplots(2, 1, figsize=(5.2, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        n_ref, _ = np.histogram(t, edges)
        h_ref = n_ref / max(len(t), 1) / np.diff(edges)
        ax.stairs(h_ref, edges, fill=True, color=STYLE["truth"]["color"], label=STYLE["truth"]["label"])
        ratios = {}
        for m in data["methods"]:
            v = data["m"][m]["obs"][key]
            v = v[np.isfinite(v)]
            n, _ = np.histogram(v, edges)
            outside = 1 - n.sum() / max(len(v), 1)
            h = n / max(len(v), 1) / np.diff(edges)
            st = STYLE[m]
            lab = st["label"] + (f" ({100 * outside:.1f}% outside)" if outside > 0.001 else "")
            ax.stairs(h, edges, color=st["color"], ls=st["ls"], lw=1.5, label=lab)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios[m] = np.where(n_ref >= min_ref, h / h_ref, np.nan)
        r_lo, r_hi, log = _ratio_limits(list(ratios.values()))
        for m, r in ratios.items():
            st = STYLE[m]
            rx.stairs(np.clip(r, r_lo, r_hi), edges, baseline=None, color=st["color"], ls=st["ls"], lw=1.3)
        rx.axhline(1, color=INK2, lw=0.8)
        if log:
            rx.set_yscale("log")
        rx.set_ylim(r_lo, r_hi)
        if all(np.isnan(r).all() for r in ratios.values()):
            rx.text(0.5, 0.5, f"no bin with >= {min_ref} truth events", transform=rx.transAxes, ha="center",
                    va="center", color=INK2, fontsize=8)
        rx.set_ylabel("ratio to truth" + (" (log)" if log else ""), fontsize=8)
        rx.set_xlabel(key)
        ax.set_ylabel("normalised")
        ax.legend(fontsize=7)
        ax.set_title(f"{key}: distribution against the truth", loc="left")
        _save(fig, d, f"hist_{key}.png")


def _paired(t, v):
    """bias, resolution (relative when the truth is positive, else absolute) and Pearson r of paired values."""
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    rel = len(t) > 0 and bool((t > 0).all())
    b, s = _resolution((v - t) / t if rel else v - t)
    r = np.corrcoef(t, v)[0, 1] if len(t) > 2 and np.std(v) > 0 and np.std(t) > 0 else np.nan
    return b, s, r, rel


def scatter_all(plt, out, data, max_points=3000):
    """scatter/scatter_<observable>.png: one panel per method, method against truth for the same events, y = x
    line; title: bias, resolution (relative in %, or absolute when the observable can be <= 0) and Pearson r."""
    d = os.path.join(out, "scatter")
    os.makedirs(d, exist_ok=True)
    methods = data["methods"]
    for key, t in data["truth"]["obs"].items():
        k = min(len(t), max_points)
        lo, hi = _range([t[:k]] + [data["m"][m]["obs"][key][:k] for m in methods], 0.5, 99.5)
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        size, alpha = (10, 0.8) if k < 500 else (4, float(np.clip(800 / k, 0.12, 0.5)))
        fig, axes = plt.subplots(1, len(methods), figsize=(3.4 * len(methods), 3.5), squeeze=False)
        for a, m in zip(axes[0], methods):
            v = data["m"][m]["obs"][key][:k]
            b, s, r, rel = _paired(t[:k], v)
            a.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=":")
            a.scatter(t[:k], v, s=size, alpha=alpha, color=STYLE[m]["color"], linewidths=0, rasterized=True)
            a.set_xlim(lo, hi)
            a.set_ylim(lo, hi)
            a.set_xlabel(f"{key}, truth")
            a.set_ylabel(f"{key}, {_label(m)}", fontsize=7)
            a.set_title(f"bias {100 * b:+.1f}%  res. {100 * s:.1f}%  r = {r:.2f}" if rel else
                        f"bias {b:+.3g}  res. {s:.3g}  r = {r:.2f}", fontsize=8)
        fig.tight_layout()
        _save(fig, d, f"scatter_{key}.png")


def profiles(plt, out, data):
    """profile_<name>.png: mean energy fraction per radial (and longitudinal) bin, every method against the
    truth, with the ratio to the truth below."""
    for name, ax_info in (data.get("profile_axes") or {}).items():
        if name not in data["truth"].get("profiles", {}):
            continue
        x = np.asarray(ax_info["centres"])
        fig, (ax, rx) = plt.subplots(2, 1, figsize=(5.2, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ref = data["truth"]["profiles"][name].mean(0)
        ax.plot(x, ref, "o-", ms=4, color=INK, lw=1.5, label=STYLE["truth"]["label"])
        for m in data["methods"]:
            mean = data["m"][m]["profiles"][name].mean(0)
            st = STYLE[m]
            ax.plot(x, mean, color=st["color"], ls=st["ls"], lw=1.6, label=st["label"])
            with np.errstate(divide="ignore", invalid="ignore"):
                rx.plot(x, np.clip(mean / ref, 0.5, 1.5), color=st["color"], ls=st["ls"], lw=1.3)
        ax.set_yscale("log")
        ax.set_ylabel("mean energy fraction")
        ax.legend(fontsize=7)
        ax.set_title(name.replace("_", " ") + " against the truth", loc="left")
        rx.axhline(1, color=INK2, lw=0.8)
        rx.set_ylim(0.5, 1.5)
        rx.set_ylabel("ratio to truth", fontsize=8)
        rx.set_xlabel(ax_info["label"])
        _save(fig, out, f"profile_{name.split('_')[0]}.png")   # profile_radial.png, profile_longitudinal.png


def w1_heatmap(plt, out, data):
    """w1_heatmap.png: W1 / sigma_truth of every observable (rows) for every method (columns), log colour scale,
    value in each cell; lower is better."""
    from scipy.stats import wasserstein_distance
    keys = list(data["truth"]["obs"])
    methods = data["methods"]
    W = np.full((len(keys), len(methods)), np.nan)
    for i, key in enumerate(keys):
        t = data["truth"]["obs"][key]
        t = t[np.isfinite(t)]
        sd = max(float(np.std(t)), 1e-12)
        for j, m in enumerate(methods):
            v = data["m"][m]["obs"][key]
            v = v[np.isfinite(v)]
            if len(t) and len(v):
                W[i, j] = wasserstein_distance(v, t) / sd
    L = np.log10(np.clip(W, 1e-3, None))
    fig, a = plt.subplots(figsize=(1.5 * len(methods) + 3.0, 0.36 * len(keys) + 1.8))
    im = a.imshow(L, cmap="Blues", aspect="auto", vmin=np.nanmin(L), vmax=np.nanmax(L))
    a.grid(False)
    a.set_xticks(range(len(methods)))
    a.set_xticklabels([_label(m) for m in methods], fontsize=7, rotation=20, ha="right")
    a.set_yticks(range(len(keys)))
    a.set_yticklabels(keys, fontsize=8)
    mid = 0.5 * (np.nanmin(L) + np.nanmax(L))
    for i in range(len(keys)):
        for j in range(len(methods)):
            v = W[i, j]
            a.text(j, i, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if L[i, j] > mid else INK)
    fig.colorbar(im, ax=a, label="log10(W1 / sigma_truth), lower is better")
    a.set_title("Distance of each observable's distribution to the truth", loc="left", fontsize=9)
    _save(fig, out, "w1_heatmap.png")


def scatter_observables(plt, out, data):
    keys = [k for k in KEY_OBSERVABLES[data["kind"]] if k in data["truth"]["obs"]]
    methods = data["methods"]
    fig, axes = plt.subplots(len(keys), len(methods), figsize=(2.9 * len(methods), 2.8 * len(keys)), squeeze=False)
    for i, key in enumerate(keys):
        t = data["truth"]["obs"][key]
        # one range per row from truth and every method, so a biased method is visible, not cut off
        lo, hi = _range([t] + [data["m"][m]["obs"][key] for m in methods], 0.5, 99.5)
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        n = len(t)
        size, alpha = (10, 0.8) if n < 500 else (4, float(np.clip(800 / n, 0.12, 0.5)))
        for j, m in enumerate(methods):
            a = axes[i, j]
            v = data["m"][m]["obs"][key]
            ok = np.isfinite(t) & np.isfinite(v)
            a.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=":")
            a.scatter(t[ok], v[ok], s=size, alpha=alpha, color=STYLE[m]["color"], linewidths=0, rasterized=True)
            r = np.corrcoef(t[ok], v[ok])[0, 1] if ok.sum() > 2 and np.std(v[ok]) > 0 else np.nan
            a.text(0.04, 0.94, f"r = {r:.3f}", transform=a.transAxes, va="top", color=INK, fontsize=8,
                   bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
            a.set_xlim(lo, hi)
            a.set_ylim(lo, hi)
            if i == 0:
                a.set_title(STYLE[m]["label"], fontsize=8)
            if j == 0:
                a.set_ylabel(f"{key}: method")
            a.set_xlabel(f"{key}: truth" if i == len(keys) - 1 else "")
    fig.suptitle("Per event: method against truth (dotted: perfect agreement)", x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, out, "scatter_observables.png")


def energy_residuals(plt, out, data, bins=60):
    panels = [("total", None)] + [(name, idx) for name, idx in data["groups"].items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.9), squeeze=False)
    for a, (name, idx) in zip(axes[0], panels):
        if idx is None:
            et = data["truth"]["E"]
            res = {m: (data["m"][m]["E"] - et) / np.maximum(et, 1e-12) for m in data["methods"]}
        else:
            et = data["truth"]["planes"][:, idx].sum(1)
            ok = et > 0
            res = {m: (data["m"][m]["planes"][ok][:, idx].sum(1) - et[ok]) / et[ok] for m in data["methods"]}
        lo, hi = _range(list(res.values()))
        lo, hi = min(lo, 0.0), max(hi, 0.0)      # keep the perfect-reconstruction line inside
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        edges = np.linspace(lo, hi, bins + 1)
        a.axvline(0, color=MUTED, lw=1)
        for m, r in res.items():
            b, s = _resolution(r)
            st = STYLE[m]
            a.hist(np.clip(r[np.isfinite(r)], lo, hi), edges, histtype="step", color=st["color"], ls=st["ls"], lw=1.6,
                   label=f"{st['label']}: bias {b:+.3f}, res. {s:.3f}")
        a.set_title(f"event energy, {name}")
        a.set_xlabel("(E - E_true) / E_true")
        a.set_yticks([])
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.5)
    fig.suptitle("Event-energy residual per method (legend: median bias and IQR/1.349 resolution)", x=0.02,
                 ha="left")
    fig.tight_layout()
    _save(fig, out, "energy_residuals.png")


def _history(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows]) for k in rows[0]} \
        if rows else None


def training_curves(plt, out, experiment):
    found = [(s, t, _history(os.path.join(experiment, s, "history.csv"))) for s, t in STAGES
             if os.path.exists(os.path.join(experiment, s, "history.csv"))]
    found = [x for x in found if x[2] is not None]
    if not found:
        return
    panels = list(found)
    pre = next((h for s, _, h in found if s == "pretrain"), None)
    if pre is not None and "val_feat_std" in pre:
        panels.append(("pretrain_std", "pretraining: feature spread (collapse check)", pre))
    nc = 3
    nr = -(-len(panels) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc, 2.8 * nr), squeeze=False)
    from matplotlib.ticker import MaxNLocator
    for a, (s, title, h) in zip(axes.flat, panels):
        ep = h["epoch"]
        mk = {"marker": "o", "ms": 4} if len(ep) <= 12 else {}   # a 1-2 epoch run is otherwise a lone dot
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(ep) == 1:
            a.set_xlim(ep[0] - 1, ep[0] + 1)
        if s == "pretrain_std":
            a.plot(ep, h["val_feat_std"], color="#2a78d6", label="val feature std", **mk)
            a.axhline(0, color=MUTED, lw=1)
            a.set_ylim(bottom=0)
        else:
            a.plot(ep, h["train_loss"], color=INK2, label="train", **mk)
            a.plot(ep, h["val_loss"], color="#2a78d6", label="validation", **mk)
            ref = {"sr": ("val_identity", "no SR (identity)"), "decoder": ("val_autoencode", "decoding E(HR)")}.get(s)
            if ref and ref[0] in h:
                a.plot(ep, h[ref[0]], color=MUTED, ls="--", lw=1.5, label=ref[1])
            if s == "sr" and "val_teacher_forcing" in h:
                a.plot(ep, h["val_teacher_forcing"], color="#2a78d6", ls=":", lw=1.5, label="val teacher forcing")
        if s.startswith("pretrain") and "train_pool" in h and (h["train_pool"] == 1).any():
            c0 = ep[np.argmax(h["train_pool"] == 1)] - 0.5
            a.axvspan(c0, ep[-1], color=GRID, alpha=0.6, lw=0)
            a.text(c0, 0.02, " full resolution", transform=a.get_xaxis_transform(), va="bottom", color=INK2,
                   fontsize=7)
        best = np.nanargmin(h["val_loss"]) if s != "pretrain_std" else None
        if best is not None and s not in ("pretrain",):
            a.plot(ep[best], h["val_loss"][best], "o", ms=6, color="#2a78d6", mec="white", mew=1.5)
        a.set_title(title + (" (val. comparable from the cooldown on)" if s == "pretrain" else ""), fontsize=9)
        a.set_xlabel("epoch")
        a.legend(fontsize=7, frameon=True, framealpha=0.9, facecolor="white", edgecolor="none")
    for a in axes.flat[len(panels):]:
        a.axis("off")
    fig.suptitle("Training curves (dot: best validation epoch, the checkpoint used downstream)", x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, out, "training_curves.png")


def probe_roc(plt, out, data):
    from sklearn.metrics import roc_auc_score, roc_curve
    y = data["truth"]["target"]
    if "probe" not in data["truth"] or len(np.unique(y)) != 2:
        return
    fig, a = plt.subplots(figsize=(4.6, 4.2))
    a.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    curves = [("truth", data["truth"]["probe"])] + [(m, data["m"][m].get("probe")) for m in data["methods"]]
    for m, p in curves:
        if p is None or not len(p):
            continue
        fpr, tpr, _ = roc_curve(y, p)
        st = STYLE[m]
        a.plot(fpr, tpr, color=INK if m == "truth" else st["color"], ls=st.get("ls", "-"),
               label=f"{st['label']}: AUC {roc_auc_score(y, p):.3f}")
    a.set_xlabel("gluon efficiency (false positive rate)")
    a.set_ylabel("quark efficiency (true positive rate)")
    a.set_title("Quark/gluon probe on each method's latent", loc="left")
    a.legend(loc="lower right", fontsize=7)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    _save(fig, out, "probe_roc.png")


def energy_vs_einc(plt, out, data, n_bins=10):
    from matplotlib.ticker import FuncFormatter
    e_inc = data["truth"]["target"] / 1000.0   # MeV -> GeV
    et = data["truth"]["E"]
    # equal-count bins (CaloChallenge samples E_inc log-uniformly, so these are close to log-spaced): no bin is
    # empty, whatever the number of test events; each point sits at its bin's median E_inc
    n_bins = max(1, min(n_bins, len(e_inc) // 20))
    edges = np.quantile(e_inc, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, e_inc, side="right") - 1, 0, n_bins - 1)
    centers = np.array([np.median(e_inc[idx == b]) for b in range(n_bins)])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for m in data["methods"]:
        r = (data["m"][m]["E"] - et) / np.maximum(et, 1e-12)
        stats = np.array([_resolution(r[idx == b]) for b in range(n_bins)])
        st = STYLE[m]
        for a, col in zip(axes, (0, 1)):
            a.plot(centers, stats[:, col], color=st["color"], ls=st["ls"], marker="o", ms=5, mec="white", mew=1,
                   label=st["label"])
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[0].set_title("bias: median of (E - E_true) / E_true", loc="left")
    axes[1].set_title("resolution: IQR / 1.349", loc="left")
    axes[1].set_ylim(bottom=0)
    for a in axes:
        a.set_xscale("log")
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        a.set_xlabel("incident energy E_inc [GeV]")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"Deposited-energy reconstruction against the incident energy ({n_bins} equal-count "
                 f"bin{'s' if n_bins > 1 else ''})",
                 x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, out, "energy_vs_einc.png")


def make_all(out, data, experiment, bins=50):
    """Every figure; a failing figure is reported and skipped, never fails the evaluation."""
    plt = _plt()
    jobs = [("examples", lambda: examples(plt, out, data)), ("channel panels", lambda: channel_panels(plt, out, data)),
            ("observables", lambda: observables(plt, out, data, bins)),
            ("histograms with ratio", lambda: hist_ratio(plt, out, data)),
            ("scatter per observable", lambda: scatter_all(plt, out, data)),
            ("scatter", lambda: scatter_observables(plt, out, data)),
            ("profiles", lambda: profiles(plt, out, data)), ("W1 heatmap", lambda: w1_heatmap(plt, out, data)),
            ("energy residuals", lambda: energy_residuals(plt, out, data)),
            ("training curves", lambda: training_curves(plt, out, experiment))]
    if data["kind"] == "jet":
        jobs.append(("probe ROC", lambda: probe_roc(plt, out, data)))
    else:
        jobs.append(("energy vs E_inc", lambda: energy_vs_einc(plt, out, data)))
    for name, job in jobs:
        try:
            job()
        except Exception as e:
            print(f"(figure '{name}' skipped: {type(e).__name__}: {e})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("experiment", help="experiment folder (contains evaluation/per_event.npz)")
    a = ap.parse_args(argv)
    out = os.path.join(a.experiment, "evaluation")
    make_all(out, load_per_event(out), a.experiment)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
