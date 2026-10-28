"""Compare finished experiments (e.g. one per encoder size): one markdown table from their evaluation/metrics.json.

    python -m sr_jepa.eval.compare <experiment folder> [<experiment folder> ...] [--out comparison.md]

Rows: experiment x method (lr_up baseline, jepa, e2e_jepa, e2e_scratch); columns: encoder size and the headline
metrics of the evaluation summary. Experiments without metrics.json are listed as missing.
"""
import argparse
import json
import os

import numpy as np

COLUMNS = [("energy bias", lambda r: r["energy"]["total"]["bias"]),
           ("energy res.", lambda r: r["energy"]["total"]["resolution"]),
           ("pixel MAE", lambda r: r["pixels"]["mae_log"]),
           ("hit F1", lambda r: r["hits"]["f1"]),
           ("mean W1/σ", lambda r: float(np.mean([v["w1_over_sigma"] for v in r["observables"].values()]))),
           ("latent L1 (pred.)", lambda r: r["latent"]["predicted_l1"])]


def _cell(get, r):
    try:
        return f"{get(r):.4g}"
    except (KeyError, TypeError, ValueError):
        return "-"


def table(folders):
    lines = ["| experiment | encoder | method | " + " | ".join(c for c, _ in COLUMNS) + " | probe |",
             "|---|---|---|" + "---|" * (len(COLUMNS) + 1)]
    missing = []
    for d in folders:
        path = os.path.join(d, "evaluation", "metrics.json")
        if not os.path.exists(path):
            missing.append(d)
            continue
        with open(path) as f:
            m = json.load(f)
        enc = m.get("hyperparameters", {}).get("encoder", {})
        size = f"{enc.get('dim', '?')} x {enc.get('depth', '?')}"
        probe_key = next(iter(m.get("probe_on_truth", {})), None)
        for method, r in m["methods"].items():
            probe = _cell(lambda x: x["probe"][probe_key], r) if probe_key else "-"
            lines.append(f"| {os.path.basename(os.path.normpath(d))} | {size} | {method} | "
                         + " | ".join(_cell(g, r) for _, g in COLUMNS) + f" | {probe} |")
    out = ["# Encoder-size comparison", "", f"{len(folders) - len(missing)} of {len(folders)} experiments evaluated "
           "(test split; lower is better except hit F1 and probe AUC).", ""] + lines
    if missing:
        out += ["", "Not evaluated (no evaluation/metrics.json): " + ", ".join(f"`{d}`" for d in missing)]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    md = table(a.experiments)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(md)
        print(f"written to {a.out}\n")
    print(md)


if __name__ == "__main__":
    main()
