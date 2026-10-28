"""Evaluation figures: per-event data survives the npz round trip and every figure is drawn for both detectors."""
import csv
import os

import numpy as np
import pytest

from sr_jepa.eval import plots


def _fake(kind, n=300, seed=0):
    rng = np.random.default_rng(seed)
    methods = ["lr_up", "jepa", "e2e_jepa", "e2e_scratch"]
    P = 3 if kind == "jet" else 6
    obs_keys = plots.KEY_OBSERVABLES[kind] + ["C2"]
    target = rng.integers(0, 2, n).astype(float) if kind == "jet" else np.exp(rng.uniform(7, 13, n))
    planes = rng.gamma(2.0, 10.0, (n, P))

    axes = ({"radial_profile": {"centres": list(np.linspace(0.025, 0.775, 16)), "label": "dR"}} if kind == "jet" else
            {"radial_profile": {"centres": list(range(9)), "label": "r"},
             "longitudinal_profile": {"centres": list(range(P)), "label": "layer"}})

    def block(scale):
        return {"E": planes.sum(1) * scale, "planes": planes * scale, "probe": target + rng.normal(0, 0.5, n),
                "obs": {k: rng.normal(1.0, 0.2, n) * scale for k in obs_keys},
                "profiles": {k: rng.dirichlet(np.ones(len(a["centres"])), n) * scale for k, a in axes.items()}}
    truth = {**block(1.0), "target": target}
    groups = {"a": [0], "b": [1, 2]} if kind == "jet" else {"early": [0, 1], "late": [2, 3, 4, 5]}
    examples = {k: rng.random((2, P, 8, 6)) for k in methods + ["hr"]}
    return {"kind": kind, "planes": [f"p{i}" for i in range(P)], "groups": groups, "methods": methods,
            "profile_axes": axes,
            "truth": truth, "m": {m: block(1.0 + 0.05 * i) for i, m in enumerate(methods)}, "examples": examples}


@pytest.mark.parametrize("kind", ["jet", "shower"])
def test_per_event_roundtrip_and_every_figure(tmp_path, kind):
    out = tmp_path / "evaluation"
    out.mkdir()
    data = _fake(kind)
    plots.save_per_event(str(out), data)
    back = plots.load_per_event(str(out))
    assert back["methods"] == data["methods"] and back["groups"] == data["groups"]
    np.testing.assert_allclose(back["m"]["jepa"]["obs"]["C2"], data["m"]["jepa"]["obs"]["C2"])
    np.testing.assert_allclose(back["truth"]["target"], data["truth"]["target"])
    np.testing.assert_allclose(back["m"]["jepa"]["profiles"]["radial_profile"],
                               data["m"]["jepa"]["profiles"]["radial_profile"])
    stage = tmp_path / "sr"
    stage.mkdir()
    with open(stage / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_identity"])
        for e in range(1, 6):
            w.writerow([e, 1 / e, 1.2 / e, 1.5])
    plots.make_all(str(out), back, str(tmp_path))
    expected = ["examples", "channels_event0", "channels_event1", "observables", "scatter_observables",
                "energy_residuals", "training_curves", "w1_heatmap", "profile_radial",
                "hist/hist_C2", "scatter/scatter_C2"] + (["probe_roc"] if kind == "jet" else
                                                         ["energy_vs_einc", "profile_longitudinal"])
    for name in expected:
        assert os.path.getsize(out / f"{name}.png") > 0, name
