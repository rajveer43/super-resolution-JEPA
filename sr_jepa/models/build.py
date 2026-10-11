"""Networks from config + data layout."""
from .vit import AttentiveProbe, Decoder, Encoder, LatentSRPredictor, MaskPredictor


def _common(cfg, layout):
    # only the angular axis (H) of showers is periodic
    return {"periodic": (False, layout.circular_u, False), "rope_base": float(cfg["model"]["rope_base"])}


def build_encoder(cfg, layout):
    e = cfg["encoder"]
    return Encoder(layout.C, layout.patch, e["dim"], e["depth"], e["heads"], e["mlp_ratio"], **_common(cfg, layout))


def build_mask_predictor(cfg, layout):
    p = cfg["predictor"]
    return MaskPredictor(cfg["encoder"]["dim"], p["dim"], p["depth"], p["heads"], p["mlp_ratio"],
                         **_common(cfg, layout))


def build_sr_predictor(cfg, layout):
    s = cfg["sr"]
    return LatentSRPredictor(cfg["encoder"]["dim"], s["dim"], s["depth"], s["heads"], s["mlp_ratio"],
                             n_levels=len(layout.levels), cond_dim=layout.cond_dim, residual=s["residual"],
                             **_common(cfg, layout))


def build_decoder(cfg, layout):
    d = cfg["decoder"]
    return Decoder(cfg["encoder"]["dim"], d["dim"], d["depth"], d["heads"], layout.patch, layout.C, d["mlp_ratio"],
                   lr_skip=d["lr_skip"], **_common(cfg, layout))


def build_probe(cfg, n_out):
    p = cfg["probe"]
    return AttentiveProbe(cfg["encoder"]["dim"], n_out, p["heads"], p["depth"])
