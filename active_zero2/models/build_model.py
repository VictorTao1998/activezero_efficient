from active_zero2.models.cfnet.build_model import build_model as build_cfnet
from active_zero2.models.psmnet.build_model import build_model as build_psmnet
from active_zero2.models.psmnet_range.build_model import build_model as build_psmnetrange


def build_model(cfg):
    if cfg.MODEL_TYPE == "PSMNet":
        model = build_psmnet(cfg)
    elif cfg.MODEL_TYPE == "CFNet":
        model = build_cfnet(cfg)
    elif cfg.MODEL_TYPE == "PSMNetRange":
        model = build_psmnetrange(cfg)
    else:
        raise ValueError(f"Unexpected model type: {cfg.MODEL_TYPE}")

    return model
