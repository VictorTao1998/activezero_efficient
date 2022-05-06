from active_zero2.models.cfnet.build_model import build_model as build_cfnet
from active_zero2.models.psmnet.build_model import build_model as build_psmnet
from active_zero2.models.psmnet_dilation.build_model import build_model as build_psmnetdilation
from active_zero2.models.psmnet_grad.build_model import build_model as build_psmnetgrad
from active_zero2.models.psmnet_kpac.build_model import build_model as build_psmnetkpac
from active_zero2.models.psmnet_range.build_model import build_model as build_psmnetrange
from active_zero2.models.psmnet_range_4.build_model import build_model as build_psmnetrange4
from active_zero2.models.smdnet.build_model import build_model as build_smdnet


def build_model(cfg):
    if cfg.MODEL_TYPE == "PSMNet":
        model = build_psmnet(cfg)
    elif cfg.MODEL_TYPE == "CFNet":
        model = build_cfnet(cfg)
    elif cfg.MODEL_TYPE == "PSMNetRange":
        model = build_psmnetrange(cfg)
    elif cfg.MODEL_TYPE == "PSMNetRange4":
        model = build_psmnetrange4(cfg)
    elif cfg.MODEL_TYPE == "PSMNetDilation":
        model = build_psmnetdilation(cfg)
    elif cfg.MODEL_TYPE == "SMDNet":
        model = build_smdnet(cfg)
    elif cfg.MODEL_TYPE == "PSMNetDilation":
        model = build_psmnetdilation(cfg)
    elif cfg.MODEL_TYPE == "PSMNetKPAC":
        model = build_psmnetkpac(cfg)
    elif cfg.MODEL_TYPE == "PSMNetGrad":
        model = build_psmnetgrad(cfg)
    else:
        raise ValueError(f"Unexpected model type: {cfg.MODEL_TYPE}")

    return model
