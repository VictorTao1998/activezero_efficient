from active_zero2.models.festereo.default import DefaultModel

def build_model(cfg):
    model = DefaultModel(
        maxdisp=cfg.PSMNet.MAX_DISP
    )
    return model