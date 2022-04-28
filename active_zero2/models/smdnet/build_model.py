from active_zero2.models.smdnet.SMDHead import SMDHead


def build_model(cfg):
    model = SMDHead(cfg)
    return model
