"""Build optimizers and schedulers"""
import warnings

import torch


def build_optimizer(cfg, model):
    name = cfg.OPTIMIZER.TYPE
    if name == "":
        warnings.warn("No optimizer is built.")
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.OPTIMIZER.LR,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
            **cfg.OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}.")


def build_g_optimizer(cfg, model):
    name = cfg.G_OPTIMIZER.TYPE
    if name == "":
        warnings.warn("No G optimizer is built.")
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.psmnet.parameters(),
            lr=cfg.G_OPTIMIZER.LR,
            weight_decay=cfg.G_OPTIMIZER.WEIGHT_DECAY,
            **cfg.G_OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}.")


def build_d_optimizer(cfg, model):
    name = cfg.D_OPTIMIZER.TYPE
    if name == "":
        warnings.warn("No D optimizer is built.")
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.D.parameters(),
            lr=cfg.D_OPTIMIZER.LR,
            **cfg.D_OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}.")


def build_lr_scheduler(cfg, optimizer):
    name = cfg.LR_SCHEDULER.TYPE
    if name == "":
        warnings.warn("No lr_scheduler is built.")
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        lr_scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.LR_SCHEDULER.get(name, dict()),
        )
        return lr_scheduler
    else:
        raise ValueError(f"Unsupported lr_scheduler: {name}.")
