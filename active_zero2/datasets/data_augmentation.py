import random

import torch
import torchvision.transforms as Transforms


def data_augmentation(data_aug_cfg=None):
    """ """
    transform_list = [Transforms.ToTensor()]
    if data_aug_cfg:
        if data_aug_cfg.GAUSSIAN_BLUR:
            gaussian_sig = random.uniform(data_aug_cfg.GAUSSIAN_MIN, data_aug_cfg.GAUSSIAN_MAX)
            transform_list += [Transforms.GaussianBlur(kernel_size=data_aug_cfg.GAUSSIAN_KERNEL, sigma=gaussian_sig)]
        if data_aug_cfg.COLOR_JITTER:
            transform_list += [
                Transforms.ColorJitter(
                    brightness=[data_aug_cfg.BRIGHT_MIN, data_aug_cfg.BRIGHT_MAX],
                    contrast=[data_aug_cfg.CONTRAST_MIN, data_aug_cfg.CONTRAST_MAX],
                    saturation=[data_aug_cfg.SATURATION_MIN, data_aug_cfg.SATURATION_MAX],
                    hue=[data_aug_cfg.HUE_MIN, data_aug_cfg.HUE_MAX],
                )
            ]
    # Normalization
    transform_list += [
        Transforms.Normalize(
            mean=[0.45],
            std=[0.224],
        )
    ]
    custom_augmentation = Transforms.Compose(transform_list)
    return custom_augmentation
