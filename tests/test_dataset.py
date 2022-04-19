#!/usr/bin/env python
import os
import os.path as osp
import sys

import torch

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)

import matplotlib.pyplot as plt

from active_zero2.config import cfg
from active_zero2.datasets.build_dataset import build_dataset
from active_zero2.utils.reprojection import apply_disparity


def main():
    config_file = osp.join(osp.dirname(__file__), "../configs/example.yml")
    cfg.merge_from_file(config_file)
    cfg.freeze()

    train_sim_dataset = build_dataset(cfg, mode="train", domain="sim")
    train_real_dataset = build_dataset(cfg, mode="train", domain="real")
    val_sim_dataset = build_dataset(cfg, mode="val", domain="sim")
    val_real_dataset = build_dataset(cfg, mode="val", domain="real")
    test_sim_dataset = build_dataset(cfg, mode="test", domain="sim")
    test_real_dataset = build_dataset(cfg, mode="test", domain="real")

    data = train_sim_dataset[0]
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)

    # check reprojection
    img_l = data["img_l"].unsqueeze(0).cuda()
    img_r = data["img_r"].unsqueeze(0).cuda()
    img_disp_l = data["img_disp_l"].unsqueeze(0).cuda()

    img_pattern_l = data["img_pattern_l"].unsqueeze(0).cuda()
    img_pattern_r = data["img_pattern_r"].unsqueeze(0).cuda()

    img_l_reprojed = apply_disparity(img_r, -img_disp_l)
    img_pattern_l_reprojed = apply_disparity(img_pattern_r, -img_disp_l)

    img_l = img_l[0, 0].cpu().numpy()
    img_r = img_r[0, 0].cpu().numpy()
    img_l_reprojed = img_l_reprojed[0, 0].cpu().numpy()
    img_pattern_l = img_pattern_l[0, 0].cpu().numpy()
    img_pattern_r = img_pattern_r[0, 0].cpu().numpy()
    img_pattern_l_reprojed = img_pattern_l_reprojed[0, 0].cpu().numpy()

    plt.figure(
        f"Check reprojection",
        figsize=(24, 16),
    )
    plt.subplot(2, 4, 1)
    plt.gca().set_title("img_l")
    plt.imshow(img_l)
    plt.subplot(2, 4, 2)
    plt.gca().set_title("img_r")
    plt.imshow(img_r)
    plt.subplot(2, 4, 3)
    plt.gca().set_title("img_l_reproj")
    plt.imshow(img_l_reprojed)
    plt.subplot(2, 4, 4)
    plt.gca().set_title("img_l_reproj_err")
    plt.imshow(img_l - img_l_reprojed)
    # plt.colorbar()
    plt.subplot(2, 4, 5)
    plt.gca().set_title("img_pattern_l")
    plt.imshow(img_pattern_l)
    plt.subplot(2, 4, 6)
    plt.gca().set_title("img_pattern_r")
    plt.imshow(img_pattern_r)
    plt.subplot(2, 4, 7)
    plt.gca().set_title("img_pattern_l_reproj")
    plt.imshow(img_pattern_l_reprojed)
    plt.subplot(2, 4, 8)
    plt.gca().set_title("img_pattern_l_reproj_err")
    plt.imshow(img_pattern_l - img_pattern_l_reprojed)
    # plt.colorbar()

    plt.show()


if __name__ == "__main__":
    main()
