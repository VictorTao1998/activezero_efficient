import os
import random

import cv2
import numpy as np
import torch
from loguru import logger
from path import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from active_zero2.utils.io import load_pickle


class MessyTableDataset(Dataset):
    def __init__(
        self,
        mode: str,
        domain: str,
        root_dir: str,
        split_file: str,
        height: int,
        width: int,
        meta_name: str,
        depth_name: str,
        normal_name: str,
        left_name: str,
        right_name: str,
        left_pattern_name: str = "",
        right_pattern_name: str = "",
    ):
        self.mode = mode

        assert domain in ["sim", "real"], f"Unknown dataset mode: [{domain}]"
        self.domain = domain
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            logger.error(f"Not exists root dir: {self.root_dir}")

        self.split_file = split_file
        self.height, self.width = height, width
        self.meta_name = meta_name
        self.depth_name = depth_name
        self.normal_name = normal_name
        self.left_name, self.right_name = left_name, right_name
        self.left_pattern_name, self.right_pattern_name = (
            left_pattern_name,
            right_pattern_name,
        )

        self.img_dirs = self._gen_path_list()

        logger.info(
            f"MessyTableDataset: domain: {domain}, root_dir: {root_dir}, length: {len(self.img_dirs)},"
            f" left_name: {left_name}, right_name: {right_name},"
            f" left_pattern_name: {left_pattern_name}, right_pattern_name: {right_pattern_name}"
        )

    def _gen_path_list(self):
        img_dirs = []
        if not self.split_file:
            logger.warning(f"Split_file is not defined. The dataset is None.")
            return img_dirs
        with open(self.split_file, "r") as f_split_file:
            for l in f_split_file.readlines():
                img_dirs.append(self.root_dir / l.strip())

        check = False
        if check:
            print("Checking img dirs...")
            for d in tqdm(img_dirs):
                if not d.exists():
                    logger.error(f"{d} not exists.")

        return img_dirs

    def __getitem__(self, index):
        data_dict = {}
        img_dir = self.img_dirs[index]

        img_l = np.array(Image.open(img_dir / self.left_name).convert(mode="L")) / 255  # [H, W]
        img_r = np.array(Image.open(img_dir / self.right_name).convert(mode="L")) / 255

        origin_h, origin_w = img_l.shape[:2]  # (960, 540)

        if self.left_pattern_name and self.right_pattern_name:
            img_pattern_l = np.array(Image.open(img_dir / self.left_pattern_name).convert(mode="L")) / 255  # [H, W]
            img_pattern_r = np.array(Image.open(img_dir / self.right_pattern_name).convert(mode="L")) / 255

        if self.depth_name and self.meta_name:
            img_depth_l = np.array(Image.open(img_dir / self.depth_name)) / 1000  # convert from mm to m
            img_depth_l = cv2.resize(img_depth_l, (origin_w, origin_h), cv2.INTER_NEAREST)

            img_meta = load_pickle(img_dir / self.meta_name)
            extrinsic_l = img_meta["extrinsic_l"]
            extrinsic_r = img_meta["extrinsic_r"]
            intrinsic_l = img_meta["intrinsic_l"]
            intrinsic_l[:2] /= 2
            baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
            focal_length = intrinsic_l[0, 0]

            mask = img_depth_l > 0
            img_disp_l = np.zeros_like(img_depth_l)
            img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]

        if self.normal_name:
            img_normal_l = np.array(Image.open(img_dir / self.normal_name))
            img_normal_l = cv2.resize(img_normal_l, (origin_w, origin_h), cv2.INTER_NEAREST)

        # random crop
        if self.mode == "test":
            x = 0
            y = 0
            assert self.height == origin_h and self.width == origin_w, f"Test mode should use the whole image."

            def crop(img):
                return img

        else:
            x = np.random.randint(0, origin_w - self.width)
            y = np.random.randint(0, origin_h - self.height)

            def crop(img):
                return img[y : y + self.height, x : x + self.width]

        img_l = crop(img_l)
        img_r = crop(img_r)
        if self.depth_name and self.meta_name:
            intrinsic_l[0, 2] -= x
            intrinsic_l[1, 2] -= y
            img_depth_l = crop(img_depth_l)
            img_disp_l = crop(img_disp_l)

        if self.left_pattern_name and self.right_pattern_name:
            img_pattern_l = crop(img_pattern_l)
            img_pattern_r = crop(img_pattern_r)

        if self.normal_name:
            img_normal_l = crop(img_normal_l)

        data_dict["img_l"] = torch.from_numpy(img_l).float().unsqueeze(0)
        data_dict["img_r"] = torch.from_numpy(img_r).float().unsqueeze(0)
        if self.depth_name and self.meta_name:
            data_dict["img_deth_l"] = torch.from_numpy(img_depth_l).float().unsqueeze(0)
            data_dict["img_disp_l"] = torch.from_numpy(img_disp_l).float().unsqueeze(0)
            data_dict["intrinsic_l"] = torch.from_numpy(intrinsic_l).float()

        if self.left_pattern_name and self.right_pattern_name:
            data_dict["img_pattern_l"] = torch.from_numpy(img_pattern_l).float().unsqueeze(0)
            data_dict["img_pattern_r"] = torch.from_numpy(img_pattern_r).float().unsqueeze(0)
        if self.normal_name:
            data_dict["img_normal_l"] = torch.from_numpy(img_normal_l).float().permute(2, 0, 1)

        return data_dict

    def __len__(self):
        return len(self.img_dirs)
