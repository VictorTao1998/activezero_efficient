import argparse
import os
import os.path as osp
import sys

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)

import time

import cv2
import numpy as np

from active_zero2.utils.geometry import cal_normal_map
from active_zero2.utils.io import load_pickle


def main():
    parser = argparse.ArgumentParser(description="Extract temporal IR pattern from temporal real images")
    parser.add_argument(
        "-s",
        "--split-file",
        type=str,
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data-folder",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    with open(args.split_file, "r") as f:
        prefix = [line.strip() for line in f]
    num = len(prefix)

    start = time.time()
    for idx in range(num):
        p = prefix[idx]
        depth = cv2.imread(os.path.join(args.data_folder, p, "depthL.png"), cv2.IMREAD_UNCHANGED)
        depth = (depth.astype(float)) / 1000
        meta = load_pickle(os.path.join(args.data_folder, p, "meta.pkl"))
        intrinsic_l = meta["intrinsic_l"]
        normal_map = cal_normal_map(depth, intrinsic_l)
        normal_map_colored = ((normal_map + 1) * 127.5).astype(np.uint8)
        cv2.imwrite(os.path.join(args.data_folder, p, "normalL_colored.png"), normal_map_colored)
        normal_map = ((normal_map + 1) * 1000).astype(np.uint16)
        cv2.imwrite(os.path.join(args.data_folder, p, "normalL.png"), normal_map)
        print(f"Generating {p} normal map {idx}/{num} time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
