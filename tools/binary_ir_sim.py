import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import time

parser = argparse.ArgumentParser(description="Extract LCN IR pattern from IR images")
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
parser.add_argument("-p", "--patch", type=int, required=True)
args = parser.parse_args()


def get_smoothed_ir_pattern2(img_ir: np.array, img: np.array, ks=11, threshold=0.005):
    h, w = img_ir.shape
    hs = int(h // ks)
    ws = int(w // ks)
    diff = np.abs(img_ir - img)
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    diff_avg = cv2.resize(diff, (ws, hs), interpolation=cv2.INTER_AREA)
    diff_avg = cv2.resize(diff_avg, (w, h), interpolation=cv2.INTER_AREA)
    ir = np.zeros_like(diff)
    diff2 = diff - diff_avg
    ir[diff2 > threshold] = 1
    ir = (ir * 255).astype(np.uint8)
    return ir


def main():
    with open(args.split_file, "r") as f:
        prefix = [line.strip() for line in f]
    num = len(prefix)

    start = time.time()
    for idx in range(num):
        for direction in ["irL", "irR"]:
            p = prefix[idx]
            f0 = os.path.join(args.data_folder, p, f"0128_{direction}_kuafu_half.png")
            f6 = os.path.join(args.data_folder, p, f"0128_{direction}_kuafu_half_no_ir.png")
            img_0 = np.array(Image.open(f0).convert(mode="L"))
            img_6 = np.array(Image.open(f6).convert(mode="L"))

            print(f"Generating {p} binary sim {direction} pattern {idx}/{num} time: {time.time() - start:.2f}s")
            binary_pattern = get_smoothed_ir_pattern2(img_6, img_0, ks=args.patch)
            cv2.imwrite(os.path.join(args.data_folder, p, f"0128_{direction}_bin_ps{args.patch}.png"), binary_pattern)


if __name__ == "__main__":
    main()
