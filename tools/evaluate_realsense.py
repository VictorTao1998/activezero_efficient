import numpy as np
import torch
from path import Path
import argparse
import cv2
import os
from active_zero2.datasets.messytable import MessyTableDataset
from active_zero2.utils.io import load_pickle
from data_rendering.utils.render_utils import visualize_depth
from active_zero2.utils.metrics import ErrorMetric
from active_zero2.config import cfg
from tqdm import tqdm
import torch
import copy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate real realsense performance")
    parser.add_argument("-d", "--data-folder", type=str, required=True)
    parser.add_argument(
        "-s",
        "--split-file",
        type=str,
        metavar="FILE",
        required=True,
    )

    args = parser.parse_args()
    return args


def register_depth(view_folder):
    view_folder = Path(view_folder)
    if (view_folder / "1024_depthL_real.png").exists():
        depth_u16 = cv2.imread(view_folder / "1024_depthL_real.png", cv2.IMREAD_UNCHANGED)
        assert depth_u16.shape == (540, 960)
        depth = (depth_u16.astype(float)) / 1000.0
    else:
        img_meta = load_pickle(view_folder / "meta.pkl")
        extrinsic_l = img_meta["extrinsic_l"]
        extrinsic_r = img_meta["extrinsic_r"]
        intrinsic_l = img_meta["intrinsic_l"]
        # intrinsic_l[:2] /= 2
        intrinsic_l[2] = np.array([0.0, 0.0, 1.0])

        rgb_cam_depth = cv2.imread(view_folder / "1024_depth_real.png", cv2.IMREAD_UNCHANGED)
        rgb_cam_depth = rgb_cam_depth.astype(float) / 1000.0
        w, h = 1920, 1080
        rt_mainl = img_meta["extrinsic"] @ np.linalg.inv(img_meta["extrinsic_l"])
        rt_lmain = np.linalg.inv(rt_mainl)
        depth = cv2.rgbd.registerDepth(
            img_meta["intrinsic"], intrinsic_l, None, rt_lmain, rgb_cam_depth, (w, h), depthDilation=True
        )
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0
        depth[depth < 0] = 0

        depth = cv2.resize(depth, (960, 540), interpolation=cv2.INTER_NEAREST)
        depth_u16 = copy.deepcopy(depth)
        depth_u16 = (depth_u16 * 1000.0).astype(np.uint16)

        cv2.imwrite(view_folder / "1024_depthL_real.png", depth_u16)
        vis_depth = visualize_depth(depth_u16)
        cv2.imwrite(view_folder / "1024_depthL_real_colored.png", vis_depth)

    return depth


def main():
    args = parse_args()

    # use default test config
    cfg.TEST.IS_DEPTH = True
    # Build metrics
    metric = ErrorMetric(
        model_type="RealSense",
        use_mask=cfg.TEST.USE_MASK,
        max_disp=cfg.TEST.MAX_DISP,
        depth_range=cfg.TEST.DEPTH_RANGE,
        num_classes=cfg.DATA.NUM_CLASSES,
        is_depth=cfg.TEST.IS_DEPTH,
    )
    metric.reset()

    dataset = MessyTableDataset(
        mode="test",
        domain="real",
        root_dir=args.data_folder,
        split_file=args.split_file,
        height=544,
        width=960,
        meta_name="meta.pkl",
        depth_name="depthL.png",
        normal_name="normalL.png",
        left_name="1024_irL_real.png",
        right_name="1024_irR_real.png",
        left_pattern_name="",
        right_pattern_name="",
        label_name="irL_label_image.png",
    )

    for data in tqdm(dataset):
        view_folder = data["dir"]
        data = {k: v.unsqueeze(0) for k, v in data.items() if isinstance(v, torch.Tensor)}
        data["dir"] = view_folder
        view_folder = os.path.join(args.data_folder, view_folder)
        print(view_folder)
        depth = register_depth(view_folder)
        pred_dict = {"depth": depth}

        metric.compute(data, pred_dict, save_folder=view_folder, real_data=True)

    print(metric.summary())


if __name__ == "__main__":
    main()
