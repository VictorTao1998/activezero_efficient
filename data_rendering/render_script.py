import os
import os.path as osp
import time
import sys

import numpy as np
from path import Path
from loguru import logger

CUR_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(osp.join(osp.dirname(__file__), ".."))

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--sub", type=int, required=True)
parser.add_argument("--total", type=int, required=True)
parser.add_argument("--target-root", type=str, required=True)
parser.add_argument("--rand-pattern", action="store_true")
parser.add_argument("--fixed-angle", action="store_true")
parser.add_argument("--primitives", action="store_true", help="use primitives")
parser.add_argument("--primitives-v2", action="store_true", help="use primitives v2")
args = parser.parse_args()

if __name__ == "__main__":
    spp = 128
    num_view = 21
    python_path = "python3"
    render_py_path = os.path.join(CUR_DIR, "render_scene.py")
    repo_root = REPO_ROOT
    target_root = args.target_root
    Path(target_root).makedirs_p()

    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    name = "render_" + target_root.split("/")[-1]
    filename = f"log.render.sub{args.sub:02d}.tot{args.total}.{timestamp}.txt"
    # set up logger
    logger.remove()
    fmt = (
        f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | "
        f"<cyan>{name}</cyan> | "
        f"<lvl>{{level}}</lvl> | "
        f"<lvl>{{message}}</lvl>"
    )

    # logger to file
    log_file = Path(target_root) / filename
    logger.add(log_file, format=fmt)

    # logger to std stream
    logger.add(sys.stdout, format=fmt)
    logger.info(f"Args: {args}")

    if args.primitives:
        total_scene = 2000
        scene_names = np.arange(total_scene)
        sub_total_scene = len(scene_names) // args.total
        sub_scene_list = (
            scene_names[(args.sub - 1) * sub_total_scene : args.sub * sub_total_scene]
            if args.sub < args.total
            else scene_names[(args.sub - 1) * sub_total_scene :]
        )
    else:
        total_scene = 1000
        scene_names = np.arange(total_scene)
        sub_total_scene = len(scene_names) // args.total
        sub_scene_list = []
        if args.sub < args.total:
            for s in scene_names[(args.sub - 1) * sub_total_scene : args.sub * sub_total_scene]:
                sub_scene_list.append(f"0-{s}")
                sub_scene_list.append(f"1-{s}")
        else:
            for s in scene_names[(args.sub - 1) * sub_total_scene :]:
                sub_scene_list.append(f"0-{s}")
                sub_scene_list.append(f"1-{s}")

    logger.info(f"Generating {len(sub_scene_list)} scenes from {sub_scene_list[0]} to {sub_scene_list[-1]}")
    start_time = time.time()

    for sc in sub_scene_list:
        if osp.exists(osp.join(target_root, f"{sc}-{num_view-1}/depthR_colored.png")):
            logger.info(f"Skip scene {sc}")
            continue
        logger.info(f"Generating scene {sc}")
        o = (
            f"{python_path} {render_py_path} --repo-root {repo_root}"
            f" --target-root {target_root} --spp {spp} --nv {num_view} -s {sc}"
        )
        if args.rand_pattern:
            o += " --rand-pattern"
        if args.fixed_angle:
            o += " --fixed-angle"
        if args.primitives:
            o += " --primitives"
        if args.primitives_v2:
            o += " --primitives-v2"
        os.system(o)
        logger.info(f"time: {time.time() - start_time}")
