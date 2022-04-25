import os
import os.path as osp
import sys
import time

import numpy as np
import sapien.core as sapien
from loguru import logger
from path import Path

CUR_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from data_rendering.render_scene import render_gt_depth_label, render_scene

if __name__ == "__main__":
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

    spp = 128
    num_view = 21

    repo_root = REPO_ROOT
    target_root = args.target_root
    data_root = osp.join(args.target_root, "data")
    Path(data_root).makedirs_p()

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

    # build scene
    sim = sapien.Engine()
    sim.set_log_level("warning")
    sapien.KuafuRenderer.set_log_level("warning")

    render_config = sapien.KuafuConfig()
    render_config.use_viewer = False
    render_config.use_denoiser = True
    render_config.spp = spp
    render_config.max_bounces = 8

    renderer = sapien.KuafuRenderer(render_config)
    sim.set_renderer(renderer)

    for sc in sub_scene_list:
        if osp.exists(osp.join(target_root, f"{sc}-{num_view-1}/meta.pkl")):
            logger.info(f"Skip scene {sc} rendering")
            continue
        logger.info(f"Rendering scene {sc}")
        render_scene(
            sim=sim,
            renderer=renderer,
            scene_id=sc,
            repo_root=repo_root,
            target_root=target_root,
            spp=spp,
            num_views=num_view,
            rand_pattern=args.rand_pattern,
            fixed_angle=args.fixed_angle,
            primitives=args.primitives,
            primitives_v2=args.primitives_v2,
        )

    renderer = None
    sim = None

    sim_vk = sapien.Engine()
    sim_vk.set_log_level("warning")

    renderer = sapien.VulkanRenderer(offscreen_only=True)
    renderer.set_log_level("warning")
    sim_vk.set_renderer(renderer)

    for sc in sub_scene_list:
        if osp.exists(osp.join(target_root, f"{sc}-{num_view - 1}/depthR_colored.png")):
            logger.info(f"Skip scene {sc} gt depth and seg")
            continue
        logger.info(f"Generating scene {sc} gt depth and seg")
        render_gt_depth_label(
            sim=sim_vk,
            renderer=renderer,
            scene_id=sc,
            repo_root=repo_root,
            target_root=target_root,
            spp=spp,
            num_views=num_view,
            rand_pattern=args.rand_pattern,
            fixed_angle=args.fixed_angle,
            primitives=args.primitives,
            primitives_v2=args.primitives_v2,
        )
