import os
import os.path as osp
import time

import numpy as np
from path import Path

CUR_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(osp.join(osp.dirname(__file__), ".."))

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--target-root", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    spp = 128
    num_view = 21
    python_path = "python3"
    render_py_path = os.path.join(CUR_DIR, "render_scene.py")
    repo_root = REPO_ROOT
    target_root = args.target_root
    Path(target_root).makedirs_p()

    sub_scene_list = [
        "0-300002",
        "0-300100",
        "0-300103",
        "0-300104",
        "0-300106",
        "0-300110",
        "0-300113",
        "0-300116",
        "0-300122",
        "0-300124",
        "0-300126",
        "0-300128",
        "0-300135",
        "0-300138",
        "0-300155",
        "0-300158",
        "0-300163",
        "0-300169",
        "0-300172",
        "0-300183",
        "0-300197",
        "1-300101",
        "1-300117",
        "1-300135",
    ]

    print(f"Generating {len(sub_scene_list)} scenes from {sub_scene_list[0]} to {sub_scene_list[-1]}")
    start_time = time.time()

    for sc in sub_scene_list:
        if osp.exists(osp.join(target_root, f"{sc}-{num_view-1}/depthR_colored.png")):
            print(f"Skip scene {sc}")
            continue
        print(f"Generating scene {sc}")
        o = (
            f"{python_path} {render_py_path} --repo-root {repo_root}"
            f" --target-root {target_root} --spp {spp} --nv {num_view} -s {sc} --fixed-angle"
        )
        os.system(o)
        print(f"time: {time.time() - start_time}")
