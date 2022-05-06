import cv2
import sys
import os
import os.path as osp
REPO_ROOT = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from tqdm import tqdm
from path import Path
from data_rendering.utils.render_utils import TEXTURE_FOLDER, TEXTURE_LIST, TEXTURE_SQ_FOLDER, TEXTURE_SQ_LIST

resolution = 256
TEXTURE_SQ_FOLDER = "/media/DATA/LINUX_DATA/activezero2/datasets/mini_imagenet_square/"
Path(TEXTURE_SQ_FOLDER).makedirs_p()
TEXTURE_SQ_LIST = "/media/DATA/LINUX_DATA/activezero2/datasets/mini_imagenet_square/list.txt"
f_list = open(TEXTURE_SQ_LIST, "w")


with open(TEXTURE_LIST, "r") as f:
    texture_list = [line.strip() for line in f]

for t in tqdm(texture_list):
    img = cv2.imread(osp.join(TEXTURE_FOLDER, t))
    img = cv2.resize(img, (resolution, resolution))
    cv2.imwrite(osp.join(TEXTURE_SQ_FOLDER, f"{t.split('.')[0]}.png"), img)
    f_list.write(f"{t.split('.')[0]}.png\n")
