import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Error metric for messy-table-dataset


class ErrorMetric(object):
    def __init__(self, model_type: str, use_mask: bool = True, max_disp: int = 192, is_depth: bool = False):
        assert model_type in ["PSMNet"], f"Unknown model type: [{model_type}]"
        self.model_type = model_type
        self.use_mask = use_mask
        self.max_disp = max_disp
        self.is_depth = is_depth

    def forward(self, data_batch, pred_dict, save_folder=""):
        """
        Compute the error metrics for predicted disparity map or depth map
        """
        focal_length = data_batch["focal_length"][0].cpu().numpy()
        baseline = data_batch["baseline"][0].cpu().numpy()

        if self.model_type == "PSMNet":
            prediction = pred_dict["pred3"]

        prediction = prediction.detach().cpu().numpy()[0, 0]
        disp_gt = data_batch["img_disp_l"].cpu().numpy()[0, 0]
        depth_gt = data_batch["img_depth_l"].cpu().numpy()[0, 0]
        if self.is_depth:
            disp_pred = focal_length * baseline / (prediction + 1e-7)
            depth_pred = prediction
        else:
            depth_pred = focal_length * baseline / (prediction + 1e-7)
            disp_pred = prediction

        if self.use_mask:
            mask = np.logical_and(disp_gt > 0, disp_gt < self.max_disp)
        else:
            mask = np.ones_like(disp_gt).astype(np.bool)

        disp_diff = disp_gt - disp_pred
        depth_diff = depth_gt - depth_pred

        epe = np.abs(disp_diff[mask]).mean()
        bad1 = (np.abs(disp_diff[mask]) > 1).sum() / mask.sum()
        bad2 = (np.abs(disp_diff[mask]) > 2).sum() / mask.sum()

        depth_abs_err = np.abs(depth_diff[mask]).mean()
        depth_err2 = (np.abs(depth_diff[mask]) > 2e-3).sum() / mask.sum()
        depth_err4 = (np.abs(depth_diff[mask]) > 4e-3).sum() / mask.sum()
        depth_err8 = (np.abs(depth_diff[mask]) > 8e-3).sum() / mask.sum()

        # TODO: add normal metric

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            plt.imsave(os.path.join(save_folder, "disp_pred.png"), disp_pred, vmin=0.0, vmax=self.max_disp, cmap="jet")
            plt.imsave(os.path.join(save_folder, "disp_gt.png"), disp_gt, vmin=0.0, vmax=self.max_disp, cmap="jet")
            plt.imsave(os.path.join(save_folder, "disp_err.png"), disp_diff, vmin=-8, vmax=8, cmap="jet")
            plt.imsave(os.path.join(save_folder, "depth_err.png"), depth_diff, vmin=-16e-3, vmax=16e-3, cmap="jet")

        err = {}
        err["epe"] = epe
        err["bad1"] = bad1
        err["bad2"] = bad2
        err["depth_abs_err"] = depth_abs_err
        err["depth_err2"] = depth_err2
        err["depth_err4"] = depth_err4
        err["depth_err8"] = depth_err8
        return err


# Error metric for messy-table-dataset object error
def compute_obj_err(disp_gt, depth_gt, disp_pred, focal_length, baseline, label, mask, obj_total_num=17):
    """
    Compute error for each object instance in the scene
    :param disp_gt: GT disparity map, [bs, 1, H, W]
    :param depth_gt: GT depth map, [bs, 1, H, W]
    :param disp_pred: Predicted disparity map, [bs, 1, H, W]
    :param focal_length: Focal length, [bs, 1]
    :param baseline: Baseline of the camera, [bs, 1]
    :param label: Label of the image [bs, 1, H, W]
    :param obj_total_num: Total number of objects in the dataset
    :return: obj_disp_err, obj_depth_err - List of error of each object
             obj_count - List of each object appear count
    """
    depth_pred = focal_length * baseline / disp_pred  # in meters

    obj_list = label.unique()  # TODO this will cause bug if bs > 1, currently only for testing
    obj_num = obj_list.shape[0]

    # Array to store error and count for each object
    total_obj_disp_err = np.zeros(obj_total_num)
    total_obj_depth_err = np.zeros(obj_total_num)
    total_obj_depth_4_err = np.zeros(obj_total_num)
    total_obj_count = np.zeros(obj_total_num)

    for i in range(obj_num):
        obj_id = int(obj_list[i].item())
        obj_mask = label == obj_id
        obj_disp_err = F.l1_loss(disp_gt[obj_mask * mask], disp_pred[obj_mask * mask], reduction="mean").item()
        obj_depth_err = torch.clip(
            torch.abs(depth_gt[obj_mask * mask] * 1000 - depth_pred[obj_mask * mask] * 1000),
            min=0,
            max=100,
        )
        obj_depth_err = torch.mean(obj_depth_err).item()
        obj_depth_diff = torch.abs(depth_gt[obj_mask * mask] - depth_pred[obj_mask * mask])
        obj_depth_err4 = obj_depth_diff[obj_depth_diff > 4e-3].numel() / obj_depth_diff.numel()

        total_obj_disp_err[obj_id] += obj_disp_err
        total_obj_depth_err[obj_id] += obj_depth_err
        total_obj_depth_4_err[obj_id] += obj_depth_err4
        total_obj_count[obj_id] += 1
    return (
        total_obj_disp_err,
        total_obj_depth_err,
        total_obj_depth_4_err,
        total_obj_count,
    )
