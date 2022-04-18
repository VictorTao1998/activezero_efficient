#!/usr/bin/env python
import os
import os.path as osp
import sys

import tensorboardX

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)

import argparse
import gc
import time
import warnings

import torch
from torch.utils.data import DataLoader

from active_zero2.config import cfg
from active_zero2.datasets.build_dataset import build_dataset
from active_zero2.models.build_model import build_model
from active_zero2.utils.cfg_utils import purge_cfg
from active_zero2.utils.checkpoint import CheckpointerV2
from active_zero2.utils.loguru_logger import setup_logger
from active_zero2.utils.metric_logger import MetricLogger
from active_zero2.utils.metrics import ErrorMetric
from active_zero2.utils.reduce import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="ActiveZero2 Evaluation")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        help="path to config file",
        type=str,
    )
    parser.add_argument("-s", "--save-file", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    # Setup the experiment
    # ---------------------------------------------------------------------------- #
    args = parse_args()
    # Load the configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()
    config_name = args.config_file.split("/")[-1].split(".")[0]

    # run name
    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)

    # Parse the output directory
    output_dir = cfg.OUTPUT_DIR
    # Replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace("@", config_path.replace("configs", "outputs"))
        if osp.isdir(output_dir):
            warnings.warn("Output directory exists.")
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(f"ActiveZero2.test [{config_name}]", output_dir, rank=0, filename=f"log.train.{run_name}.txt")
    logger.info(args)
    from active_zero2.utils.collect_env import collect_env_info

    logger.info("Collecting env info (might take some time)\n" + collect_env_info())
    logger.info(f"Loaded config file: '{args.config_file}'")
    logger.info(f"Running with configs:\n{cfg}")
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    # Build model
    set_random_seed(cfg.RNG_SEED)
    model = build_model(cfg)
    logger.info("Build model:\n{}".format(str(model)))
    model = model.cuda()
    # Enable CUDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Build checkpoint
    checkpointer = CheckpointerV2(model, save_dir=output_dir, logger=logger)
    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        checkpointer.load(None, resume=True)

    # Build data loader
    logger.info(f"Build dataloader")
    set_random_seed(cfg.RNG_SEED)
    test_sim_dataset = build_dataset(cfg, mode="test", domain="sim")
    test_real_dataset = build_dataset(cfg, mode="test", domain="real")

    assert cfg.TEST.BATCH_SIZE == 1, "Only support cfg.TEST.BATCH_SIZE=1"

    if test_sim_dataset:
        test_sim_loader = DataLoader(
            test_sim_dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            num_workers=cfg.TEST.NUM_WORKERS,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )

    else:
        test_sim_loader = None
    if test_real_dataset:
        test_real_loader = DataLoader(
            test_real_dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            num_workers=cfg.TEST.NUM_WORKERS,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )
    else:
        test_real_loader = None

    # Build metrics

    metric = ErrorMetric(
        model_type=cfg.MODEL_TYPE, use_mask=cfg.TEST.USE_MASK, max_disp=cfg.TEST.MAX_DISP, is_depth=cfg.TEST.IS_DEPTH
    )

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    model.eval()
    logger.info("Begin evaluation...")
    eval_tic = tic = time.time()
    if test_sim_loader:
        logger.info("Sim Evaluation")
        test_sim_meters = MetricLogger(delimiter="  ")
        test_sim_meters.reset()
        with torch.no_grad():
            for iteration, data_batch in enumerate(test_sim_loader):
                cur_iter = iteration + 1
                data_time = time.time() - tic
                data_dir = data_batch["dir"][0]
                data_batch = {
                    k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)
                }
                # Forward
                pred_dict = model(data_batch)
                err_dict = metric.compute(
                    data_batch, pred_dict, save_folder=output_dir / data_dir if args.save_file else ""
                )

                batch_time = time.time() - tic
                test_sim_meters.update(time=batch_time, data=data_time)
                test_sim_meters.update(**err_dict)

                # Logging
                if cfg.TEST.LOG_PERIOD > 0 and cur_iter % cfg.TEST.LOG_PERIOD == 0:
                    logger.info(
                        "Sim Test "
                        + test_sim_meters.delimiter.join(
                            [
                                "iter: {iter:6d}",
                                "{meters}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            iter=cur_iter,
                            meters=str(test_sim_meters),
                            memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                        )
                    )

                tic = time.time()

        epoch_time_eval = time.time() - eval_tic
        logger.info("Sim Test {}  total_time: {:.2f}s".format(test_sim_meters.summary_str, epoch_time_eval))
    if test_real_loader:
        logger.info("Real Evaluation")
        eval_tic = tic = time.time()
        test_real_meters = MetricLogger(delimiter="  ")
        test_real_meters.reset()
        with torch.no_grad():
            for iteration, data_batch in enumerate(test_real_loader):
                cur_iter = iteration + 1
                data_time = time.time() - tic
                data_dir = data_batch["dir"][0]
                data_batch = {
                    k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)
                }
                # Forward
                pred_dict = model(data_batch)
                err_dict = metric.compute(
                    data_batch, pred_dict, save_folder=output_dir / data_dir if args.save_file else ""
                )

                batch_time = time.time() - tic
                test_real_meters.update(time=batch_time, data=data_time)
                test_real_meters.update(**err_dict)

                # Logging
                if cfg.TEST.LOG_PERIOD > 0 and (cur_iter % cfg.TEST.LOG_PERIOD == 0 or cur_iter == 1):
                    logger.info(
                        "Real Test "
                        + test_real_meters.delimiter.join(
                            [
                                "iter: {iter:6d}",
                                "{meters}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            iter=cur_iter,
                            meters=str(test_real_meters),
                            memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                        )
                    )

                tic = time.time()

        # END
        epoch_time_eval = time.time() - eval_tic
        logger.info("Real Test {}  total_time: {:.2f}s".format(test_real_meters.summary_str, epoch_time_eval))
