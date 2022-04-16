from active_zero2.datasets.messytable import MessyTableDataset


def build_dataset(cfg, mode, domain):
    if mode == "train":
        dataset_cfg = cfg.DATA.TRAIN
    elif mode == "val":
        dataset_cfg = cfg.DATA.VAL
    elif mode == "test":
        dataset_cfg = cfg.DATA.TEST
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")

    if domain == "sim":
        dataset_cfg = dataset_cfg.SIM
    elif domain == "real":
        dataset_cfg = dataset_cfg.REAL
    else:
        raise ValueError(f"Unknown dataset domain: {domain}")

    if not dataset_cfg.SPLIT_FILE:
        return None

    dataset = MessyTableDataset(
        mode=mode,
        domain=domain,
        root_dir=dataset_cfg.ROOT_DIR,
        split_file=dataset_cfg.SPLIT_FILE,
        height=dataset_cfg.HEIGHT,
        width=dataset_cfg.WIDTH,
        meta_name=dataset_cfg.META_NAME,
        depth_name=dataset_cfg.DEPTH_NAME,
        normal_name=dataset_cfg.NORMAL_NAME,
        left_name=dataset_cfg.LEFT_NAME,
        right_name=dataset_cfg.RIGHT_NAME,
        left_pattern_name=dataset_cfg.LEFT_PATTERN_NAME,
        right_pattern_name=dataset_cfg.RIGHT_PATTERN_NAME,
    )

    return dataset
