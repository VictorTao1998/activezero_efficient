from yacs.config import CfgNode as CN
from yacs.config import load_cfg

_C = CN()
cfg = _C

# ---------------------------------------------------------------------------- #
# Resume
# ---------------------------------------------------------------------------- #
# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True
# Whether to resume the optimizer and the scheduler
_C.RESUME_STATES = True
# Path of weights to resume
_C.RESUME_PATH = ""
_C.RESUME_STRICT = True

# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = "@"

# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
# -1 means not to set explicitly.
_C.RNG_SEED = 1

# ---------------------------------------------------------------------------- #
# DATA
# ---------------------------------------------------------------------------- #
_C.DATA = CN()
_C.DATA.NUM_CLASSES = 17

_C.DATA.TRAIN = CN()

_C.DATA.TRAIN.SIM = CN()
_C.DATA.TRAIN.SIM.ROOT_DIR = ""
_C.DATA.TRAIN.SIM.SPLIT_FILE = ""
_C.DATA.TRAIN.SIM.HEIGHT = 256
_C.DATA.TRAIN.SIM.WIDTH = 512
_C.DATA.TRAIN.SIM.META_NAME = ""
_C.DATA.TRAIN.SIM.DEPTH_NAME = ""
_C.DATA.TRAIN.SIM.NORMAL_NAME = ""
_C.DATA.TRAIN.SIM.LEFT_NAME = ""
_C.DATA.TRAIN.SIM.RIGHT_NAME = ""
_C.DATA.TRAIN.SIM.LEFT_PATTERN_NAME = ""
_C.DATA.TRAIN.SIM.RIGHT_PATTERN_NAME = ""
_C.DATA.TRAIN.SIM.LABEL_NAME = ""
_C.DATA.TRAIN.REAL = CN()
_C.DATA.TRAIN.REAL.ROOT_DIR = ""
_C.DATA.TRAIN.REAL.SPLIT_FILE = ""
_C.DATA.TRAIN.REAL.HEIGHT = 256
_C.DATA.TRAIN.REAL.WIDTH = 512
_C.DATA.TRAIN.REAL.META_NAME = ""
_C.DATA.TRAIN.REAL.DEPTH_NAME = ""
_C.DATA.TRAIN.REAL.NORMAL_NAME = ""
_C.DATA.TRAIN.REAL.LEFT_NAME = ""
_C.DATA.TRAIN.REAL.RIGHT_NAME = ""
_C.DATA.TRAIN.REAL.LEFT_PATTERN_NAME = ""
_C.DATA.TRAIN.REAL.RIGHT_PATTERN_NAME = ""
_C.DATA.TRAIN.REAL.LABEL_NAME = ""

_C.DATA.VAL = CN()

_C.DATA.VAL.SIM = CN()
_C.DATA.VAL.SIM.ROOT_DIR = ""
_C.DATA.VAL.SIM.SPLIT_FILE = ""
_C.DATA.VAL.SIM.HEIGHT = 256
_C.DATA.VAL.SIM.WIDTH = 512
_C.DATA.VAL.SIM.META_NAME = ""
_C.DATA.VAL.SIM.DEPTH_NAME = ""
_C.DATA.VAL.SIM.NORMAL_NAME = ""
_C.DATA.VAL.SIM.LEFT_NAME = ""
_C.DATA.VAL.SIM.RIGHT_NAME = ""
_C.DATA.VAL.SIM.LEFT_PATTERN_NAME = ""
_C.DATA.VAL.SIM.RIGHT_PATTERN_NAME = ""
_C.DATA.VAL.SIM.LABEL_NAME = ""
_C.DATA.VAL.REAL = CN()
_C.DATA.VAL.REAL.ROOT_DIR = ""
_C.DATA.VAL.REAL.SPLIT_FILE = ""
_C.DATA.VAL.REAL.HEIGHT = 256
_C.DATA.VAL.REAL.WIDTH = 512
_C.DATA.VAL.REAL.META_NAME = ""
_C.DATA.VAL.REAL.DEPTH_NAME = ""
_C.DATA.VAL.REAL.NORMAL_NAME = ""
_C.DATA.VAL.REAL.LEFT_NAME = ""
_C.DATA.VAL.REAL.RIGHT_NAME = ""
_C.DATA.VAL.REAL.LEFT_PATTERN_NAME = ""
_C.DATA.VAL.REAL.RIGHT_PATTERN_NAME = ""
_C.DATA.VAL.REAL.LABEL_NAME = ""

_C.DATA.TEST = CN()

_C.DATA.TEST.SIM = CN()
_C.DATA.TEST.SIM.ROOT_DIR = ""
_C.DATA.TEST.SIM.SPLIT_FILE = ""
_C.DATA.TEST.SIM.HEIGHT = 544
_C.DATA.TEST.SIM.WIDTH = 960
_C.DATA.TEST.SIM.META_NAME = ""
_C.DATA.TEST.SIM.DEPTH_NAME = ""
_C.DATA.TEST.SIM.NORMAL_NAME = ""
_C.DATA.TEST.SIM.LEFT_NAME = ""
_C.DATA.TEST.SIM.RIGHT_NAME = ""
_C.DATA.TEST.SIM.LEFT_PATTERN_NAME = ""
_C.DATA.TEST.SIM.RIGHT_PATTERN_NAME = ""
_C.DATA.TEST.SIM.LABEL_NAME = ""
_C.DATA.TEST.REAL = CN()
_C.DATA.TEST.REAL.ROOT_DIR = ""
_C.DATA.TEST.REAL.SPLIT_FILE = ""
_C.DATA.TEST.REAL.HEIGHT = 544
_C.DATA.TEST.REAL.WIDTH = 960
_C.DATA.TEST.REAL.META_NAME = ""
_C.DATA.TEST.REAL.DEPTH_NAME = ""
_C.DATA.TEST.REAL.NORMAL_NAME = ""
_C.DATA.TEST.REAL.LEFT_NAME = ""
_C.DATA.TEST.REAL.RIGHT_NAME = ""
_C.DATA.TEST.REAL.LEFT_PATTERN_NAME = ""
_C.DATA.TEST.REAL.RIGHT_PATTERN_NAME = ""
_C.DATA.TEST.REAL.LABEL_NAME = ""


# data augmentation
_C.DATA_AUG = CN()
_C.DATA_AUG.COLOR_JITTER = True
_C.DATA_AUG.GAUSSIAN_BLUR = True
_C.DATA_AUG.GAUSSIAN_MIN = 0.1
_C.DATA_AUG.GAUSSIAN_MAX = 2.0
_C.DATA_AUG.GAUSSIAN_KERNEL = 9
_C.DATA_AUG.BRIGHT_MIN = 0.4
_C.DATA_AUG.BRIGHT_MAX = 1.4
_C.DATA_AUG.CONTRAST_MIN = 0.8
_C.DATA_AUG.CONTRAST_MAX = 1.2

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #

# PSMNet, PSMNetRay
_C.MODEL_TYPE = ""

_C.PSMNet = CN()
_C.PSMNet.MAX_DISP = 192

_C.CFNet = CN()
_C.CFNet.MAX_DISP = 256
_C.CFNet.USE_CONCAT_VOLUME = True

_C.PSMNetRange = CN()
_C.PSMNetRange.MIN_DISP = 0
_C.PSMNetRange.MAX_DISP = 0
_C.PSMNetRange.NUM_DISP = 0
_C.PSMNetRange.SET_ZERO = False

# ---------------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------------- #

_C.LOSS = CN()
_C.LOSS.SIM_REPROJ = CN()
_C.LOSS.SIM_REPROJ.WEIGHT = 0.0
_C.LOSS.SIM_REPROJ.USE_MASK = True
_C.LOSS.SIM_REPROJ.PATCH_SIZE = 11
_C.LOSS.SIM_REPROJ.ONLY_LAST_PRED = True
_C.LOSS.SIM_DISP = CN()
_C.LOSS.SIM_DISP.WEIGHT = 1.0

_C.LOSS.REAL_REPROJ = CN()
_C.LOSS.REAL_REPROJ.WEIGHT = 1.0
_C.LOSS.REAL_REPROJ.USE_MASK = False
_C.LOSS.REAL_REPROJ.PATCH_SIZE = 11
_C.LOSS.REAL_REPROJ.ONLY_LAST_PRED = True
_C.LOSS.REAL_DISP = CN()
_C.LOSS.REAL_DISP.WEIGHT = 0.0

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = ""

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 0.0
# Maximum norm of gradients. Non-positive for disable
_C.OPTIMIZER.MAX_GRAD_NORM = 0.0

# Specific parameters of optimizers
_C.OPTIMIZER.SGD = CN()
_C.OPTIMIZER.SGD.momentum = 0.9

_C.OPTIMIZER.Adam = CN()
_C.OPTIMIZER.Adam.betas = (0.9, 0.999)

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.TYPE = ""

# Specific parameters of schedulers
_C.LR_SCHEDULER.StepLR = CN()
_C.LR_SCHEDULER.StepLR.step_size = 0
_C.LR_SCHEDULER.StepLR.gamma = 0.1

_C.LR_SCHEDULER.MultiStepLR = CN()
_C.LR_SCHEDULER.MultiStepLR.milestones = ()
_C.LR_SCHEDULER.MultiStepLR.gamma = 0.1

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Batch size
_C.TRAIN.BATCH_SIZE = 1
# Number of workers (dataloader)
_C.TRAIN.NUM_WORKERS = 1
# Period to save checkpoints. 0 for disable
_C.TRAIN.CHECKPOINT_PERIOD = 1
# Period to log training status. 0 for disable
_C.TRAIN.LOG_PERIOD = 100
# Max number of checkpoints to keep
_C.TRAIN.MAX_TO_KEEP = 5
# Max number of iteration
_C.TRAIN.MAX_ITER = 1


# ---------------------------------------------------------------------------- #
# Specific validation options
# ---------------------------------------------------------------------------- #
_C.VAL = CN()

# Batch size
_C.VAL.BATCH_SIZE = 1
# Number of workers (dataloader)
_C.VAL.NUM_WORKERS = 1
# Period to validate. 0 for disable
_C.VAL.PERIOD = 0
# Period to log validation status. 0 for disable
_C.VAL.LOG_PERIOD = 100

# The metric for best validation performance
_C.VAL.METRIC = ""
_C.VAL.METRIC_ASCEND = True


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

_C.TEST.BATCH_SIZE = 1
_C.TEST.NUM_WORKERS = 1
# The path of weights to be tested. "@" has similar syntax as OUTPUT_DIR.
# If not set, the last checkpoint will be used by default.
_C.TEST.WEIGHT = ""

_C.TEST.LOG_PERIOD = 10
_C.TEST.METRIC = ""

_C.TEST.USE_MASK = True
_C.TEST.MAX_DISP = 192
_C.TEST.IS_DEPTH = False
