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
_C.DATA.TRAIN.SIM.DEPTH_R_NAME = ""
_C.DATA.TRAIN.SIM.LEFT_OFF_NAME = ""
_C.DATA.TRAIN.SIM.RIGHT_OFF_NAME = ""
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
_C.DATA.TRAIN.REAL.DEPTH_R_NAME = ""
_C.DATA.TRAIN.REAL.LEFT_OFF_NAME = ""
_C.DATA.TRAIN.REAL.RIGHT_OFF_NAME = ""

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
_C.DATA.VAL.SIM.DEPTH_R_NAME = ""
_C.DATA.VAL.SIM.LEFT_OFF_NAME = ""
_C.DATA.VAL.SIM.RIGHT_OFF_NAME = ""
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
_C.DATA.VAL.REAL.DEPTH_R_NAME = ""
_C.DATA.VAL.REAL.LEFT_OFF_NAME = ""
_C.DATA.VAL.REAL.RIGHT_OFF_NAME = ""

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
_C.DATA.TEST.SIM.DEPTH_R_NAME = ""
_C.DATA.TEST.SIM.LEFT_OFF_NAME = ""
_C.DATA.TEST.SIM.RIGHT_OFF_NAME = ""
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
_C.DATA.TEST.REAL.DEPTH_R_NAME = ""
_C.DATA.TEST.REAL.LEFT_OFF_NAME = ""
_C.DATA.TEST.REAL.RIGHT_OFF_NAME = ""


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
_C.DATA_AUG.SATURATION_MIN = 0.4
_C.DATA_AUG.SATURATION_MAX = 1.6
_C.DATA_AUG.HUE_MIN = -0.4
_C.DATA_AUG.HUE_MAX = 0.4

_C.DATA_AUG.SIM_IR = False
_C.DATA_AUG.SPECKLE_SHAPE_MIN = 350
_C.DATA_AUG.SPECKLE_SHAPE_MAX = 450
_C.DATA_AUG.GAUSSIAN_MU = -1e-3
_C.DATA_AUG.GAUSSIAN_SIGMA = 4e-3

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

_C.PSMNetRange4 = CN()
_C.PSMNetRange4.MIN_DISP = 0
_C.PSMNetRange4.MAX_DISP = 0
_C.PSMNetRange4.NUM_DISP = 0
_C.PSMNetRange4.SET_ZERO = False

_C.PSMNetDilation = CN()
_C.PSMNetDilation.MIN_DISP = 0
_C.PSMNetDilation.MAX_DISP = 0
_C.PSMNetDilation.NUM_DISP = 0
_C.PSMNetDilation.SET_ZERO = False
_C.PSMNetDilation.DILATION = 3
_C.PSMNetDilation.USE_OFF = False

_C.PSMNetKPAC = CN()
_C.PSMNetKPAC.MIN_DISP = 0
_C.PSMNetKPAC.MAX_DISP = 0
_C.PSMNetKPAC.NUM_DISP = 0
_C.PSMNetKPAC.SET_ZERO = False
_C.PSMNetKPAC.DILATION_LIST = (3, 5, 9)
_C.PSMNetKPAC.DILATION = 3

_C.PSMNetGrad = CN()
_C.PSMNetGrad.MIN_DISP = 0
_C.PSMNetGrad.MAX_DISP = 0
_C.PSMNetGrad.NUM_DISP = 0
_C.PSMNetGrad.SET_ZERO = False
_C.PSMNetGrad.DILATION = 3
_C.PSMNetGrad.EPSILON = 1.0
_C.PSMNetGrad.USE_OFF = False

_C.SMDNet = CN()
# data
_C.SMDNet.NUM_SAMPLES = 0
_C.SMDNet.DILATION = 10
# network
_C.SMDNet.OUTPUT_REPRESENTATION = "bimodal"
_C.SMDNet.MAX_DISP = 192
_C.SMDNet.NO_SINE = False
_C.SMDNet.NO_RESIDUAL = False

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

_C.LOSS.SIM_SMD = CN()
_C.LOSS.SIM_SMD.WEIGHT = 0.0

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
_C.TEST.WEIGHT = '/edward-slow-vol/checkpoints/smd_activezero2/model_005000.pth'

_C.TEST.LOG_PERIOD = 10
_C.TEST.METRIC = ""

_C.TEST.USE_MASK = True
_C.TEST.MAX_DISP = 192
_C.TEST.DEPTH_RANGE = (0.2, 1.25)
_C.TEST.IS_DEPTH = False

# ---------------------------------------------------------------------------- #
# Adversarial Learning
# ---------------------------------------------------------------------------- #

PI = 3.1415926
_C.PSMNetADV = CN()
_C.PSMNetADV.MIN_DISP = 0
_C.PSMNetADV.MAX_DISP = 0
_C.PSMNetADV.NUM_DISP = 0
_C.PSMNetADV.SET_ZERO = False
_C.PSMNetADV.DILATION = 3
_C.PSMNetADV.EPSILON = 1.0
_C.PSMNetADV.D_CHANNELS = 16
_C.PSMNetADV.DISP_ENCODING = (PI / 32, PI / 8, PI / 2)
_C.PSMNetADV.WGANGP_NORM = 1.0
_C.PSMNetADV.WGANGP_LAMBDA = 10.0
_C.PSMNetADV.USE_SIM_PRED = False  # ADV between sim disp pred and real disp pred or gt disp and real disp pred

_C.PSMNetADV4 = CN()
_C.PSMNetADV4.MIN_DISP = 0
_C.PSMNetADV4.MAX_DISP = 0
_C.PSMNetADV4.NUM_DISP = 0
_C.PSMNetADV4.SET_ZERO = False
_C.PSMNetADV4.DILATION = 3
_C.PSMNetADV4.EPSILON = 1.0
_C.PSMNetADV4.D_CHANNELS = 16
_C.PSMNetADV4.DISP_ENCODING = (PI / 32, PI / 8, PI / 2)
_C.PSMNetADV4.WGANGP_NORM = 1.0
_C.PSMNetADV4.WGANGP_LAMBDA = 10.0
_C.PSMNetADV4.USE_SIM_PRED = False  # ADV between sim disp pred and real disp pred or gt disp and real disp pred
_C.PSMNetADV4.D_TYPE = "D3"

_C.PSMNetGrad2DADV = CN()
_C.PSMNetGrad2DADV.MIN_DISP = 0
_C.PSMNetGrad2DADV.MAX_DISP = 0
_C.PSMNetGrad2DADV.NUM_DISP = 0
_C.PSMNetGrad2DADV.SET_ZERO = False
_C.PSMNetGrad2DADV.DILATION = 3
_C.PSMNetGrad2DADV.EPSILON = 1.0
_C.PSMNetGrad2DADV.D_CHANNELS = 16
_C.PSMNetGrad2DADV.WGANGP_NORM = 1.0
_C.PSMNetGrad2DADV.WGANGP_LAMBDA = 10.0
_C.PSMNetGrad2DADV.USE_SIM_PRED = False  # ADV between sim disp pred and real disp pred or gt disp and real disp pred
_C.PSMNetGrad2DADV.SUB_AVG_SIZE = 0
_C.PSMNetGrad2DADV.DISP_GRAD_NORM = "L1"

_C.G_OPTIMIZER = CN()
_C.G_OPTIMIZER.TYPE = ""

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.G_OPTIMIZER.LR = 1e-3
_C.G_OPTIMIZER.WEIGHT_DECAY = 0.0

_C.G_OPTIMIZER.Adam = CN()
_C.G_OPTIMIZER.Adam.betas = (0.5, 0.9)

_C.D_OPTIMIZER = CN()
_C.D_OPTIMIZER.TYPE = ""

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.D_OPTIMIZER.LR = 1e-3

_C.D_OPTIMIZER.Adam = CN()
_C.D_OPTIMIZER.Adam.betas = (0.5, 0.9)

# use the adversarial loss after ADV_ITER
_C.ADV_ITER = 0
_C.LOSS.ADV = 1.0

_C.LOSS.SIM_GRAD = 0.01
_C.LOSS.REAL_GRAD = 0.03
