""" 
@Date: 2022/07/17
@description:
"""
import os
import logging
from yacs.config import CfgNode as CN

_C = CN()
_C.DEBUG = False
_C.MODE = 'train'
_C.TAG = 'default'
_C.COMMENT = 'add some comments to help you understand'
_C.SHOW_BAR = True
_C.SAVE_EVAL = False
_C.MODEL = CN()
_C.MODEL.NAME = 'model_name'
_C.MODEL.SAVE_BEST = True
_C.MODEL.SAVE_LAST = True
_C.MODEL.ARGS = []
_C.MODEL.FINE_TUNE = []

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.SCRATCH = False
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.DETERMINISTIC = False
_C.TRAIN.SAVE_FREQ = 5

_C.TRAIN.BASE_LR = 5e-4

_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.RESUME_LAST = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.DEVICE = 'cuda'

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = ''
_C.TRAIN.LR_SCHEDULER.ARGS = []


# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Criterion
_C.TRAIN.CRITERION = CN()
# Depth loss
_C.TRAIN.CRITERION.DEPTH = CN()
_C.TRAIN.CRITERION.DEPTH.NAME = 'depth'
_C.TRAIN.CRITERION.DEPTH.LOSS = 'L1Loss'
_C.TRAIN.CRITERION.DEPTH.WEIGHT = 0.0
_C.TRAIN.CRITERION.DEPTH.WEIGHTS = []
_C.TRAIN.CRITERION.DEPTH.NEED_ALL = False
# Segmentation loss
_C.TRAIN.CRITERION.SEG = CN()
_C.TRAIN.CRITERION.SEG.NAME = 'aux_segmentation'
_C.TRAIN.CRITERION.SEG.LOSS = 'SegLoss'
_C.TRAIN.CRITERION.SEG.WEIGHT = 0.0
_C.TRAIN.CRITERION.SEG.WEIGHTS = []
_C.TRAIN.CRITERION.SEG.NEED_ALL = True
# Ratio loss
_C.TRAIN.CRITERION.RATIO = CN()
_C.TRAIN.CRITERION.RATIO.NAME = 'ratio'
_C.TRAIN.CRITERION.RATIO.LOSS = 'L1Loss'
_C.TRAIN.CRITERION.RATIO.WEIGHT = 0.0
_C.TRAIN.CRITERION.RATIO.WEIGHTS = []
_C.TRAIN.CRITERION.RATIO.NEED_ALL = False
# Grad loss
_C.TRAIN.CRITERION.GRAD = CN()
_C.TRAIN.CRITERION.GRAD.NAME = 'grad'
_C.TRAIN.CRITERION.GRAD.LOSS = 'GradLoss'
_C.TRAIN.CRITERION.GRAD.WEIGHT = 0.0
_C.TRAIN.CRITERION.GRAD.WEIGHTS = [1.0, 1.0]
_C.TRAIN.CRITERION.GRAD.NEED_ALL = True

_C.TRAIN.VIS_MERGE = True
_C.TRAIN.VIS_WEIGHT = 1024
# -----------------------------------------------------------------------------
# Output settings
# -----------------------------------------------------------------------------
_C.CKPT = CN()
_C.CKPT.PYTORCH = './'
_C.CKPT.ROOT = "./checkpoints"
_C.CKPT.DIR = os.path.join(_C.CKPT.ROOT, _C.MODEL.NAME, _C.TAG)
_C.CKPT.RESULT_DIR = os.path.join(_C.CKPT.DIR, 'results', _C.MODE)

_C.LOGGER = CN()
_C.LOGGER.DIR = os.path.join(_C.CKPT.DIR, "logs")
_C.LOGGER.LEVEL = logging.DEBUG

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2'), Please confirm your device support FP16(Half).
# overwritten by command line argument
_C.AMP_OPT_LEVEL = 'O1'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False

# -----------------------------------------------------------------------------
# FIX
# -----------------------------------------------------------------------------
_C.LOCAL_RANK = 0
_C.WORLD_SIZE = 0

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Sub dataset of pano_s2d3d
_C.DATA.SUBSET = None
# Dataset name
_C.DATA.DATASET = 'mp3d'
# Path to dataset, could be overwritten by command line argument
_C.DATA.DIR = ''
# Max wall number
_C.DATA.WALL_NUM = 0  # all
# Panorama image size
_C.DATA.SHAPE = [512, 1024]
# Really camera height
_C.DATA.CAMERA_HEIGHT = 1.6
# Debug use, fast test performance of model
_C.DATA.FOR_TEST_INDEX = None

# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 8
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# Training augment
_C.DATA.AUG = CN()
_C.DATA.AUG.FLIP = True
_C.DATA.AUG.STRETCH = True
_C.DATA.AUG.ROTATE = True
_C.DATA.AUG.GAMMA = True

_C.DATA.KEYS = []


_C.EVAL = CN()
_C.EVAL.POST_PROCESSING = None
_C.EVAL.NEED_CPE = True
_C.EVAL.NEED_F1 = False
_C.EVAL.FORCE_CUBE = True
_C.EVAL.NEED_RMSE = False


def merge_from_file(cfg_path):
    config = _C.clone()
    config.merge_from_file(cfg_path)
    return config


def get_config(args=None):
    config = _C.clone()
    if args:
        if 'cfg' in args and args.cfg:
            config.merge_from_file(args.cfg)

        if 'mode' in args and args.mode:
            config.MODE = args.mode

        if 'debug' in args and args.debug:
            config.DEBUG = args.debug

        if 'bs' in args and args.bs:
            config.DATA.BATCH_SIZE = args.bs

        if 'save_eval' in args and args.save_eval:
            config.SAVE_EVAL = True

        if 'val_name' in args and args.val_name:
            config.VAL_NAME = args.val_name

        if 'post_processing' in args and args.post_processing:
            config.EVAL.POST_PROCESSING = args.post_processing

        if 'need_cpe' in args and args.need_cpe:
            config.EVAL.NEED_CPE = args.need_cpe

        if 'need_f1' in args and args.need_f1:
            config.EVAL.NEED_F1 = args.need_f1

        if 'need_rmse' in args and args.need_rmse:
            config.EVAL.NEED_RMSE = args.need_rmse

        if 'force_cube' in args and args.force_cube:
            config.EVAL.FORCE_CUBE = args.force_cube

        if 'wall_num' in args and args.wall_num:
            config.DATA.WALL_NUM = args.wall_num

    config.CKPT.DIR = os.path.join(config.CKPT.ROOT, "My_Layout_Net",
                                   config.TAG, 'debug' if config.DEBUG else '')
    config.CKPT.RESULT_DIR = os.path.join(config.CKPT.DIR, 'results', config.MODE)
    config.LOGGER.DIR = os.path.join(config.CKPT.DIR, "logs")

    config.freeze()
    return config


def get_rank_config(cfg, local_rank, world_size):
    local_rank = 0 if local_rank is None else local_rank
    config = cfg.clone()
    config.defrost()

    config.LOCAL_RANK = local_rank
    config.WORLD_SIZE = world_size
    config.SEED = config.SEED + local_rank

    config.freeze()
    return config
