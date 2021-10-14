import yaml
from yacs.config import CfgNode as CN

_C = CN()

# running mode
_C.MODE = "train"
# name
_C.EXPERIMENT_NAME = "debug"
# random seed
_C.SEED = 222

# system config
_C.SYSTEM = CN()
_C.SYSTEM.GPU = ['cuda:0']

# model architecture
_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 51
_C.MODEL.ARCH = "slowfast"
_C.MODEL.NAME = "slowfast"
_C.MODEL.RESUME = ""
# =====>slowfast
_C.MODEL.SLOWFAST = CN()
# =====>vivit
_C.MODEL.VIVIT = CN()
_C.MODEL.VIVIT.INPUT_SIZE = (224, 224)
_C.MODEL.VIVIT.FRAME_PER_CLIP = 32
# tubelet size: T*H*W
_C.MODEL.VIVIT.T = 2
_C.MODEL.VIVIT.H = 16
_C.MODEL.VIVIT.W = 16
# MSA
_C.MODEL.VIVIT.NUM_HEAD = 8
# transformer layer
_C.MODEL.VIVIT.NUM_LAYER = 12
# feature dimension
_C.MODEL.VIVIT.D_MODEL = 512
_C.MODEL.VIVIT.D_FEATURE = 2048

# data
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 8
_C.DATA.NUM_WORKER = 56
_C.DATA.PREFETCH = 1
_C.DATA.IMG_SIZE = (224, 224)
_C.DATA.FRAME_PER_CLIP = 64
# drop some frame
_C.DATA.SKIP_FRAME = 2
# chose a dataset
_C.DATA.DATASET = "hmdb51"
# =====>kinetics
_C.DATA.KINETICS = CN()
_C.DATA.KINETICS.VIDEO_FOLDER = None
# =====>hmdb51
_C.DATA.HMDB51 = CN()
_C.DATA.HMDB51.VIDEO_FOLDER = None
# split set
_C.DATA.HMDB51.ANNOTATION = None

# train
_C.TRAIN = CN()
_C.TRAIN.EPOCH = 100
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.ACCUMULATION_STEP = 1
# save checkpoint
_C.TRAIN.SAVE_FREQ = 1
# run evaluation during training
_C.TRAIN.EVAL_FREQ = 1  # set -1 to disable evaluation during training
# resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True

# lr settings
_C.TRAIN.LR_BASE = 1e-4
_C.TRAIN.CLIP_GRAD = 5.0
# optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adam"
# scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "step"
_C.TRAIN.LR_SCHEDULER.WARMUP_EPOCH = 10
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCH = 30

_C.LOG = CN()
_C.LOG.LOG_DIR = "./log"


def check_config(config):
    if config.MODEL.ARCH == "vivit":
        assert config.DATA.FRAME_PER_CLIP // config.DATA.SKIP_FRAME == config.MODEL.VIVIT.FRAME_PER_CLIP


def get_config(args):
    config = _C.clone()
    config.merge_from_file(args.config)
    config.freeze()
    check_config(config)
    return config


default_cfg = _C
default_cfg.freeze()
