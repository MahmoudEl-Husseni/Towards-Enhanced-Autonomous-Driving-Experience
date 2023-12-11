import os
from colorama import Fore, Back, Style

class config:
  EXPERIMENT_NAME = "cnn motion: EffnetB0"
  ENCODER_NAME = 'efficientnet-b1'

  VIS_HEIGHT = 640
  VIS_WIDTH = 480

  TRAIN_SZ = 0.8
  VAL_SZ = 0.1
  TEST_SZ = 0.1

  TRAIN_BS = 16
  VAL_BS = 4
  TEST_BS = 4

  N_TRAJ = 6
  FUTURE_TS = 80

  LR = 1e-3
  EPOCHS = 100
  LOG_STEP=10
  DEVICE = 'cuda'
  LOSS = 'neg_multi_log_likelihood'

  LOAD_MODEL = False
  LOAD_MODEL_FILE = "last_model.pth"

  CKPT_EPOCH = 10

class DIR:
  MAIN_DIR = "/main/Datasets/waymo/testing"
  DATA_DIR = os.path.join(MAIN_DIR, "waymo-dataset-testing")
  VIS_DATA_DIR = os.path.join(MAIN_DIR, "vis-data")
  RENDER_DIR = os.path.join(MAIN_DIR, "render")

  OUT_DIR = os.path.join(MAIN_DIR, "output")
  CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
  TB_DIR = os.path.join(OUT_DIR, "tb")
  VIS_DIR = os.path.join(OUT_DIR, "vis")


RENDER_ARGS = {
    'out_path': DIR.RENDER_DIR,
    'data': DIR.DATA_DIR,
    'n_shards': 8,
    'validate': True,
    'use_vectorize': False,
    'n_jobs': 20,
    'each': 0,
    'no_valid': False
}

blk = Style.BRIGHT + Fore.BLACK
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
grn_bck = Back.GREEN
res = Style.RESET_ALL