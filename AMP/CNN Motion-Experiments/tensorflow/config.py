import os
from colorama import Fore, Back, Style

class config:
  EXPERIMENT_NAME = "cnn motion: EffnetB0"
  ENCODER_NAME = 'efficientnet-b1'

  TRAIN_SZ = 0.8
  VAL_SZ = 0.1
  TEST_SZ = 0.1

  TRAIN_BS = 32
  VAL_BS = 32
  TEST_BS = 8

  N_TRAJ = 6

  DEVICE = 'cuda'
  LR = 1e-2
  EPOCHS = 100

class DIR:
  MAIN_DIR = "/main_dir/Datasets/waymo/testing"
  DATA_DIR = os.path.join(MAIN_DIR, "waymo-dataset-testing")
  RENDER_DIR = os.path.join(MAIN_DIR, "render")
  OUT_DIR = os.path.join(MAIN_DIR, "output")

  TB_DIR = os.path.join(OUT_DIR, "tb")

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
