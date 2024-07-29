"""
Configuration settings for the Autonomous Driving Project.

This file contains all the configuration parameters used throughout the project. These settings include directories for data storage, model hyperparameters, statistical paths, padding parameters, and various encodings used in the model architecture.

Attributes:
    EXPERIMENT_NAME (str): The name of the current experiment.
    MAIN_DIR (str): The main directory where the Argoverse Dataset is stored.
    TRAIN_DIR (str): The directory for training data.
    VAL_DIR (str): The directory for validation data.
    TEST_DIR (str): The directory for test data.
    OUT_DIR (str): The directory for output files specific to the current experiment.
    TB_DIR (str): The directory for TensorBoard logs.
    CKPT_DIR (str): The directory for model checkpoints.

    N_PAST (int): The number of past time steps considered in the model.
    N_FUTURE (int): The number of future time steps predicted by the model.
    RADIUS_OFFSET (float): Offset value for the radius of radius which defines scene boundaries.
    VELOCITY_DISTANCE_RATIO (int): Ratio used to calculate velocity from distance.
    TRAJ_DT (float): Time delta for temporal vectors.
    LANE_DL (float): Length delta for spatial vectors.

    ARGO_PAST_TIME (int): Past time window for Argoverse data.
    ARGO_SAMPLE_RATE (int): Sample rate for Argoverse data.

    EPOCHS (int): Number of training epochs.
    LOG_STEP (int): Step interval for logging.
    STEPS_PER_EPOCH (int): Number of steps per epoch.

    DEVICE (str): Device used for computation ('cuda' or 'cpu').
    CKPT_EPOCH (int): Interval for saving checkpoints.

    TRAIN_BS (int): Batch size for training.
    VAL_BS (int): Batch size for validation.
    LR (float): Learning rate for the optimizer.

    LANE_MEANS (str): Path to lane means statistics file.
    LANE_STDS (str): Path to lane standard deviations statistics file.
    AGENT_MEANS (str): Path to agent means statistics file.
    AGENT_STDS (str): Path to agent standard deviations statistics file.
    OBJ_MEANS (str): Path to object means statistics file.
    OBJ_STDS (str): Path to object standard deviations statistics file.
    GT_MEANS (str): Path to ground truth means statistics file.
    GT_STDS (str): Path to ground truth standard deviations statistics file.

    OBJ_PAD_LEN (int): Padding length for object data.
    LANE_PAD_LEN (int): Padding length for lane data.

    track_category_mapping (dict): Mapping of track categories to descriptive names.
    object_color_code (dict): Color codes for different object types.
    map_object_type (dict): Mapping of object types to numerical values.

    OUT_ENC_DIM (int): Output encoding dimension.
    N_TRAJ (int): Number of trajectories.
    OUT_DIM (int): Output dimension.

    AGENT_ENC (dict): Configuration for agent encoder.
    OBJ_ENC (dict): Configuration for object encoder.
    LANE_ENC (dict): Configuration for lane encoder.
    GRAPH_AGENT_ENC (dict): Configuration for graph-based agent encoder.
    GRAPH_OBJ_ENC (dict): Configuration for graph-based object encoder.
    GRAPH_LANE_ENC (dict): Configuration for graph-based lane encoder.
    GLOBAL_ENC (dict): Configuration for global encoder.
    GLOBAL_ENC_TRANS (dict): Configuration for global encoder transformer.
    DECODER (dict): Configuration for decoder.

    blk (str): Style code for bright black text.
    red (str): Style code for bright red text.
    blu (str): Style code for bright blue text.
    grn_bck (str): Style code for green background.
    res (str): Style reset code.
"""

import os 
from colorama import Fore, Back, Style

EXPERIMENT_NAME = "Argo-1"

MAIN_DIR = "/main/Argoverse Dataset/"
TRAIN_DIR = os.path.join(MAIN_DIR, "train_interm")
VAL_DIR = os.path.join(MAIN_DIR, "val_interm")
TEST_DIR = os.path.join(MAIN_DIR, "test")

OUT_DIR = os.path.join(MAIN_DIR, f"out/{EXPERIMENT_NAME}_out")
TB_DIR = os.path.join(OUT_DIR, "tb")
CKPT_DIR = os.path.join(OUT_DIR, "ckpt")


# Configs

N_PAST = 60
N_FUTURE = 50
RADIUS_OFFSET = 1.5
VELOCITY_DISTANCE_RATIO = 10
TRAJ_DT = 0.1
LANE_DL = 1e13

ARGO_PAST_TIME = 5
ARGO_SAMPLE_RATE = 10


EPOCHS = 100
LOG_STEP = 10
STEPS_PER_EPOCH = 71

DEVICE = 'cuda'
CKPT_EPOCH = 10

TRAIN_BS = 64
VAL_BS = 128
LR = 1e-3


# Stats Paths
LANE_MEANS = "stats/lanes/lane_means.npy"
LANE_STDS = "stats/lanes/lane_stds.npy"

AGENT_MEANS = "stats/agents/agent_means.npy"
AGENT_STDS = "stats/agents/agent_stds.npy"

OBJ_MEANS = "stats/objects/obj_means.npy"
OBJ_STDS = "stats/objects/obj_stds.npy"

GT_MEANS = "stats/gt/gt_means.npy"
GT_STDS = "stats/gt/gt_stds.npy"

# Padding params
OBJ_PAD_LEN = 67
LANE_PAD_LEN = 145



track_category_mapping = {
    0 : "TRACK_FRAGMENT",
    1 : "UNSCORED_TARCK",
    2 : "SCORED_TARCK",
    3 : "FOCAL_TARCK"
}

object_color_code = {
    'vehicle'           : "#ff1d00",
    'bus'               : "#e2e817",
    'pedestrian'        : "#40BF64",
    'motorcyclist'      : "#2dd294",
    'riderless_bicycle' : "#1549ea",
    'background'        : "#112222",
    'static'            : "#112222",
    'construction'      : "#112222",
    'unknown'           : "#112222",
}

map_object_type = {
    'vehicle'           : 0,
    'bus'               : 1,
    'bike'              : 2,
    'cyclist' 		 : 2,
    'pedestrian'        : 2,
    'motorcyclist'      : 3,
    'riderless_bicycle' : 4,
    'background'        : 5,
    'static'            : 6,
    'construction'      : 7,
    'unknown'           : 8,
}

OUT_ENC_DIM = 16
N_TRAJ = 6
OUT_DIM = 2 * N_TRAJ * N_FUTURE + N_TRAJ


# Architecture configs
AGENT_ENC = {
    'd_model' : 8,
    'n_heads' : 2,
    'hidden_dim' : 32,
    'hidden_nheads' : 4,
    'output_dim' : OUT_ENC_DIM
}

OBJ_ENC = {
    'd_model' : 11,
    'n_heads' : 1,
    'hidden_dim' : 32,
    'hidden_nheads' : 4,
    'output_dim' : OUT_ENC_DIM
}

LANE_ENC = {
    'd_model' : 9,
    'n_heads' : 3,
    'hidden_dim' : 32,
    'hidden_nheads' : 4,
    'output_dim' : OUT_ENC_DIM
}

GRAPH_AGENT_ENC = {
    'd_model' : 8,
    'n_heads' : 2,
    'd_hidden' : 32,
    'd_out' : OUT_ENC_DIM
}

GRAPH_OBJ_ENC = {
    'd_model' : 11,
    'n_heads' : 1,
    'd_hidden' : 32,
    'd_out' : OUT_ENC_DIM
}

GRAPH_LANE_ENC = {
    'd_model' : 9,
    'n_heads' : 3,
    'd_hidden' : 32,
    'd_out' : OUT_ENC_DIM
}


GLOBAL_ENC = {
    'in_dim' : 17,
    # 'n_heads' : 2,
    # 'hidden_dim' : 32,
    # 'hidden_nheads' : 2,
    'out_dim' : 64
}

GLOBAL_ENC_TRANS = {
    'd_model' : 17,
    'num_heads' : 1,
    'd_ff' : 32,
    'output_dim' : 64
}


DECODER = {
    'in_dim' : GLOBAL_ENC['out_dim'], # 64
    'hidden_dim' : 128, 
    'out_dim' : OUT_DIM
}


blk = Style.BRIGHT + Fore.BLACK
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
grn_bck = Back.GREEN
res = Style.RESET_ALL