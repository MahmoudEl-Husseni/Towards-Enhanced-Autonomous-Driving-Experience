# ------------------------------------------
# This file is part of argo rasterize.
# File: config.py

# Autor: Mahmoud ElHusseni
# Created on 2024/02/01.
# Github: https://github.com/MahmoudEl-Husseni
# Email: mahmoud.a.elhusseni@gmail.com
# ------------------------------------------

# PATHS
RAW_DIR = '/main/raw'
OUT_DIR = '/main/rasterize/render'
TRAIN_RAW_DIR = RAW_DIR + '/train'
VAL_RAW_DIR = RAW_DIR + '/val'
TEST_RAW_DIR = RAW_DIR + '/test'
TRAIN_OUT_DIR = OUT_DIR + '/train'
VAL_OUT_DIR = OUT_DIR + '/val'
TEST_OUT_DIR = OUT_DIR + '/test'

# RASTER PARAMS
RADIUS_OFFSET = 1.5
N_PAST = 60
raster_size = [224, 224]

# MAP PARAMS
road_colors = ['#37dd9c',
                '#c49665',
                '#8c278a',
                '#3d342c',
                '#b0d14f',
                '#aefcc6',
                '#99538f',
                '#e8f2d6',
                '#0bef93',
                '#6a86a8',
                '#f5e940']

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

AGENT_COLOR = '#000000'
OBJ_COLOR = '#ff0000'
