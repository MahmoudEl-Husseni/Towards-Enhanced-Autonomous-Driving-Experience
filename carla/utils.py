from __future__ import print_function
import math
import numpy as np
import socket
import json
from kalman_filter import *


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
from stm import STM


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_transform_matrix(transform):
    # Convert rotation to radians
    pitch = math.radians(transform.rotation.pitch)
    yaw = math.radians(transform.rotation.yaw)
    roll = math.radians(transform.rotation.roll)

    # Define rotation matrices
    rotation_matrix_pitch = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])

    rotation_matrix_yaw = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])

    rotation_matrix_roll = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])

    # Combine rotation matrices
    rotation_matrix = rotation_matrix_yaw @ rotation_matrix_pitch @ rotation_matrix_roll

    # Create translation matrix
    translation_matrix = np.array([
        [1, 0, 0, transform.location.x],
        [0, 1, 0, transform.location.y],
        [0, 0, 1, transform.location.z],
        [0, 0, 0, 1]
    ])

    # Combine rotation and translation
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = [transform.location.x, transform.location.y, transform.location.z]

    return transform_matrix


def get_inverse_matrix(transform) :

    matrix = get_transform_matrix(transform) 
    # Calculate inverse matrix
    rot = matrix[:3, :3]

    inv = np.eye(4)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ matrix[:3, 3]

    return inv


# data 
edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

def get_camera_intrinsic(camera_bp):
    # Get the camera intrinsic matrix
    # camera_bp: carla.sensor.CameraBlueprint object
    width = camera_bp.get_attribute('image_size_x').as_int()
    height = camera_bp.get_attribute('image_size_y').as_int()
    fov = camera_bp.get_attribute('fov').as_float()
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    

    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    

    return K
def get_inverse_intrinsic(camera_bp):
    # Get the camera intrinsic matrix
    # camera_bp: carla.sensor.CameraBlueprint object
    width = camera_bp.get_attribute('image_size_x').as_int()
    height = camera_bp.get_attribute('image_size_y').as_int()
    fov = camera_bp.get_attribute('fov').as_float()
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    

    K = np.identity(3)
    K[0, 0] = K[1, 1] = 1/focal
    K[0, 2] = - width / (2.0 * focal)
    K[1, 2] = - height / (2.0 * focal)
    

    return K


def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]


def transform_3d_image_to_world(xyz : np.ndarray, K_inv: np.ndarray, camera_2_world : np.ndarray) -> np.ndarray:

    '''
    transform 3d image coordinate to world coordinate
    xyz : np.ndarray : 3d image homogeneous coordinate 
    K_inv: numpu matrix of shape (3, 3) : inverse of camera intrinsic matrix
    camera_2_world : numpy matrix of shape (4, 4) : camera to world transformation matrix
    '''

    # Get the camera intrinsic matrix

    
    camera_points = np.dot(K_inv, xyz.T).T
    camera_points = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1))], axis=1)

    x, y, z = camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]

    camera_points = np.stack([y, -z, x], axis=1)
    
    camera_points = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1))], axis=1)

    world_points = np.dot(camera_2_world, camera_points.T).T
    world_points = world_points[:, :3]

    return world_points



def get_transform_matrix(transform):
    # Convert rotation to radians
    pitch = math.radians(transform.rotation.pitch)
    yaw = math.radians(transform.rotation.yaw)
    roll = math.radians(transform.rotation.roll)

    # Define rotation matrices
    rotation_matrix_pitch = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])

    rotation_matrix_yaw = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])

    rotation_matrix_roll = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])

    # Combine rotation matrices
    rotation_matrix = rotation_matrix_yaw @ rotation_matrix_pitch @ rotation_matrix_roll

    # Create translation matrix
    translation_matrix = np.array([
        [1, 0, 0, transform.location.x],
        [0, 1, 0, transform.location.y],
        [0, 0, 1, transform.location.z],
        [0, 0, 0, 1]
    ])

    # Combine rotation and translation
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = [transform.location.x, transform.location.y, transform.location.z]

    return transform_matrix


def get_inverse_matrix(transform) :

    matrix = get_transform_matrix(transform) 
    # Calculate inverse matrix
    rot = matrix[:3, :3]

    inv = np.eye(4)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ matrix[:3, 3]

    return inv


# data 
edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

def get_camera_intrinsic(camera_bp):
    # Get the camera intrinsic matrix
    # camera_bp: carla.sensor.CameraBlueprint object
    width = camera_bp.get_attribute('image_size_x').as_int()
    height = camera_bp.get_attribute('image_size_y').as_int()
    fov = camera_bp.get_attribute('fov').as_float()
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    

    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    

    return K
def get_inverse_intrinsic(camera_bp):
    # Get the camera intrinsic matrix
    # camera_bp: carla.sensor.CameraBlueprint object
    width = camera_bp.get_attribute('image_size_x').as_int()
    height = camera_bp.get_attribute('image_size_y').as_int()
    fov = camera_bp.get_attribute('fov').as_float()
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    

    K = np.identity(3)
    K[0, 0] = K[1, 1] = 1/focal
    K[0, 2] = - width / (2.0 * focal)
    K[1, 2] = - height / (2.0 * focal)
    

    return K


def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]


def transform_3d_image_to_world(xyz : np.ndarray, K_inv: np.ndarray, camera_2_world : np.ndarray) -> np.ndarray:

    '''
    transform 3d image coordinate to world coordinate
    xyz : np.ndarray : 3d image homogeneous coordinate 
    K_inv: numpu matrix of shape (3, 3) : inverse of camera intrinsic matrix
    camera_2_world : numpy matrix of shape (4, 4) : camera to world transformation matrix
    '''

    # Get the camera intrinsic matrix

    
    camera_points = np.dot(K_inv, xyz.T).T
    camera_points = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1))], axis=1)

    x, y, z = camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]

    camera_points = np.stack([y, -z, x], axis=1)
    
    camera_points = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1))], axis=1)

    world_points = np.dot(camera_2_world, camera_points.T).T
    world_points = world_points[:, :3]

    return world_points



class stm_steering_map  : 
    def __init__(self, in_mn, in_mx, out_mn, out_mx) : 
        self.in_mn = in_mn
        self.in_mx = in_mx
        self.out_mn = out_mn
        self.out_mx = out_mx
        
        
    
    def do_map (self, _input) :
        _input = max (self.in_mn , min(self.in_mx , _input))
        slope  = (self.out_mx - self.out_mn) / ( self.in_mx - self.in_mn)
        
        offset = self.out_mn - slope * self.in_mn 
        return (slope * _input + offset) 
    