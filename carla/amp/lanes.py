import sys
import os 
import glob
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    
    pass

import carla
from .VectorNet.Argoverse2.utils.data import pad_lane_vectors
from .amp_config import LANE_NEAR_DISTANCE

# configs
RESOLUTION=1
'''
Resolution :  1   -> Number of vectors :  18285
'''


def calc_direction(xyz):
  direction = []
  lastx, lasty, lastz = xyz[0]
  for d in xyz[1:]:
    x, y, z = d
    dir = (y - lasty) / (x - lastx + 1e-5)
    lastx, lasty, lastz = [x, y, z]
    direction.append(dir)

  direction.append(dir)
  return np.array(direction)


def _build_topology(_map, resolution=RESOLUTION):
    """
    This function retrieves topology from the server as a list of
    road segments as pairs of waypoint objects, and processes the
    topology into a list of dictionary objects with the following attributes

    - entry (carla.Waypoint): waypoint of entry point of road segment
    - entryxyz (tuple): (x,y,z) of entry point of road segment
    - exit (carla.Waypoint): waypoint of exit point of road segment
    - exitxyz (tuple): (x,y,z) of exit point of road segment
    - path (list of carla.Waypoint):  list of waypoints between entry to exit, separated by the resolution
    """
    topology = []
    # Retrieving waypoints to construct a detailed topology
    for segment in _map.get_topology():
        wp1, wp2 = segment[0], segment[1]
        l1, l2 = wp1.transform.location, wp2.transform.location
        # Rounding off to avoid floating point imprecision
        x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)

        wp1.transform.location, wp2.transform.location = l1, l2
        seg_dict = dict()
        seg_dict['entry'], seg_dict['exit'] = wp1, wp2
        seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
        seg_dict['path'] = []
        endloc = wp2.transform.location
        if wp1.transform.location.distance(endloc) > resolution:
            w = wp1.next(resolution)[0]
            point = [w.transform.location.x, w.transform.location.y, w.transform.location.z, w.is_intersection, 0]
            while w.transform.location.distance(endloc) > resolution:
                seg_dict['path'].append(point)
                next_ws = w.next(resolution)
                if len(next_ws) == 0:
                    break
                w = next_ws[0]
        else:
            next_wps = wp1.next(resolution)
            if len(next_wps) == 0:
                continue
            point = [next_wps[0].transform.location.x, 
                     next_wps[0].transform.location.y, 
                     next_wps[0].transform.location.z, 
                     next_wps[0].is_intersection, 
                     0
                     ]
            seg_dict['path'].append(point)

        if len(seg_dict['path']) < 2:
            continue
        topology.append(seg_dict)

    return topology


def vectorize_lane(lane : np.ndarray) -> np.ndarray:
    '''
    args : 
        lane : np.ndarray [n_points, 5]: lane points [x, y, z, intersection, lane_type]
    
    returns : 
        lane : np.ndarray [35, 9]: lane vectors [xs, ys, zs, xe, ye, ze, intersection, direction, lane_type]
    '''
    # prepare return array
    ret = np.zeros((lane.shape[0]-1, 9))

    # set xs, ys, zs
    ret[:, 0:3] = lane[:-1, 0:3]
    
    # set xe, ye, ze
    ret[:, 3:6] = lane[1:, 0:3]
    
    # set intersection, direction
    ret[:, 6] = lane[:-1, 3]
    d = calc_direction(lane[:, :3])
    d = (d[:-1] + d[1:]) / 2
    ret[:, 7] = np.arctan(d)

    # set lane_type
    ret[:, 8] = lane[:-1, 4]

    ret = pad_lane_vectors(ret, n_vec=35)

    return ret



def get_lane_vectors (world : carla.World) -> np.ndarray:
    '''
    args : 
        world : carla.World : carla world object
    
    returns : 
        vectors : np.ndarray [n_lanes, 35, 9] : lane vectors [xs, ys, zs, xe, ye, ze, intersection, direction, lane_type]
    '''
    carla_map = world.get_map()
    topo = _build_topology(carla_map)
    vectors = np.empty((0, 35, 9))

    for segment in topo:
        lane = np.array(segment['path'])
        lane = vectorize_lane(lane)
        vectors = np.vstack((vectors, lane[None, ...]))

    return vectors


def get_nearby_lane_vectors (lane_vectors : np.ndarray, player_loc : np.ndarray, threshold : float = LANE_NEAR_DISTANCE) -> np.ndarray: 
    '''
    args : 
        lane_vectors : np.ndarray [n_lanes, 35, 11] : lane vectors [xs, ys, zs, xe, ye, ze, intersection, direction, lane_type, cx, cy]
        player_loc : np.ndarray [3] : player location [x, y, z]

    returns :
        nearby_lane_vectors : np.ndarray [n_lanes, 35, 11] : nearby lane vectors [xs, ys, zs, xe, ye, ze, intersection, direction, lane_type, cx, cy]
    '''
    distance = np.linalg.norm(lane_vectors[:, :, :3] - player_loc, axis=2).min(axis=1)
    nearby_lane_vectors = lane_vectors[distance < threshold]

    return nearby_lane_vectors


class LaneVectors:
    def __init__(self, world : carla.World) -> None:
        self.vectors = get_lane_vectors(world)

    def get_lane_vectors(self) -> np.ndarray:
        return self.vectors
    
if __name__=='__main__' : 
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()
    carla_map = world.get_map()

    vectors = get_lane_vectors(world)

    print(vectors.shape)

    player_loc = np.array([0, 0, 0])
    nearby_lane_vectors = get_nearby_lane_vectors(vectors, player_loc)
    print(nearby_lane_vectors.shape)