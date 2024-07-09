import os
import sys
import cv2
import glob
from queue import Queue

from pygame_client import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    
    pass

import carla



def get_dummy_lane_lines (world : carla.World) -> list:
    carla_map = world.get_map()
    road_segments = carla_map.get_topology ()
    lane_lines = []
    for lane in road_segments:
        start = lane[0].transform.location
        start = [start.x, start.y, start.z]

        end = lane[-1].transform.location
        end = [end.x, end.y, end.z]

        lane_lines.append([start, end])
    lane_lines = np.array(lane_lines)
    return lane_lines

if __name__=='__main__' : 
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    # Load the map
    world = client.get_world()
    carla_map = world.get_map()

    # Get the road segments
    road_segments = carla_map.get_topology()


    # ========================================================================================================
    # Extract lane lines from road segments
    lane_lines = []

    # GPT -> not working 
    for segment in road_segments:
        for lane_marking in segment:
            # Extract lane line points
            # print(dir(lane_marking))
            # print(lane_marking.transform.location)
            # print(50*'-')
            # print(lane_marking.get_landmarks(0.5))

            # lane_line_points = [lane_marking.transform.location for lane_marking in lane_marking]
            if len(lane_marking.get_landmarks(100, True)) == 0:
                continue
            print(lane_marking.get_landmarks(100, True))
            lane_line_points = [lane_marking.transform.location for lane_marking in lane_marking.get_landmarks(10)]
            lane_lines.append(lane_line_points)

    # Visualize the lane lines (for demonstration purposes)
    for lane_line_points in lane_lines:
        for i in range(len(lane_line_points) - 1):
            start_point = lane_line_points[i]
            end_point = lane_line_points[i + 1]
            # For visualization, you can draw lines using the start and end points
            print(f"Draw line from {start_point} to {end_point}")





    # for lane in road_segments : 
    #     print("Start: ",lane[0].transform.location)
    #     print("End: ",lane[-1].transform.location)

    lanes = get_dummy_lane_lines(world)
    print(lanes.shape)