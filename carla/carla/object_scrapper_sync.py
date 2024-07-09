#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append('/media/mahmoud/New Volume/faculty/level2/study/machine learning/Towards Enhanced Autonomous Vehicle/carla/amp')


import carla

import random
from amp.amp_config import carla_objects_bp


try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

from object_scrapper import ClientSideBoundingBoxes, VIEW_FOV, VIEW_HEIGHT, VIEW_WIDTH, project_3d_to_2d



class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self.objects = []
        for obj in carla_objects_bp.values():
            # for obj in ['vehicle.*', ]:
                
                _obj = []
                for i in obj:
                    tmp = self.world.get_actors().filter(i)
                    for j in tmp:
                        _obj.append(j)

                # print(_obj)
                self.objects.append(_obj)


    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout, camera_rgb):
        self.frame = self.world.tick()
        vehicles = self.world.get_actors().filter('vehicle.*')
        bboxes = []
        for i, obj in enumerate(self.objects) : 
            _bboxes = ClientSideBoundingBoxes.get_bounding_boxes(obj, camera_rgb)

            _bboxes = np.array(_bboxes).reshape(-1, 4)
            class_id = np.ones((_bboxes.shape[0], 1)) * i
            _bboxes = np.hstack([class_id, _bboxes])
            bboxes.append(_bboxes)
            
        bboxes = np.vstack(bboxes)
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        data.append(bboxes)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data




def main():
    actor_list = []


    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.*')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        

        camera_rgb = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        
        calibration = np.identity(3)
        calibration[0, 2] = 960 / 2.0
        calibration[1, 2] = 540 / 2.0
        calibration[0, 0] = calibration[1, 1] = 540 / (2.0 * np.tan(90 * np.pi / 360.0))
        camera_rgb.calibration = calibration


        actor_list.append(camera_rgb)


        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, fps=30) as sync_mode:
            i=0
            while True:
                print(i)
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, bboxes = sync_mode.tick(timeout=2.0, camera_rgb=camera_rgb)

                # Choose the next waypoint and update the car location.
                waypoint = random.choice(waypoint.next(1.5))
                vehicle.set_transform(waypoint.transform)

                # Draw Bounding Boxes
                print("Number of bounding boxes: ", len(bboxes))
                array = np.frombuffer(image_rgb.raw_data, dtype=np.uint8)
                array = array.reshape((image_rgb.height, image_rgb.width, 4))
                array = array[:, :, :3]
                array = array.astype(np.uint8)
                _array = ClientSideBoundingBoxes.draw_2d_bounding_boxes_on_image(array, bboxes)

                serial_number = str(i).zfill(5)
                cv2.imwrite(f'output/{serial_number}.png', _array)

                i+=1

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')