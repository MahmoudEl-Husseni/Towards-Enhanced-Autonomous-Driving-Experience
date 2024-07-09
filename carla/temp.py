#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import time
from agents.navigation.basic_agent import BasicAgent
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from utils import *
from config import *
from sensors import DepthCamera


from mapping import HDMap

import cv2
import math
import carla
import random
from kalman_filter import *

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w


except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


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

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

def Gnss_callback( kal_obj , event):
    gnss_time = event.timestamp
    height = 10
    x, y, z = kal_obj.gnss_to_xyz(event) 
    kal_obj.measure(np.asarray([x, y]).reshape(2,1), gnss_time)
    x, y = kal_obj.state[0][0], kal_obj.state[1][0]
    return carla.Location(x=x, y=y, z=height)


def IMU_callback( kal_obj , sensor_data):
    imu_time = sensor_data.timestamp
    limits = (-99.9, 99.9)
    accelerometer = (
        max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
        max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
        max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
    gyroscope = (
        max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
        max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
        max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
    accel_x = accelerometer[0]
    accel_y = accelerometer[1]
    yaw_vel = gyroscope[2]
    kal_obj.update(np.asarray([accel_x, accel_y, yaw_vel]).reshape(3,1), imu_time)
    return accelerometer , gyroscope





def draw_image(surface, array, blend=False):
    # array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    # array = np.reshape(array, (image.height, image.width, 4))
    # array = array[:, :, :3]
    # array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def car_control(car):
    """
    Applies control to main car based on pygame pressed keys.
    Will return True If ESCAPE is hit, otherwise False to end main loop.
    """
    keys = pygame.key.get_pressed()
    if keys[K_ESCAPE]:
        return True

    control = car.get_control()
    control.throttle = 0
    if keys[K_w]:
        control.throttle = 1
        control.reverse = False
    elif keys[K_s]:
        control.throttle = 1
        control.reverse = True
    if keys[K_a]:
        control.steer = max(-1., min(control.steer - 0.05, 0))
    elif keys[K_d]:
        control.steer = min(1., max(control.steer + 0.05, 0))
    else:
        control.steer = 0
    control.hand_brake = keys[K_SPACE]
    car.apply_control(control)
    return False



def car_control_with_stm(car , stm_data , current_lights ):

    control = car.get_control()
    control.throttle = stm_data.get('throttle', control.throttle)
    control.steer = stm_data.get('gear', control.steer)
    control.reverse = stm_data.get('gear', control.reverse)
    control.brake = stm_data.get('brake', control.brake)
    leftBlinker = stm_data.get('leftBlinker', current_lights & carla.VehicleLightState.LeftBlinker)
    rightBlinker = stm_data.get('rightBlinker', current_lights & carla.VehicleLightState.RightBlinker)
    if leftBlinker :
        current_lights ^= carla.VehicleLightState.LeftBlinker
    if rightBlinker :
        current_lights ^= carla.VehicleLightState.RightBlinker
    car.set_light_state(carla.VehicleLightState(current_lights))
    return control

def main():

    actor_list = []
    pygame.init()
    

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    autoPilot = False

    try:
        map = world.get_map()
        start_pose = random.choice(m.get_spawn_points())

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter("model3")),
            start_pose)
        
        current_lights = carla.VehicleLightState.NONE
        actor_list.append(vehicle)
        agent = BasicAgent(vehicle)

        # vehicle.set_simulate_physics(False)

        transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        # spawn sensors
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            transform,
            attach_to=vehicle)
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        camera_rgb.calibration = calibration
        actor_list.append(camera_rgb)

        lidar = world.spawn_actor(
            blueprint_library.find('sensor.lidar.ray_cast'),
            transform,
            attach_to=vehicle)
        actor_list.append(lidar)

        depth = world.spawn_actor(
            blueprint_library.find('sensor.camera.depth'),
            transform,
            attach_to=vehicle)
        actor_list.append(depth)


        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.033')
        imu = world.spawn_actor(
            imu_bp,
            transform,
            attach_to=vehicle)
        actor_list.append(imu)

        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', '0.033')
        gnss = world.spawn_actor(
            gnss_bp,
            transform,
            attach_to=vehicle)
        actor_list.append(gnss)



        # create objects 
        mapping = HDMap()

        # yolo object
        if OBJECT_DETECTION_YOLO : 
            object_detector = ObjectDetection(OBJECT_DETECTION_YOLO_WEIGHTS)
        

        init_vel    = vehicle.get_velocity()
        init_loc    = vehicle.get_location()
        init_rot    = vehicle.get_transform().rotation
        timestamp   = world.get_snapshot().timestamp.elapsed_seconds
        init_state  = np.asarray([init_loc.x, init_loc.y, init_rot.yaw * np.pi /180, init_vel.x, init_vel.y]).reshape(5,1)
        map_geo = world.get_map().transform_to_geolocation(carla.Location(0,0,0))
        geo_centre_lat = kalman_filter.deg_to_rad(map_geo.latitude) 
        geo_centre_lon = kalman_filter.deg_to_rad(map_geo.longitude)
        geo_centre_alt = map_geo.altitude
        gnss_var    = 30
        imu_var_g   = 0.01
        imu_var_a   = 0.05

        kal_obj = kalman_filter(init_state, 
                                    timestamp, 
                                    imu_var_a, 
                                    imu_var_g, 
                                    gnss_var, 
                                    geo_centre_lon,
                                    geo_centre_lat,
                                    geo_centre_alt)

        stm = STM()
        print("STM Initialized..")
        start = time.time()
        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, lidar, depth, imu, gnss, fps=30) as sync_mode:
            while True:
                if STM_CONTROL :
                    stm.send(
                        autoPilot,
                        1000 * vehicle.forward_speed , # m/s to km/h
                        vehicle.get_control().gear,
                        vehicle.get_control().throttle,
                        vehicle.get_control().steer,
                        vehicle.get_control().brake,
                        current_lights & carla.VehicleLightState.LeftBlinker, 
                        current_lights & carla.VehicleLightState.RightBlinker , 
                        False,   # warning we need to implement this
                        False  # alert we need to implement this
                    )
                clock.tick()
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, lidar_raw, depth_raw, imu_raw ,gnss_raw = sync_mode.tick(timeout=2.0)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                
                
                # process raw data 
                array = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image_rgb.height, image_rgb.width, 4))
                array = array[:, :, :3].astype('uint8')
                


                wr = weakref.ref(depth)
                depth_array = DepthCamera._parse_image(wr, depth_raw)


                # Perception modules here

                # object detection
                bboxes = object_detector.detect(array)
                # print (bboxes)
                COLOR = ( 255,0,0)
                for box in bboxes : 
                    x_min, y_min, x_max, y_max = box[1:].astype(int)
                    cv2.rectangle(array, (x_min, y_min), (x_max, y_max),  COLOR, 2)




                # Traffic sign detection



                # localization to get location
                _ , gyroscope = IMU_callback(kal_obj, imu_raw)
                location = Gnss_callback(kal_obj, gnss_raw)
                ego_x, ego_y = location.x, location.y
                ego_yaw = gyroscope[2]



                # # mapping
                camera_2_world = get_transform_matrix(camera_rgb.get_transform())
                mapping.update([ego_x, ego_y, ego_yaw], bboxes, depth_array, np.linalg.inv(camera_rgb.calibration), camera_2_world)
                isFront = mapping.front_nearby_objects()
                print (f'isFront : {sum(isFront)}')



                _array = mapping.generate_map()




                # Draw the display.
                draw_image(display, _array)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()
                # pygame.event.pump()
                # if car_control(vehicle):
                #     return
                if STM_CONTROL : 
                    stmData = stm.receive()
                    autoPilot = stmData.get('autoPilot', autoPilot)
                    if autoPilot :
                        destination = stmData.get('destination', None)
                        if destination is not None:
                            destination = carla.Location(destination[0], destination[1], destination[2])
                            agent.set_destination(destination)
                        agent.run_step()
                        car_control_with_stm(vehicle, stmData, current_lights)
                    else :
                        vehicle.apply_control(car_control_with_stm(vehicle, stmData, current_lights))
                else : 
                    control = agent.run_step(sum(isFront)>0)
                    print("control command: ", control)
                    vehicle.apply_control(control)

                
                print (1 / (time.time()-start ))
                start = time.time()

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
