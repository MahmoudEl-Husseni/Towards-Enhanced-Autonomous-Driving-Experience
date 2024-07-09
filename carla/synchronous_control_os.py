#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import math
import random

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





def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
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

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter("model3")),
            start_pose)
        actor_list.append(vehicle)
        # vehicle.set_simulate_physics(False)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)

        lidar = world.spawn_actor(
            blueprint_library.find('sensor.lidar.ray_cast'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(lidar)

        depth = world.spawn_actor(
            blueprint_library.find('sensor.camera.depth'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(depth)


        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.033')
        imu = world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(imu)

        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', '0.033')
        gnss = world.spawn_actor(
            gnss_bp,
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(gnss)

        start = time.time()
        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, lidar, depth , imu , gnss, fps=30) as sync_mode:
            while True:
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg, lidar_raw, depht_raw , imu_raw  , gnss_raw  = sync_mode.tick(timeout=2.0)
                
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb)
                draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()
                pygame.event.pump()
                if car_control(vehicle):
                    return
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
