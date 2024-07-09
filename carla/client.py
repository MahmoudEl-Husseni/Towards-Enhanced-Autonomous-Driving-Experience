#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import os
import sys
import glob
from queue import Queue

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append('carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append('/main/CARLA_0.9.15/PythonAPI/amp')
sys.path.append('/main/CARLA_0.9.15/PythonAPI/examples2/amp/VectorNet/Argoverse2/')


from config import *
from sensors import *
from mapping import HDMap
from pygame_client import *
from kalman_filter import *



from stm import STM
# from amp import AMP

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import re
import carla
import pygame
import random
import logging
import argparse


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)


        self.record = [False, ]
        self.hud = hud
        self.player = None

        self.imu_sensor = None
        self.gnss_sensor = None
        self.lidar_sensor = None

        self.ss_camera = None
        self.depth_camera = None
        self.camera_manager = None
        self.bev = None

        self.ego_q = None
        self.depth_q = None
        self.lanes_q = None
        self.bboxes_q = None
        self.da_q = Queue(maxsize = DA_MAX_QUEUE_SIZE)




        self.hdmap = HDMap([self.ego_q, self.lanes_q, self.bboxes_q, self.da_q])
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma

        self.restart()

        if VERSION >= 0.9 : 
            # Localization Setup
            self.imu_std_dev_a     = 0.1
            self.gnss_std_dev_geo  = 7e-5
            self.imu_std_dev_g     = 0.001
            gnss_var    = 30
            imu_var_g   = 0.01
            imu_var_a   = 0.05
            map_geo = self.world.get_map().transform_to_geolocation(carla.Location(0,0,0))
            self.geo_centre_lat = kalman_filter.deg_to_rad(map_geo.latitude) 
            self.geo_centre_lon = kalman_filter.deg_to_rad(map_geo.longitude)
            self.geo_centre_alt = map_geo.altitude

            init_vel    = self.player.get_velocity()
            init_loc    = self.player.get_location()
            init_rot    = self.player.get_transform().rotation
            timestamp   = self.world.get_snapshot().timestamp.elapsed_seconds
            init_state  = np.asarray([init_loc.x, init_loc.y, init_rot.yaw * np.pi /180, init_vel.x, init_vel.y]).reshape(5,1)
            self.kal_obj = kalman_filter(init_state, 
                                        timestamp, 
                                        imu_var_a, 
                                        imu_var_g, 
                                        gnss_var, 
                                        self.geo_centre_lon,
                                        self.geo_centre_lat,
                                        self.geo_centre_alt)

            self.gnss_sensor = GnssSensor(self.player  , std_dev = self.gnss_std_dev_geo , kal_obj = self.kal_obj)
            self.imu_sensor = IMUSensor(self.player  , accel_std_dev= self.imu_std_dev_a ,  gyro_std_dev= self.imu_std_dev_g , kal_obj = self.kal_obj)

        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0


        self.stm_gear =""
        self.autoP = 0
        self.leftBlink=0
        self.rightBlink=0
        self.rightsignal=1
        self.leftsignal=1
        self.handBrake=0
        self.throttleCan=0.0
        self.BrakeCan=0.0
        self.steering_stm=0.0




    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        # Get a random blueprint.
        blueprint = self.world.get_blueprint_library().filter("model3")[0]
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else : 
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        
        # Set up the sensors.

        self.camera_manager = CameraManager(self.player, 
                                            self.hud, 
                                            self._gamma, 
                                            self.lanes_q, 
                                            self.bboxes_q, 
                                            self, 
                                            self.record
                                            )
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(notify=False, force_respawn=True)

        if VERSION >= 0.9 : 
            self.bev = CameraManager(self.player, 
                                    self.hud, 
                                    self._gamma, 
                                    self.lanes_q, 
                                    self.bboxes_q, 
                                    self, 
                                    self.record, 
                                    transform_index=-1)
            self.bev.transform_index = -1
            self.bev.set_sensor(notify=False, force_respawn=True)

            self.depth_camera = DepthCamera(self.player, 
                                            self.hud, 
                                            self.depth_q, 
                                            self.record, 
                                            self
                                            )
            self.depth_camera.transform_index = cam_pos_index
            self.depth_camera.set_sensor(notify=False, force_respawn=True)

            self.ss_camera = SemanticSegmentationCamera (self.player, 
                                                        self.hud, 
                                                        self.record,
                                                        self
                                                        )
            
            self.ss_camera.transform_index = cam_pos_index
            self.ss_camera.set_sensor(notify=False, force_respawn=True)



            self.lidar_sensor = LidarSensor(self.player, 
                                            self.hud, 
                                            self.da_q, 
                                            self.record
                                            )
            self.lidar_sensor.transform_index = cam_pos_index
            self.lidar_sensor.set_sensor(notify=False, force_respawn=True)


            self.views = [self.camera_manager, 
                          self.bev, 
                          self.lidar_sensor, 
                          self.depth_camera, 
                          self.ss_camera, 
                        #   self.hdmap
                          ] # camera - lidar - map
        else : 
            self.views = [
                self.camera_manager, 
                # self.hdmap
                ] # camera - map

        self.view = 0

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])


    def tick(self, clock):
        self.hud.tick(self, clock)
    

    def world_generate_map (self) : 
        if self.ego_q is not None and self.camera_manager is not None and self.camera_manager is not None and self.lidar_sensor is not None :
            return self.hdmap.generate_map (self.ego_q, 
                                            self.camera_manager.bboxes_q, 
                                            self.camera_manager.lanes_q, 
                                            self.lidar_sensor.da_q, 
                                            self.camera_manager.camera_2_world_map
                                            )

    def render(self, display):
        if self.views[self.view] is not None: 
            if self.view==len(self.views)-1: 
                self.world_generate_map()
                
            self.views[self.view].render(display)
            self.hud.render(display)

    def next_view(self) : 
        self.view = (self.view + 1) % len(self.views)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager,
            self.bev,
            self.depth_camera,
            self.ss_camera, 
            self.lidar_sensor,
            self.gnss_sensor,
            self.imu_sensor,
            ]
        
        for actor in actors:
            if actor is not None and actor.sensor is not None:
                actor.sensor.destroy()

        if self.player is not None:
            self.player.destroy()

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        # # Set auto pilot speed
        # tm = client.get_trafficmanager(8000)
        # tm_port = tm.get_port()
        # tm.global_percentage_speed_difference(70)
        
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        
        world = client.get_world() 

        world = World(world, hud, args)

        print("Carla Connected")        
        # amp = AMP(world.world, world.player, object_types = None)

        # stm = STM()
        # print("STM Initialized..")
        # controller = Controller(stm, world, client)
        controller = KeyboardControl(world, args.autopilot)

        
        clock = pygame.time.Clock()
        i=0
        while True:
            i+=1
            clock.tick_busy_loop(60)


            if isinstance (controller, Controller) :
                if controller.parse_events(client, world, clock) : 
                    return
            elif isinstance (controller, KeyboardControl) : 
                if controller.parse_events(client, world, clock) : 
                    return 


            world.tick(clock)

            # batch = amp.get_amp_inputs()
            # # amp batch
            # if i%100 == 0 :  
            #     if len(batch) > 0 : 
            #         outputs = amp.get_amp_outputs(batch)

            #         for it, out in enumerate(outputs) : 
            #             # Draw only for vehicles
            #             object_vectors = amp.objects_vectors
            #             if object_vectors[it][0][-1] not in [0, 1, 3] : 
            #                 continue

            #             for i in range(len(out)-1) : 
            #                 sx, sy = out[i]
            #                 ex, ey = out[i+1]
            #                 start = carla.Location(x=float(sx), y=float(sy), z=1.0)
            #                 end = carla.Location(x=float(ex), y=float(ey), z=1.0)
            #                 world.world.debug.draw_line( start, end, thickness=0.1,  color=carla.Color(r=150, g=0, b=0), life_time=3.5)




            # get ego vehicle location
            # ego_vehicle = world.player.get_transform()
            # ego_vehicle = [ego_vehicle.location.x, ego_vehicle.location.y, ego_vehicle.rotation.yaw]

            # if world.ego_q is None: 
            #     world.ego_q = Queue(maxsize = EGO_MAX_QUEUE_SIZE)
            #     world.hdmap.ego_q = world.ego_q

            # if not world.ego_q.full() : 
            #     world.ego_q.put(ego_vehicle)
            # else : 
            #     world.ego_q.get()
            #     world.ego_q.put(ego_vehicle)
            

            
            world.render(display)
            pygame.display.flip()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1920x1080',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()

