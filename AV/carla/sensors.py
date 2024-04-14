import os
import sys
import glob
import cv2
import numpy as np
import weakref
import math 
import pygame

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    
    pass

import carla
from carla import ColorConverter as cc

# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- LidarSensor -------------------------------------------------------------
# ==============================================================================


class LidarSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.lidar_range = 200

        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=0, z=3), carla.Rotation(pitch=0)), Attachment.Rigid),

            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid), 
            (carla.Transform(carla.Location(x=-10, z=20.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm)
            ]
        self.transform_index = 1
        
        self.sensor_data = ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}]
        
        world = self._parent.get_world()

        bp_library = world.get_blueprint_library()
        
        bp = bp_library.find(self.sensor_data[0])
        # self.lidar_range = 100
        bp.set_attribute('upper_fov', str(30.0))
        bp.set_attribute('lower_fov', str(-25.0))
        bp.set_attribute('channels', str(64))
        bp.set_attribute('range', str(self.lidar_range))
        bp.set_attribute('points_per_second', str(1000000))
        bp.set_attribute('sensor_tick', str(0))
        bp.set_attribute('rotation_frequency', str(50))
        
        self.sensor_data.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(notify=False, force_respawn=True)
        print(self.sensor.get_transform().get_matrix())

    def set_sensor(self, notify=True, force_respawn=False):
        needs_respawn = force_respawn or (self.sensor_data[2] != self.sensor_data[2])
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None

            self.sensor = self._parent.get_world().spawn_actor(
                self.sensor_data[-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: LidarSensor._parse_image(weak_self, image))

        if notify:
            self.hud.notification(self.sensor_data[2])

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod 
    def lidar_converter(lidar_data: np.ndarray) -> np.ndarray:
        """
        Convert lidar data to colors

        """
        
        distance = lidar_data[:, 2]

        if len(distance) == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        # color map viridis
        color = np.clip(distance / 20.0, 0.0, 1.0)
        color = (color * 255).astype(np.uint8)
        color = cv2.applyColorMap(color, cv2.COLORMAP_VIRIDIS)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        
        return color.astype(np.uint8)[:, 0]
    
    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        z_mean = np.mean(points[:, 2])
        points = points[points[:, 2] <= z_mean]
        
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
        lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        convex_hull = True
        if convex_hull: 
            color = self.lidar_converter(points)
            lidar_img[tuple(lidar_data.T)] = color

            kernel = np.ones((5, 5), np.uint8) 
            lidar_img = cv2.morphologyEx(lidar_img, cv2.MORPH_CLOSE, kernel, iterations=2)

        else :
            # calculate lidar colors based on elevation
            color = self.lidar_converter(points)
            lidar_img[tuple(lidar_data.T)] = color

        self.surface = pygame.surfarray.make_surface(lidar_img)

        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=0, z=3), carla.Rotation(pitch=0)), Attachment.Rigid),

            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid), 
            (carla.Transform(carla.Location(x=-10, z=20.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm)
            ]
        self.transform_index = 1
        self.sensor_data = ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}]
        
        world = self._parent.get_world()

        bp_library = world.get_blueprint_library()
        bp = bp_library.find(self.sensor_data[0])
        bp.set_attribute('image_size_x', str(hud.dim[0]))
        bp.set_attribute('image_size_y', str(hud.dim[1]))
        if bp.has_attribute('gamma'):
            bp.set_attribute('gamma', str(gamma_correction))
        for attr_name, attr_value in self.sensor_data[3].items():
            bp.set_attribute(attr_name, attr_value)

        self.sensor_data.append(bp)


    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(notify=False, force_respawn=True)

    def set_sensor(self, notify=True, force_respawn=False):
        needs_respawn = (force_respawn or (self.sensor_data[2] != self.sensor_data[2]))

        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None

            self.sensor = self._parent.get_world().spawn_actor(
                self.sensor_data[-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1]
                )
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensor_data[2])


    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    
    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        image.convert(self.sensor_data[1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]



        # TODO: Object Detection



        # TODO: Traffic Sign Detection


        # TODO: Lane Lines Detection



        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

