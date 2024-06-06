import os
import sys
import cv2
import math 
import glob
import time
import pygame
import weakref
import numpy as np
from PIL import Image

sys.path.append('/main/Towards-Enhanced-Autonomous-Driving-Experience/AV/visual perception')
sys.path.append('/home/asphalt/Agent Motion Prediction/Towards-Enhanced-Autonomous-Driving-Experience/AV/visual perception')
sys.path.append('/main/Towards-Enhanced-Autonomous-Driving-Experience/AV/YOLOP')  
sys.path.append('/home/asphalt/Agent Motion Prediction/Towards-Enhanced-Autonomous-Driving-Experience/AV/YOLOP')

from dummy.dummy_object_detection import get_bboxes
from dummy.dummy_lanes import get_dummy_lane_lines

# from ObjectDetection.object_detection import ObjectDetection

 
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    
    pass

import carla
from carla import ColorConverter as cc

from utils import *
from config import *
from lidar_to_camera import lidar_2_world_t, world_2_image_t, VID_RANGE, VIRIDIS


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor, std_dev=3e-5, kal_obj = None ):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        # For timing
        self.kal_obj = kal_obj
        self.gnss_to_xyz = kal_obj.gnss_to_xyz
        self.gnss_time = 0
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        bp.set_attribute('sensor_tick', '0.033')
        bp.set_attribute('noise_lat_stddev', str(std_dev))
        bp.set_attribute('noise_lon_stddev', str(std_dev))
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
        self.gnss_time = event.timestamp
        self.lat = event.latitude
        self.lon = event.longitude
        last_x , last_y  = self.kal_obj.state[0][0], self.kal_obj.state[1][0]  
        height = 10
        last_locathon = carla.Location(x=last_x, y=last_y, z=height)
        x, y, z = self.gnss_to_xyz(event) 
        # self.debug.draw_point(carla.Location(x=x, y=y, z=height), size=0.05 , color=carla.Color(r=0, g=255, b=0) ,life_time=0.1)
        self.kal_obj.measure(np.asarray([x, y]).reshape(2,1), self.gnss_time)
        x, y = self.kal_obj.state[0][0], self.kal_obj.state[1][0]
        current_locathon = carla.Location(x=x, y=y, z=height)
        # self.debug.draw_line( last_locathon, current_locathon,thickness=0.1,  color=carla.Color(r=0, g=150, b=0), life_time=20)


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor, accel_std_dev = 0, gyro_std_dev = 0 , kal_obj = None):
        self.sensor = None
        # For timing
        self.kal_obj = kal_obj
        self.imu_time = 0
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.imu')
        bp.set_attribute('sensor_tick', '0.033')
        # bp.set_attribute('noise_gyro_stddev_y',  str(gyro_std_dev))
        # bp.set_attribute('noise_gyro_stddev_x',  str(gyro_std_dev))
        # bp.set_attribute('noise_gyro_stddev_z',  str(gyro_std_dev))
        # bp.set_attribute('noise_accel_stddev_y', str(accel_std_dev))
        # bp.set_attribute('noise_accel_stddev_x', str(accel_std_dev))
        # bp.set_attribute('noise_accel_stddev_z', str(accel_std_dev))
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
        self.imu_time = sensor_data.timestamp
        
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
        accel_x = self.accelerometer[0]
        accel_y = self.accelerometer[1]
        yaw_vel = self.gyroscope[2]
        last_x , last_y = self.kal_obj.state[0][0], self.kal_obj.state[1][0] 
        last_locathon = carla.Location(x=last_x, y=last_y, z=10)
        self.kal_obj.update(np.asarray([accel_x, accel_y, yaw_vel]).reshape(3,1), self.imu_time)
        x, y= self.kal_obj.state[0][0], self.kal_obj.state[1][0]
        current_locathon = carla.Location(x=x, y=y, z=10)
        # self.debug.draw_line( last_locathon, current_locathon, thickness=0.1,  color=carla.Color(r=0, g=150, b=0), life_time=20)

# ==============================================================================
# -- LidarSensor -------------------------------------------------------------
# ==============================================================================

class LidarSensor(object) :
    def __init__(self, 
                 parent_actor, 
                 hud, 
                 da_q, 
                 record,
                 mx_da_qsize=DA_MAX_QUEUE_SIZE):
        
        self.recording = record
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.da_q = da_q
        self.da_q_size = mx_da_qsize
        self.lidar_range = 20


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


        
        self.lidar_range = 20
        bp.set_attribute('upper_fov', str(30.0))
        bp.set_attribute('lower_fov', str(-25.0))
        bp.set_attribute('channels', str(64))
        bp.set_attribute('range', str(self.lidar_range))
        bp.set_attribute('points_per_second', str(1000000))
        bp.set_attribute('sensor_tick', str(0.033))
        bp.set_attribute('rotation_frequency', str(50))

        if VERSION>0.9:
            bp.set_attribute('dropoff_general_rate', '0.0')
            bp.set_attribute('dropoff_intensity_limit', '1.0')
            bp.set_attribute('dropoff_zero_intensity', '0.0')
        
        self.sensor_data.append(bp)
        self.index = None

        self.lidar_2_world = np.eye(4)

        self.world_points = np.empty((4, 0))
        self.intensity = np.empty((0, 1))

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(notify=False, force_respawn=True)

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
            
            if VERSION<=0.9:
                self.lidar_2_world = get_transform_matrix(self.sensor.get_transform())
            else :
                self.lidar_2_world = self.sensor.get_transform().get_matrix()
            
            self.lidar_2_world = np.array(self.lidar_2_world).reshape(4, 4)

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: LidarSensor._parse_image(weak_self, image))

        if notify:
            self.hud.notification(self.sensor_data[2])

    def toggle_recording(self):
        self.recording[0] = not self.recording[0]
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
        start_time = time.time()
        self = weak_self()
        if not self:
            return

        self.world_points, self.intensity = lidar_2_world_t(image, self.lidar_2_world)
        
        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / LIDAR_CHANNELS), LIDAR_CHANNELS))
        
        z_mean = np.mean(points[:, 2])
        points = points[points[:, 2] <= z_mean]
        if VERSION<=0.9 : 
            points = np.vstack((points.T, np.ones(points.shape[0]))).T
            self.lidar_2_world = get_transform_matrix(self.sensor.get_transform())
        else :
            self.lidar_2_world = self.sensor.get_transform().get_matrix()
        

        _points = self.lidar_2_world @ points.T
        
        if self.da_q.qsize() == self.da_q_size:
            self.da_q.get()
        self.da_q.put(_points.T)
        
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
        lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        convex_hull = False
        if convex_hull : 
            color = self.lidar_converter(points)
            lidar_img[tuple(lidar_data.T)] = color

            kernel = np.ones((5, 5), np.uint8) 
            lidar_img = cv2.morphologyEx(lidar_img, cv2.MORPH_CLOSE, kernel, iterations=2)

        else :
            # calculate lidar colors based on elevation
            color = self.lidar_converter(points)
            lidar_img[tuple(lidar_data.T)] = color

        self.surface = pygame.surfarray.make_surface(lidar_img)
        lidar_img = Image.fromarray(lidar_img)

        if self.recording[0]:
            if not os.path.exists("_out/lidar") : 
                os.mkdir("_out/lidar")

            path = f"_out/lidar/{image.frame}.png"
            lidar_img.save(path)
        if DEBUG_TIME : 
            print(f"> Time taken for LiDAR processing = {time.time() - start_time} seconds.")

# ==============================================================================
# -- DepthCamera -------------------------------------------------------------
# ==============================================================================

class DepthCamera(object):
    def __init__(self, 
                 parent_actor, 
                 hud, 
                 depth_q, 
                 record, 
                 world
                 ):
        self.world = world
        self.recording = record
        self.sensor = None  
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        
        self.depth_q = depth_q


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
        
        # self.sensor_data = ['sensor.camera.depth', cc.Depth, 'Camera Depth', {}]
        
        self.sensor_data = ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}]

        
        world = self._parent.get_world()

        bp_library = world.get_blueprint_library()
        
        bp = bp_library.find(self.sensor_data[0])
        bp.set_attribute('image_size_x', str(hud.dim[0]))
        bp.set_attribute('image_size_y', str(hud.dim[1]))
        bp.set_attribute('fov', str(110))
        bp.set_attribute('sensor_tick', str(0.033))
        if bp.has_attribute('gamma'):
            bp.set_attribute('gamma', str(2.2))
        for attr_name, attr_value in self.sensor_data[3].items():
            bp.set_attribute(attr_name, attr_value)

        self.sensor_data.append(bp)
        self.index = None

        self.camera_2_world = np.eye(4)

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(notify=False, force_respawn=True)
    
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
            
            if VERSION<=0.9:
                self.camera_2_world = get_transform_matrix(self.sensor.get_transform())
            else :
                self.camera_2_world = self.sensor.get_transform().get_matrix()
            
            self.camera_2_world = np.array(self.camera_2_world).reshape(4, 4)

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: DepthCamera._parse_image(weak_self, image))

        if notify:
            self.hud.notification(self.sensor_data[2])

    def toggle_recording(self):
        self.recording[0] = not self.recording[0]
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
    
    @staticmethod
    def _parse_image(weak_self, image):
        start_time = time.time()
        self = weak_self()
        if not self:
            return

        if VERSION<=0.9:
            self.camera_2_world = get_transform_matrix(self.sensor.get_transform())
        else : 
            self.camera_2_world = self.sensor.get_transform().get_matrix()

        self.camera_2_world = np.array(self.camera_2_world).reshape(4, 4)

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        # to_display = array

        R, G, B = array[:, :, 0], array[:, :, 1], array[:, :, 2]
        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized
        array = in_meters

        array = array[:, :, None]
        array = np.concatenate((array, array, array), axis=2)
        self.depth_q = array


        self.surface = pygame.surfarray.make_surface(array)

        if self.recording[0]:
            path = f"_out/depth/{image.frame}.png"
            image.save_to_disk(path)


        # generate map
        generated_map = self.world.world_generate_map ()
        if self.recording[0]:
            path = f"_out/map/{image.frame}.png"
            image = Image.fromarray(generated_map)

            if not os.path.exists('_out/map') :
                os.mkdir('_out/map')

            image.save(path)

        if DEBUG_TIME: 
            print(f"> Time taken for Depth Camera processing = {time.time() - start_time} seconds.")


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, 
                 parent_actor, 
                 hud, 
                 gamma_correction, 
                 lanes_q, 
                 bboxes_q, 
                 world, 
                 record, 
                 transform_index=0
                 ):
        self.recording = record
        # YOLOP Configurations
        # from yolop_tools.demo import Detector
        # from lib.config import cfg

        class Opt : 
            def __init__(self) : 
                self.device='0'
                self.save_dir=''
                self.weights='/main/Towards-Enhanced-Autonomous-Driving-Experience/AV/YOLOP/weights/End-to-end.pth'
                self.img_size=640
                self.conf_thres=0.25
                self.iou_thres=0.45

        opt = Opt()
        # self.yolop_detector = Detector(cfg, opt)

        # For Yolo Object Detection
        # self.detector = ObjectDetection()

        self.hud = hud
        self.world = world
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        
        self.lanes_q = get_dummy_lane_lines(self.world.world)
        self.bboxes_q = bboxes_q

        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=0, z=3), carla.Rotation(pitch=0)), Attachment.Rigid),

            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid), 
            (carla.Transform(carla.Location(x=-10, z=20.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm), 
            (carla.Transform(carla.Location(x=-10, z=30.0), carla.Rotation(pitch=4.0)), Attachment.SpringArm)
            ]
        
        self.transform_index = transform_index
        self.sensor_data = ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}]
        
        world = self._parent.get_world()

        bp_library = world.get_blueprint_library()
        bp = bp_library.find(self.sensor_data[0])
        bp.set_attribute('image_size_x', str(hud.dim[0]))
        bp.set_attribute('image_size_y', str(hud.dim[1]))
        bp.set_attribute('fov', str(110))
        bp.set_attribute('sensor_tick', str(0.033))
        if bp.has_attribute('gamma'):
            bp.set_attribute('gamma', str(gamma_correction))
        for attr_name, attr_value in self.sensor_data[3].items():
            bp.set_attribute(attr_name, attr_value)

        self.sensor_data.append(bp)

        
        self.K = get_camera_intrinsic(bp)
        self.K_inv = np.linalg.inv(self.K)
        self.camera_2_world_map = np.ones((hud.dim[1], hud.dim[0], 1), dtype=np.int) * -1
        # self.sensor_2_agent = get_inverse_matrix(self._parent.get_transform())



    def toggle_camera(self):
        if self.transform_index == -1 : 
            return 
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
            
            
            self.K = get_camera_intrinsic(self.sensor_data[-1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensor_data[2])


    def toggle_recording(self):
        self.recording[0] = not self.recording[0]
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    
    @staticmethod
    def _parse_image(weak_self, image):
        start_time = time.time()
        self = weak_self()
        if not self:
            return

        if VERSION<=0.9:
            self.camera_2_world = get_transform_matrix(self.sensor.get_transform())
        else : 
            self.camera_2_world = self.sensor.get_transform().get_matrix()

        self.camera_2_world = np.array(self.camera_2_world).reshape(4, 4)

        image.convert(self.sensor_data[1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.transform_index==-1: 
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            if self.recording[0]:
                path = f"_out/bev/{image.frame}.png"
                image.save_to_disk(path)

            return 


        # Depth from depth camera
        depth_f = False
        if depth_f:
            depth = self.world.depth_camera.depth_q
            array = cv2.addWeighted(array.astype('uint8'), 0.5, depth.astype('uint8'), 1, 0)
        else : 
            depth = np.zeros_like(array)

        # h, w = array.shape[:2]
        # xy = np.meshgrid(np.arange(w), np.arange(h))
        # xy = np.stack(xy, axis=-1)
        # xy = xy.transpose(1, 0, 2).reshape(-1, 2).astype('float32')
        # z = depth.mean(-1).reshape(-1)
        # xy *= z[:, None]
        # xyz = np.concatenate([xy, z[:, None]], axis=1)
        
        # xyz = transform_3d_image_to_world(xyz, self.K_inv, self.camera_2_world)
        
        
        # TODO: Object Detection
        # YOLOP Detection
        yolop=False
        if yolop:
            pass
            # lanes = get_dummy_lane_lines(self.world.world)
            # self.lanes_q = lanes
        

        object_detection_f = False
        if object_detection_f : 
            st_time = time.time()
            # self.detector.detect(img=array , save_path=f'out/output_image_{image.frame}.jpg', save=False)
            print(f'Time in detection = {time.time() - st_time} seconds.')


        dummy_object_detection_f = False
        if dummy_object_detection_f: 
            from dummy_object_detection import get_bboxes
            bboxes = get_bboxes(self._parent.get_world(), self._parent)

            self.bboxes_q = bboxes


        # TODO: Traffic Sign Detection


        # TODO: Lane Lines Detection


        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording[0]:
            path = f"_out/rgb/{image.frame}.png"
            image.save_to_disk(path)



        # if DEBUG_TIME : 
        #     print(f"> Time taken for Camera processing = {time.time() - start_time} seconds.")



# ==============================================================================
# -- SemanticSegmentationCamera -------------------------------------------------------------
# ==============================================================================

class SemanticSegmentationCamera(object):
    def __init__(self, parent_actor, hud, record, world=None):
        self.sensor = None  
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = record
        self.world=world
        

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
        
        self.sensor_data = ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}]
        
        world = self._parent.get_world()

        bp_library = world.get_blueprint_library()
        
        bp = bp_library.find(self.sensor_data[0])
        bp.set_attribute('image_size_x', str(hud.dim[0]))
        bp.set_attribute('image_size_y', str(hud.dim[1]))
        bp.set_attribute('fov', str(110))
        bp.set_attribute('sensor_tick', str(0.033))

        for attr_name, attr_value in self.sensor_data[3].items():
            bp.set_attribute(attr_name, attr_value)

        self.sensor_data.append(bp)
        self.index = None

        self.camera_2_world = np.eye(4)

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(notify=False, force_respawn=True)
    
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
            
            if VERSION<=0.9:
                self.camera_2_world = get_transform_matrix(self.sensor.get_transform())
            else :
                self.camera_2_world = self.sensor.get_transform().get_matrix()
            
            self.camera_2_world = np.array(self.camera_2_world).reshape(4, 4)

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: SemanticSegmentationCamera._parse_image(weak_self, image))

        if notify:
            self.hud.notification(self.sensor_data[2])

    def toggle_recording(self):
        self.recording[0] = not self.recording[0]
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        start_time = time.time()
        self = weak_self()
        if not self:
            return

        image.convert(self.sensor_data[1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording[0]:
            path = f"_out/ss/{image.frame}.png"
            image.save_to_disk(path)

        if DEBUG_TIME : 
            print(f"> Time taken for Semantic Segmentation processing = {time.time() - start_time} seconds.")
