#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import os
import sys
import cv2
import glob

sys.path.append('/main/Towards Enhanced Autonomous Vehicle/carla/amp')



def save_bboxes(bboxes, frame, output_dir):
    txt_filename = os.path.join(output_dir, f"{frame:06d}.txt")
    with open(txt_filename, 'w') as f:
        for bbox in bboxes:
            f.write(f"{frame} " + " ".join(map(str, bbox)) + '\n')


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a, K_p
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)


from amp.amp_config import carla_objects_bp

OBJECT_COLORS =[
    (255, 0, 0), # Red
    (0, 255, 0), # Green
    (0, 0, 255), # Blue

    (255, 255, 0), # Yellow
    (255, 0, 255), # Magenta
    (0, 255, 255), # Cyan

    (255, 255, 255), # White
    (0, 0, 0), # Black
    (128, 128, 128), # Gray
]

def project_3d_to_2d(bbox_3d):
    """
    Projects a 3D bounding box to a 2D bounding box on the XY plane.
    
    Args:
    bbox_3d: List of 8 tuples representing the 3D bounding box corners.
             Each tuple is (x, y, z).
    
    Returns:
    bbox_2d: Tuple containing (min_x, min_y, max_x, max_y) representing the 2D bounding box.
    """
    # Extract x, y coordinates from the 3D bounding box
    bbox_3d = np.transpose(bbox_3d, (1, 0))
    x_coords = bbox_3d[0]
    y_coords = bbox_3d[1]
    
    # Find the min and max coordinates
    min_x = x_coords.min()
    min_y = y_coords.min()
    max_x = x_coords.max()
    max_y = y_coords.max()
    
    return (min_x, min_y, max_x, max_y)

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        _2d_bounding_boxes = [project_3d_to_2d(bb) for bb in bounding_boxes]

        return _2d_bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        display.blit(bb_surface, (0, 0))


    def draw_2d_bounding_boxes(display, bounding_boxes):
        """
        Draws 2D bounding boxes on pygame display.
        
        Args:
        display: Pygame display surface.
        bounding_boxes: List of tuples containing (min_x, min_y, max_x, max_y) representing the 2D bounding boxes.
        """

        # Create a transparent surface
        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT), pygame.SRCALPHA)

        for bbox in bounding_boxes:
            class_id, min_x, min_y, max_x, max_y = bbox
            # Draw the rectangle on the surface
            COLOR = OBJECT_COLORS[int(class_id)]
            pygame.draw.rect(bb_surface, COLOR, pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y), 2)
        
        # Blit the bounding box surface onto the display
        display.blit(bb_surface, (0, 0))


    def draw_2d_bounding_boxes_on_image(image, bounding_boxes):
        """
        Draws 2D bounding boxes on pygame display.
        
        Args:
        image : np.array of shape (H, W, 3)
        bounding_boxes: List of tuples containing (min_x, min_y, max_x, max_y) representing the 2D bounding boxes.
        """

        # Create a transparent surface

        for bbox in bounding_boxes:
            class_id, min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
            # Draw the rectangle on the surface
            COLOR = OBJECT_COLORS[int(class_id)]
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), COLOR, 2)


        return image

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))

        # get object id 
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], 
        bbox[:, 1] / bbox[:, 2], 
        bbox[:, 2], ], axis=1)
        return camera_bbox



    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        try : 
            extent = vehicle.bounding_box.extent
            cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
            cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
            cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
            cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
            cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
            cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
            cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
            cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        finally : 
            return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.capture = True

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def control(self, car):
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
        # set autopilot 
        elif keys[K_p]:
            car.set_autopilot(True)
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def game_loop(self):
        """
        Main program loop.
        """

        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter('vehicle.*')

            objects = []

            for obj in carla_objects_bp.values():
            # for obj in ['vehicle.*', ]:
                
                _obj = []
                for i in obj:
                    tmp = self.world.get_actors().filter(i)
                    for j in tmp:
                        _obj.append(j)

                # print(_obj)
                objects.append(_obj)
            
            j=0
            while True:
                self.world.tick()

                self.capture = True
                pygame_clock.tick_busy_loop(10)

                self.render(self.display)
                for i, obj in enumerate(objects) : 
                    bboxes = ClientSideBoundingBoxes.get_bounding_boxes(obj, self.camera)
                    bboxes = np.array(bboxes).reshape(-1, 4)
                    class_id = np.ones((bboxes.shape[0], 1)) * i
                    bboxes = np.hstack([class_id, bboxes])
                    
                    ClientSideBoundingBoxes.draw_2d_bounding_boxes(self.display, bboxes)
                    array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (self.image.height, self.image.width, 4))
                    array = array[:, :, :3].astype('uint8')
                    try : 
                        array = ClientSideBoundingBoxes.draw_2d_bounding_boxes_on_image(array, bboxes)
                    except : 
                        print("Error in drawing")
                    
                    cv2.imwrite(f"output/{str(j).zfill(4)}.png", array)
                    j+=1
                    continue
                pygame.display.flip()

                pygame.event.pump()
                if self.control(self.car):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
