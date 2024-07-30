import sys
import numpy as np
import carla
import pygame
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class CameraWindow(QMainWindow):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.image_label = QLabel(self)
        self.setCentralWidget(self.image_label)
        self.resize(800, 600)

    def update_image(self, image_data):
        image = QImage(image_data, image_data.shape[1], image_data.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)

def camera_callback(image, window):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3][:, :, ::-1]  # Convert from BGRA to RGB
    window.update_image(array)

def segmentation_callback(image, window):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]  # Convert from BGRA to RGB
    window.update_image(array)

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    segmentation_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    segmentation_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    segmentation_camera = world.spawn_actor(segmentation_bp, segmentation_transform, attach_to=vehicle)

    app = QApplication(sys.argv)

    camera_window = CameraWindow('Camera View')
    segmentation_window = CameraWindow('Semantic Segmentation View')

    camera.listen(lambda image: camera_callback(image, camera_window))
    segmentation_camera.listen(lambda image: segmentation_callback(image, segmentation_window))

    camera_window.show()
    segmentation_window.show()

    timer = QTimer()
    timer.timeout.connect(lambda: vehicle.apply_control(carla.VehicleControl(throttle=0.5)))
    timer.start(50)

    try:
        sys.exit(app.exec_())
    finally:
        camera.stop()
        segmentation_camera.stop()
        camera.destroy()
        segmentation_camera.destroy()
        vehicle.destroy()

if __name__ == '__main__':
    main()