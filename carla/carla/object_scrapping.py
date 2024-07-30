import carla
import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world and the blueprint library
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Set up the vehicle
vehicle_bp = blueprint_library.filter('vehicle')[0]

# Try multiple spawn points until one is free
spawn_points = world.get_map().get_spawn_points()
vehicle = None
for spawn_point in spawn_points:
    try:
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        break
    except RuntimeError as e:
        print(f"Spawn failed at {spawn_point.location}: {e}")
if vehicle is None:
    raise RuntimeError("Failed to find a free spawn point")

# Set up the camera
camera_bp = blueprint_library.find('sensor.camera.rgb')
# Place the camera on the dashboard
camera_transform = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.2), carla.Rotation(pitch=0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Set the camera image save directory
image_dir = '/main/Towards Enhanced Autonomous Vehicle/carla/object_tracking/output/'
os.makedirs(image_dir, exist_ok=True)


# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.eval()

def process_image(image):
    # Convert CARLA image to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Remove alpha channel

    # Save the image
    image_path = os.path.join(image_dir, f"{image.frame}.png")
    cv2.imwrite(image_path, array)

    # Convert numpy array to PIL Image
    img = Image.fromarray(array)

    # Perform object detection using YOLOv3
    results = model(img)

    # Extract bounding boxes and class labels
    detected_objects = results.pandas().xyxy[0].to_dict(orient="records")

    # Create the YOLO format annotation file
    annotation_path = image_path.replace('.png', '.txt')
    with open(annotation_path, 'w') as f:
        for obj in detected_objects:
            class_id = obj['class']  # Class ID
            x_min = obj['xmin']
            y_min = obj['ymin']
            x_max = obj['xmax']
            y_max = obj['ymax']

            x_center = (x_min + x_max) / 2 / image.width
            y_center = (y_min + y_max) / 2 / image.height
            width = (x_max - x_min) / image.width
            height = (y_max - y_min) / image.height

            f.write(f"{image.frame} {obj['name']} {class_id} {x_center} {y_center} {width} {height}\n")

# Add the callback to the camera
camera.listen(lambda image: process_image(image))

try:
    vehicle.set_autopilot(True)
    while True:
        world.wait_for_tick()
except KeyboardInterrupt:
    print("Stopping the vehicle and cleaning up...")
finally:
    camera.stop()
    vehicle.destroy()