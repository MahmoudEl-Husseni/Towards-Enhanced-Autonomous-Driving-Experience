import math
import numpy as np

def get_transform_matrix(transform):
    # Convert rotation to radians
    pitch = math.radians(transform.rotation.pitch)
    yaw = math.radians(transform.rotation.yaw)
    roll = math.radians(transform.rotation.roll)

    # Define rotation matrices
    rotation_matrix_pitch = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])

    rotation_matrix_yaw = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])

    rotation_matrix_roll = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])

    # Combine rotation matrices
    rotation_matrix = rotation_matrix_yaw @ rotation_matrix_pitch @ rotation_matrix_roll

    # Create translation matrix
    translation_matrix = np.array([
        [1, 0, 0, transform.location.x],
        [0, 1, 0, transform.location.y],
        [0, 0, 1, transform.location.z],
        [0, 0, 0, 1]
    ])

    # Combine rotation and translation
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = [transform.location.x, transform.location.y, transform.location.z]

    return transform_matrix


def get_inverse_matrix(transform) :

    matrix = get_transform_matrix(transform) 
    # Calculate inverse matrix
    rot = matrix[:3, :3]

    inv = np.eye(4)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ matrix[:3, 3]

    return inv


# data 
edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

def get_camera_intrinsic(camera_bp):
    # Get the camera intrinsic matrix
    # camera_bp: carla.sensor.CameraBlueprint object
    width = camera_bp.get_attribute('image_size_x').as_int()
    height = camera_bp.get_attribute('image_size_y').as_int()
    fov = camera_bp.get_attribute('fov').as_float()
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    

    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    

    return K
def get_inverse_intrinsic(camera_bp):
    # Get the camera intrinsic matrix
    # camera_bp: carla.sensor.CameraBlueprint object
    width = camera_bp.get_attribute('image_size_x').as_int()
    height = camera_bp.get_attribute('image_size_y').as_int()
    fov = camera_bp.get_attribute('fov').as_float()
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    

    K = np.identity(3)
    K[0, 0] = K[1, 1] = 1/focal
    K[0, 2] = - width / (2.0 * focal)
    K[1, 2] = - height / (2.0 * focal)
    

    return K


def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]


def transform_3d_image_to_world(xyz : np.ndarray, camera_bp, camera_2_world) -> np.ndarray:
    '''
    transform 3d image coordinate to world coordinate
    xyz : np.ndarray : 3d image homogeneous coordinate 
    '''

    # Get the camera intrinsic matrix
    K = get_camera_intrinsic(camera_bp)
    K_inv = np.linalg.inv(K)



    camera_points = np.dot(K_inv, xyz.T).T
    camera_points = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1))], axis=1)

    x, y, z = camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]

    camera_points = np.stack([y, -z, x], axis=1)
    # camera_points = np.stack([z, x, -y], axis=1)

    camera_points = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1))], axis=1)

    world_points = np.dot(camera_2_world, camera_points.T).T
    world_points = world_points[:, :3]

    # print(world_points[:5])
    # print(50*'-')
    
    return world_points