import numpy as np


def get_vehicle_perpendicular_plan (x : float, y : float, yaw : float) -> np.ndarray:
    """
    Get the perpendicular plane of the vehicle

    return : 
        np.ndarray : the perpendicular plane of the vehicle represented by two points start and end
    """
    yaw = np.deg2rad(yaw)
    v_unit = np.array([np.cos(yaw), np.sin(yaw)])
    vx, vy = [-v_unit[1], v_unit[0]]
    x1, y1 = x + 10*vx, y + 10*vy
    x2, y2 = x - 10*vx, y - 10*vy
    return np.array([[x, y], [x1, y1]]) 

def point_between_2points(a : np.ndarray, b : np.ndarray, x : np.ndarray) : 
    '''
    function to check whether the point x is between the two points a and b
    '''
    ax = x - a
    bx = x - b
    ab = b - a

    return (np.dot(ax.T, ab) * np.dot(bx.T, ab)) <= 0

def get_boundary_points (point : np.ndarray, yaw : float, distance : float=10) : 
    '''
    Get the boundary points of the vehicle
    '''
    yaw = np.deg2rad(yaw)
    v_unit = np.array([np.cos(yaw), np.sin(yaw)])
    vx, vy = [-v_unit[1], v_unit[0]]
    # vx, vy = v_unit

    x, y = point
    x1, y1 = x + distance*vx, y + distance*vy
    x2, y2 = x - distance*vx, y - distance*vy
    return np.array([[x1, y1], [x2, y2]])
