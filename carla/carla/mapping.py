import cv2 
import numpy as np
from matplotlib import pyplot as plt

import pygame
from config import *
from geometry import *


def add_weighted_centered (img1 : np.ndarray, img2 : np.ndarray, weight1 : float, weight2 : float) -> np.ndarray : 
    cx, cy = np.array(img1.shape[:-1][::-1]) // 2

    h, w = img2.shape[:2]
    h=h//2
    w=w//2

    img1[cy-w:cy+w, cx-h:cx+h] = img2 * weight2 + img1[cy-w:cy+w, cx-h:cx+h] * weight1

    return img1

def is_in_view (a : np.ndarray, b : np.ndarray, x : np.ndarray) :
    ax = x - a
    bx = x - b
    ab = b - a

    return np.dot(ax, ab) * np.dot(bx, ab) < 0

def distance (p1 : np.ndarray, p2 : np.ndarray) :
    return np.linalg.norm(p1-p2)


class HDMap() : 
    def __init__(self, buffers,
                 height=MAP_HEIGHT, width=MAP_WIDTH, 
                 bx1=WORLD_BOUNDING.x1, bx2=WORLD_BOUNDING.x2, by1=WORLD_BOUNDING.y1, by2=WORLD_BOUNDING.y2) : 

        self.ego_q, self.bboxes_q, self.lane_lines_q, self.da_q = buffers
        self.height = height
        self.width = width
        self.bx1 = bx1
        self.bx2 = bx2
        self.by1 = by1
        self.by2 = by2

        self.map = np.zeros((self.height, self.width, 3), dtype = "uint8")
        self.lanes_map = np.zeros((self.height, self.width, 3), dtype = "uint8")

        self.world_to_image = np.array([
            [self.width/(self.bx2-self.bx1), 0, -self.width*self.bx1/(self.bx2-self.bx1)],
            [0, -self.height/(self.by2-self.by1), self.height*self.by2/(self.by2-self.by1)],
            [0, 0, 1]
            ]
        )

        carla_logo = cv2.imread('carla.png')
        carla_logo = cv2.resize(carla_logo, (CARLA_LOGO_SIZE, CARLA_LOGO_SIZE))
        fill_color = [int(i) for i in carla_logo[:1, 0].mean(axis=0)]
        cv2.circle(self.map, (MAP_WIDTH//2, MAP_HEIGHT//2), 35, fill_color, -1)

        self.map = add_weighted_centered(self.map, carla_logo, 0.0, 1.0)
        self.canvas = np.zeros((self.height, self.width, 3), dtype = "uint8")
        
        self.surface = pygame.surfarray.make_surface(self.canvas.swapaxes(0, 1))


    def transform_to_image(self, points) :
        points = np.array([points[:, 0], points[:, 1], np.ones(len(points))])
        image_point = np.dot(self.world_to_image, points)
        image_point = image_point[:2]
        x, y = image_point
        x = np.where(x<0, 0, x)
        x = np.where(x>self.width, self.width, x)
        y = np.where(y<0, 0, y)
        y = np.where(y>self.height, self.height, y)

        return x, y
    
    def get_last_ego(self) : 
        n = self.ego_q.qsize()
        while n>0 :
            n -= 1
            ego_vehicle = self.ego_q.get ()
            x_, y_, yaw = ego_vehicle 
            self.ego_q.task_done ()
            self.ego_q.put (ego_vehicle)

        ego_xy_yaw = np.array([[x_, y_, yaw]])
        return ego_xy_yaw
    
    def draw_lanes(self) : 
        if self.ego_q is None: 
            return 
        n = self.ego_q.qsize ()
        ego_start_end = []
        while n>0 :
            n -= 1
            ego_vehicle = self.ego_q.get()
            x_, y_, yaw = ego_vehicle 
            ego_start_end.append(np.array([x_, y_, yaw]))
            self.ego_q.task_done()
            self.ego_q.put(ego_vehicle)

        # ====================================================================================================
        # dummy lane lines
        for lane in self.lane_lines_q:
            start = lane[0]
            end = lane[-1]


            # check if the lane is in the camera view
            condition = distance(ego_start_end[0][:2], start[:2]) < 25 \
                and distance(ego_start_end[1][:2], start[:2]) <     25 \
                and distance(ego_start_end[0][:2], end[:2]) <       25 \
                and distance(ego_start_end[1][:2], end[:2]) <       25
            
            # if is_in_view(ego_start_end[0][:2], ego_start_end[1][:2] , start[:2]) \
            #     and is_in_view(ego_start_end[0][:2], ego_start_end[1][:2], end[:2]) \
            #     and distance(ego_start_end[0][:2], start[:2]) < 50 \
            #     and distance(ego_start_end[1][:2], start[:2]) < 50 \
            #     and distance(ego_start_end[0][:2], end[:2]) <   50 \
            #     and distance(ego_start_end[1][:2], end[:2]) <   50 \
            #     :
            if condition : 
                color = (0, 255, 0)
            else : 
                color = (255, 255, 255)
            


            x, y = self.transform_to_image(lane)
            for i in range(len(x)-1) :
                cv2.line(self.map, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), color, 2)
                


    def draw_ego (self) : 
        pass

    def draw_objects (self) : 
        pass

    def draw_da (self) : 
        pass

    def generate_map(self, ego_vehicle_q, bboxes, lane_lines, da_q, camera_2_world_map) :
        
        self.bboxes_q = bboxes
        self.lane_lines_q = lane_lines
        self.da_q = da_q
        self.ego_q = ego_vehicle_q
        
        
        if self.map is None: 
            self.map = np.zeros((self.height, self.width, 3), dtype='uint8')
        

        # Draw lane lines

        if DRAW_LANELINES: 
            self.draw_lanes()


            # points = lane_lines.reshape(-1, 3)[:,:2]
            # x, y = self.transform_to_image(points)
            # for i in range(len(x)):
            #     cv2.circle(self.lanes_map, (int(x[i]), int(y[i])), 1, (255, 255, 255), -1)
            
            
        self.array = self.map.copy()
        # Draw DA
        if DRAW_DA and da_q.qsize()>0: 
            n = da_q.qsize()
            while n>0 :
                n -= 1
                da = da_q.get()
                da_c = da.copy()
                if len(da) : 
                    da = np.array(da)
                    da = da.reshape(-1, 2)

                    x, y = self.transform_to_image(da)
                    for i in range(len(x)):
                        cv2.circle(self.array, (int(x[i]), int(y[i])), 5, (50, 50, 50), -1)

                da_q.task_done()

                if da_q.qsize() < da_q.maxsize :
                    da_q.put(da_c)
        


        # ====================================================================================================
        # Draw ego vehicle
        if DRAW_EGO and ego_vehicle_q is not None: 
            n = ego_vehicle_q.qsize()
            while n>0 :
                n -= 1
                ego_vehicle = ego_vehicle_q.get()
                x_, y_, yaw = ego_vehicle 

                point = np.array([x_, y_]).reshape(-1, 2)
                x, y = self.transform_to_image(point)
                cv2.circle(self.array, (int(x), int(y)), 3, (0, 255, 0), -1)
                ego_vehicle_q.task_done()
                ego_vehicle_q.put(ego_vehicle)

            # cv2.circle(self.array, (int(x), int(y)), 200, (0, 255, 0), 1)

            xy_yaw = self.get_last_ego()
            plane = get_vehicle_perpendicular_plan(xy_yaw[0, 0], xy_yaw[0, 1], xy_yaw[0, 2])
            self.boundary_points = get_boundary_points(xy_yaw[0, :2], xy_yaw[0, 2], distance=20)

            x, y = self.transform_to_image(self.boundary_points)
            cv2.line(self.array, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), (0, 255, 0), 2)

            x, y = self.transform_to_image(plane)
            cv2.line(self.array, (int(x[0]), int(y[0])), (int(x[1]), int(y[1]),), (255, 255, 0), 2)



        # ====================================================================================================
        # Draw bounding boxes
        if DRAW_BBOX and bboxes is not None: 

            bboxes = bboxes.reshape(-1, 5)
            colors = np.random.randint(0, 255, size=(len(bboxes), 3), dtype="uint8")

            right_obs = self.right_obstacles(bboxes, distance=100).reshape(-1)
            # is_between = self.front_vehicles(distance=20).reshape(-1)

            is_between = np.ones(len(bboxes), dtype=bool)
            _colors = [(0, 255, 0), (255, 0, 0)]
            if len(bboxes) > 0: 
                for c, bbox, valid_obs in zip(colors, bboxes, is_between):
                    x1, y1, w, h, _ = bbox
                    
                    x2, y2 = x1 + w, y1 + h
                    
                    point1 = np.array([x1, y1]).reshape(-1, 2)
                    point2 = np.array([x2, y2]).reshape(-1, 2)

                    x1, y1 = self.transform_to_image(point1)
                    x2, y2 = self.transform_to_image(point2)
                    x, y = (x1+x2)/2, (y1+y2)/2

                    # cv2.rectangle(self.array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.circle(self.array, (int(x), int(y)), 5, _colors[int(valid_obs)], -1)

        self.surface = pygame.surfarray.make_surface(self.array.swapaxes(0, 1))

        return self.array



    def right_obstacles(self, bboxes=None, distance=20) -> bool:
        '''
        this fucntion will return True if there are obstacles on the right lane within the distance
        self.bboxes_q : queue of bounding boxes [x, y, w, h, class_id] of shape (n, 5)
        '''
        if bboxes is None :
            bboxes = self.bboxes_q
        
        if bboxes is None :
            return [False, ]
        obj_xy = bboxes[:, :2]
        
        ego_xy_yaw = self.get_last_ego()
        obj_ego_vec = obj_xy - ego_xy_yaw[0, :2]

        D = np.linalg.norm(obj_ego_vec, axis=1).reshape(-1, 1)

        v_unit = np.array([np.cos(np.deg2rad(ego_xy_yaw[0, 2])), np.sin(np.deg2rad(ego_xy_yaw[0, 2]) )]).reshape(-1, 1)
        dot_product = np.dot(obj_ego_vec, v_unit)
        return (D < distance) & (dot_product > 0)



    def front_vehicles(self, distance=20) : 
        ego_xy_yaw = self.get_last_ego()
        bboxes = self.bboxes_q
        if bboxes is None :
            return [False, ]

        bboxes_image = self.transform_to_image(bboxes[:, :2])
        
        # boundary_points = get_boundary_points(ego_xy_yaw[0][:2], ego_xy_yaw[0][2], distance=distance)
        boundary_points_image = self.transform_to_image(self.boundary_points)
        # print(boundary_points_image)

        cv2.circle(self.array, (int(boundary_points_image[0][0]), int(boundary_points_image[0][1])), 10, (255, 255, 255), -1)
        cv2.circle(self.array, (int(boundary_points_image[1][0]), int(boundary_points_image[1][1])), 10, (255, 255, 255), -1)

        is_between = point_between_2points(boundary_points_image[0], boundary_points_image[1], np.array(bboxes_image).T)

        v_unit = np.array([np.cos(np.deg2rad(ego_xy_yaw[0, 2])), np.sin(np.deg2rad(ego_xy_yaw[0, 2]) )]).reshape(-1, 1)
        obj_xy = bboxes[:, :2]
        obj_ego_vec = obj_xy - ego_xy_yaw[0, :2]

        dot_product = np.dot(obj_ego_vec, v_unit)
        return is_between
        # return is_between & (dot_product > 0)






    def can_turn_right(self) : 
        right_obstacles = self.right_obstacles()
        return not np.any(right_obstacles)
    
    def obstacles_left(self) -> bool:
        pass

    def left_lane(self) -> bool:
        pass
    def right_lane(self) -> bool: 
        pass


    def render(self, display) : 
        display.blit(self.surface, (0, 0))


if __name__ == "__main__" :
    map = HDMap()

