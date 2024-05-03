import cv2 
import numpy as np
from matplotlib import pyplot as plt

import pygame
from config import *

def add_weighted_centered (img1 : np.ndarray, img2 : np.ndarray, weight1 : float, weight2 : float) -> np.ndarray : 
    cx, cy = np.array(img1.shape[:-1][::-1]) // 2

    h, w = img2.shape[:2]
    h=h//2
    w=w//2

    img1[cy-w:cy+w, cx-h:cx+h] = img2 * weight2 + img1[cy-w:cy+w, cx-h:cx+h] * weight1

    return img1

def is_in_view (p1 : np.ndarray, p2 : np.ndarray, p3 : np.ndarray) :
    v1 = p2 - p1
    v2 = p3 - p1

    return np.cross(v1, v2) > 0

def distance (p1 : np.ndarray, p2 : np.ndarray) :
    return np.linalg.norm(p1-p2)


class HDMap() : 
    def __init__(self, height=MAP_HEIGHT, width=MAP_WIDTH, 
                 bx1=WORLD_BOUNDING.x1, bx2=WORLD_BOUNDING.x2, by1=WORLD_BOUNDING.y1, by2=WORLD_BOUNDING.y2) : 

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



    def generate_map(self, ego_vehicle_q, bboxes, lane_lines, da_q, camera_2_world_map) :
        if self.map is None: 
            self.map = np.zeros((self.height, self.width, 3), dtype='uint8')
        
        n = ego_vehicle_q.qsize()
        ego_start_end = []
        while n>0 :
            n -= 1
            ego_vehicle = ego_vehicle_q.get()
            x_, y_, yaw = ego_vehicle 
            ego_start_end.append(np.array([x_, y_, yaw]))
            ego_vehicle_q.task_done()
            ego_vehicle_q.put(ego_vehicle)

        # ====================================================================================================
        # Draw lane lines
        draw_lanes=True

        if draw_lanes: 
            # dummy lane lines
            for lane in lane_lines:
                start = lane[0]
                end = lane[-1]


                # check if the lane is in the camera view
                if is_in_view(ego_start_end[0][:2], ego_start_end[1][:2] , start[:2]) \
                    and is_in_view(ego_start_end[0][:2], ego_start_end[1][:2], end[:2]) \
                    and distance(ego_start_end[0][:2], start[:2]) < 100 \
                    and distance(ego_start_end[1][:2], start[:2]) < 100 \
                    and distance(ego_start_end[0][:2], end[:2]) < 100 \
                    and distance(ego_start_end[1][:2], end[:2]) < 100 \
                    :
                


                    x, y = self.transform_to_image(lane)
                    for i in range(len(x)-1):
                        cv2.line(self.map, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), (255, 255, 255), 2)



            # points = lane_lines.reshape(-1, 3)[:,:2]
            # x, y = self.transform_to_image(points)
            # for i in range(len(x)):
            #     cv2.circle(self.lanes_map, (int(x[i]), int(y[i])), 1, (255, 255, 255), -1)
            
        self.array = self.map.copy()
        # Draw DA
        draw_da=True
        if draw_da and da_q.qsize()>0: 
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
        draw_ego=True
        if draw_ego : 
            n = ego_vehicle_q.qsize()
            while n>0 :
                n -= 1
                ego_vehicle = ego_vehicle_q.get()
                x_, y_, yaw = ego_vehicle 

                point = np.array([x_, y_]).reshape(-1, 2)
                x, y = self.transform_to_image(point)
                cv2.circle(self.array, (int(x), int(y)), 3, (0, 255, 0), -1)
                # Draw heading
                x2 = x + np.cos(yaw)
                y2 = y + np.sin(yaw)
                cv2.line(self.array, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
                ego_vehicle_q.task_done()
                ego_vehicle_q.put(ego_vehicle)



        # ====================================================================================================
        # Draw bounding boxes
        draw_bboxes=False
        if draw_bboxes: 
            bboxes = bboxes.reshape(-1, 5)
            colors = np.random.randint(0, 255, size=(len(bboxes), 3), dtype="uint8")
            if len(bboxes) > 0: 
                for c, bbox in zip(colors, bboxes):
                    x1, y1, w, h, _ = bbox
                    
                    x2, y2 = x1 + w, y1 + h
                    
                    point1 = np.array([x1, y1]).reshape(-1, 2)
                    point2 = np.array([x2, y2]).reshape(-1, 2)

                    x1, y1 = self.transform_to_image(point1)
                    x2, y2 = self.transform_to_image(point2)
                    x, y = (x1+x2)/2, (y1+y2)/2
                    
                    # cv2.rectangle(self.array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.circle(self.array, (int(x), int(y)), 5, (0, 0, 255), -1)



            

        self.surface = pygame.surfarray.make_surface(self.array.swapaxes(0, 1))



    def render(self, display) : 
        display.blit(self.surface, (0, 0))


if __name__ == "__main__" :
    map = HDMap()
