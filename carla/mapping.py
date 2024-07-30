import cv2 
import numpy as np
from matplotlib import pyplot as plt

import pygame
from config import *
from geometry import *
from utils import transform_3d_image_to_world, is_point_in_polygon


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
    def __init__(self,
                 height=MAP_HEIGHT, width=MAP_WIDTH, 
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

        self.ego_pos = None
        self.bboxes = None
        self.depth_array = None
        self.velocity = None


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
    
    # utils
    def get_last_ego(self, ) : 
        n = len()
        while n>0 :
            n -= 1
            ego_vehicle = self.ego_q.get ()
            x_, y_, yaw = ego_vehicle 
            self.ego_q.task_done ()
            self.ego_q.put (ego_vehicle)

        ego_xy_yaw = np.array([[x_, y_, yaw]])
        return ego_xy_yaw
    
    def _3d_image_to_world (self, h, w, depth) : 
        xy = np.meshgrid(np.arange(w), np.arange(h))
        xy = np.stack(xy, axis=-1)
        xy = xy.transpose(1, 0, 2).reshape(-1, 2).astype('float32')
        z = depth.mean(-1).reshape(-1)
        xy *= z[:, None]
        self.xyz = np.concatenate([xy, z[:, None]], axis=1)

        return self.xyz
    



    @staticmethod
    def get_vector(ego_pos) : 
        yaw = ego_pos[2]
        v_unit = np.array([np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw) )]).reshape(-1, 1)
        return v_unit
    
    def update(self, 
               ego_pos : list, 
               objects, 
               depth_array, 
               velocity,
               K_inv, 
               camera_2_world, 
               ) :
        '''
        ego_pos: [x, y, yaw] list containing latest x, y position of vehicle
        objects: np.ndarray of shape (n, 5) containing objects in scene
        ''' 

        self.velocity = velocity
        self.image_height, self.image_width = depth_array.shape[:2]



        if self.ego_pos is None : 
            self.ego_pos = []
        # Update ego vehicle track
        if len(self.ego_pos) == EGO_MAX_QUEUE_SIZE: 
            self.ego_pos.pop(0)

        self.ego_pos.append(ego_pos)
        
        # # Update depth array
        
        # self.depth_array = depth_array
        # h, w = depth_array.shape[:2]
        # self.xyz = self._3d_image_to_world(h, w, depth_array)

        # # Update detected bounding boxes
        # # get bounding boxes center in xyz matrix
        # if len(objects)==0 : 
        #     objects = np.random.randint(0, 100, (10, 5))
        # xy = objects[:, 1:3].astype('int32')
        

        # # get homogeneous coords of each object center
        # points_indices = xy[:, 0] * h + xy[:, 1]
        # objects_xyz = self.xyz[points_indices]
        
        # # transform homogenous coords to world coords
        # objects_xyz_3d = transform_3d_image_to_world(objects_xyz, K_inv, camera_2_world)

        ## get bounding boxes in world coordinate
        self.bboxes = objects





    # Geometry
    def front_nearby_objects_world (self, distance=50) : 
        '''
        this function checks if there are vehicles in the same lane in front of ego vehicle
        
        Returns 
            - true if there is vehicles in fron of the car.
            - false if there is no vehicles in fron of the car. 
        '''
        if self.ego_pos is None or self.bboxes is None: 
            return False

        # Get distance from each object 
        last_ego_pos = np.array(self.ego_pos[-1])[:-1].reshape(1, 2)
        objects_pos = self.bboxes[:, 1:3]
        
        DIST = np.linalg.norm(objects_pos - last_ego_pos, axis=-1)
        

        # get vector of ego vehicle
        ego_vector = self.get_vector(self.ego_pos[-1])
        vec_to_object = objects_pos - last_ego_pos

        dot_product = np.dot(vec_to_object, ego_vector)

        self.front_vehicles = (DIST < distance).reshape(-1) & (dot_product > 0).reshape(-1)

        return (dot_product > 0).reshape(-1) & (DIST < distance).reshape(-1)

    def front_nearby_objects_image (self) : 
            '''
            this function checks if there are vehicles in the same lane in front of ego vehicle
            using information in image coordinate

            Returns 
                - true if there is vehicles in fron of the car.
                - false if there is no vehicles in fron of the car. 
            '''
            if self.bboxes is None: 
                return False

            relative_vel = self.velocity * VELOCITY_FRAC + 0.3
            print("relative velocity: ", relative_vel)
            # get boundaries 
            bottom_left = np.array([0, self.image_height])
            bottom_right = np.array([self.image_width, self.image_height])
            top_left = np.array([int(0.35*self.image_width), int(self.image_height-relative_vel*self.image_height)])
            top_right = np.array([int(0.65*self.image_width), int(self.image_height-relative_vel*self.image_height)])
            boundaries = [top_right, top_left, bottom_left, bottom_right]


            # Get distance from each object
            ret = np.zeros(len(self.bboxes), dtype=bool) 
            for i, box in enumerate(self.bboxes) : 
                cls_id, x_min, y_min, x_max, y_max = box
                
                if is_point_in_polygon((x_min, y_min), boundaries) or is_point_in_polygon((x_max, y_max), boundaries) : 
                    ret[i] = 1

            return ret
            

    # Drawing
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
        
        if self.ego_pos is None : 
            return 
        
        # transform points to image coordinate
        image_ego_pos = self.transform_to_image(np.array(self.ego_pos)[:, :2])
        x, y = image_ego_pos
        for i in range(len(x)) :
            cv2.circle(self.array, (int(x[i]), int(y[i])), MAP_POINT_SIZE, (0, 255, 0), -1)


    def draw_objects (self) : 
        
        if self.bboxes is None : 
            return
        
        xy_objects_world = self.bboxes[:, 1:3]
        x, y = self.transform_to_image(xy_objects_world.reshape(-1, 2))

        if self.front_vehicles is not None :
            front_vehicles = self.front_vehicles
            for i in range(len(x)) :
                cv2.circle(self.array, (int(x[i]), int(y[i])), MAP_POINT_SIZE, (0, 0, 255) if front_vehicles[i] else (255, 0, 0), -1)
        else :
            for i in range(len(x)) :
                cv2.circle(self.array, (int(x[i]), int(y[i])), MAP_POINT_SIZE, (255, 0, 0), -1)


    def draw_da (self) : 
        pass

    


    
    def generate_map(self) :
        
        
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
        # if DRAW_DA and da_q.qsize()>0: 
        #     n = da_q.qsize()
        #     while n>0 :
                # n -= 1
                # da = da_q.get()
                # da_c = da.copy()
                # if len(da) : 
                #     da = np.array(da)
                #     da = da.reshape(-1, 2)

                #     x, y = self.transform_to_image(da)
                #     for i in range(len(x)):
                #         cv2.circle(self.array, (int(x[i]), int(y[i])), 5, (50, 50, 50), -1)

                # da_q.task_done()

                # if da_q.qsize() < da_q.maxsize :
                #     da_q.put(da_c)
        


        # ====================================================================================================
        # Draw ego vehicle
        if DRAW_EGO and self.ego_pos is not None: 
            self.draw_ego()



        # ====================================================================================================
        # Draw bounding boxes
        if DRAW_BBOX:
            self.draw_objects() 

        #     bboxes = bboxes.reshape(-1, 5)
        #     colors = np.random.randint(0, 255, size=(len(bboxes), 3), dtype="uint8")

        #     right_obs = self.right_obstacles(bboxes, distance=100).reshape(-1)
        #     # is_between = self.front_vehicles(distance=20).reshape(-1)

        #     is_between = np.ones(len(bboxes), dtype=bool)
        #     _colors = [(0, 255, 0), (255, 0, 0)]
        #     if len(bboxes) > 0: 
        #         for c, bbox, valid_obs in zip(colors, bboxes, is_between):
        #             x1, y1, w, h, _ = bbox
                    
        #             x2, y2 = x1 + w, y1 + h
                    
        #             point1 = np.array([x1, y1]).reshape(-1, 2)
        #             point2 = np.array([x2, y2]).reshape(-1, 2)

        #             x1, y1 = self.transform_to_image(point1)
        #             x2, y2 = self.transform_to_image(point2)
        #             x, y = (x1+x2)/2, (y1+y2)/2

        #             # cv2.rectangle(self.array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        #             cv2.circle(self.array, (int(x), int(y)), 5, _colors[int(valid_obs)], -1)

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

        v_unit = np.array([np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw) )]).reshape(-1, 1)
        dot_product = np.dot(obj_ego_vec, v_unit)
        return (D < distance) & (dot_product > 0)



    # def front_vehicles(self, distance=20) : 
    #     ego_xy_yaw = self.get_last_ego()
    #     bboxes = self.bboxes_q
    #     if bboxes is None :
    #         return [False, ]

    #     bboxes_image = self.transform_to_image(bboxes[:, :2])
        
    #     # boundary_points = get_boundary_points(ego_xy_yaw[0][:2], ego_xy_yaw[0][2], distance=distance)
    #     boundary_points_image = self.transform_to_image(self.boundary_points)
    #     # print(boundary_points_image)

    #     cv2.circle(self.array, (int(boundary_points_image[0][0]), int(boundary_points_image[0][1])), 10, (255, 255, 255), -1)
    #     cv2.circle(self.array, (int(boundary_points_image[1][0]), int(boundary_points_image[1][1])), 10, (255, 255, 255), -1)

    #     is_between = point_between_2points(boundary_points_image[0], boundary_points_image[1], np.array(bboxes_image).T)

    #     v_unit = np.array([np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw) )]).reshape(-1, 1)
    #     obj_xy = bboxes[:, :2]
    #     obj_ego_vec = obj_xy - ego_xy_yaw[0, :2]

    #     dot_product = np.dot(obj_ego_vec, v_unit)
    #     return is_between
    #     # return is_between & (dot_product > 0)






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

