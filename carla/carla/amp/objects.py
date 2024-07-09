import numpy as np
from amp_config import *


class Object_Tracking : 

    def __init__(self, world, parent, object_types = ['vehicle']) :
        self.world = world
        self.parent = parent
        self.object_types = object_types 

        self.tracks = dict()


    def track_obj_info(self) : 
        '''
        Track the information of the objects in the world

        Stores
        -------
        objects_info : np.array
            Array of shape (n, 6) where n is the number of objects in the world
            Each row contains the information of the object in the following order:
            [x, y, yaw, vel_x, vel_y, object_type]
        '''
        objects_info = np.empty((0, 6))

        for bp_name, obj_type in self.object_types:
            for npc in self.world.get_actors().filter(bp_name):


                # check if first instances or not 
                if self.tracks.get(npc.id) is None:
                    self.tracks[npc.id] = np.empty((0, 6))

                    
                # bb = npc.bounding_box
                vel = npc.get_velocity()

                dist = npc.get_transform().location.distance(self.parent.get_transform().location)
                if dist < 100:
                # if True: 
                    forward_vec = self.parent.get_transform().get_forward_vector()
                    forward_vec = np.array([forward_vec.x, forward_vec.y, forward_vec.z])
                    ray = npc.get_transform().location - self.parent.get_transform().location
                    ray = np.array([ray.x, ray.y, ray.z])

                    # if forward_vec.dot(ray) > 1:
                    if True:
                        # multiplying x, y by 2 to approximate the argoverse distribution
                        # multiplying vx, vy by 10 to approximate the argoverse distribution
                        
                        obj_info = [npc.get_transform().location.x * 2, 
                                    npc.get_transform().location.y * 2, 
                                    npc.get_transform().rotation.yaw * np.pi/180,
                                    vel.x * 10, 
                                    vel.y * 10, 
                                    carla_object_type[obj_type]]
                        
                        objects_info = np.vstack((objects_info, obj_info))

                if len(self.tracks[npc.id]) == OBJECT_TRACK_LENGTH:
                    self.tracks[npc.id] = self.tracks[npc.id][1:]

                self.tracks[npc.id] = np.vstack((self.tracks[npc.id], obj_info))




    def nearby_objects (self, player_loc=None, threshold=OBJECT_NEAR_DISTANCE) : 
        '''
        Get the objects in the world that are nearby the player

        Parameters
        ----------
        threshold : float
            The maximum distance from the player to the object

        Returns
        -------
        np.array
            Array of object ids that are nearby the player
        '''
        
        if player_loc is None:
            player_loc = self.parent.get_transform().location
            player_loc = np.array([player_loc.x, player_loc.y])

        objects_info = []
        for key in self.tracks.keys():
            # print(key, self.tracks[key].shape)
            objects_info.append(self.tracks[key][-1])
        
        objects_info = np.vstack(objects_info)

        distances = np.linalg.norm(objects_info[:, :2] - player_loc, axis=1)

        return np.array(list(self.tracks.keys()))[distances < threshold]

    @staticmethod
    def _vectorize_obj(object_info : np.ndarray, player_loc : np.ndarray) -> np.ndarray : 
        '''
        Vectorize the object information
        Parameters
        ----------
        object_info : np.array
            Array of shape (n, 6) where n is the number of objects in the world
            Each row contains the information of the object in the following order:
            [x, y, yaw, vel_x, vel_y, object_type]
        player_loc : np.array
            Array of shape (2,) containing the x and y coordinates of the player

        Returns
        -------
        np.array
            Array of shape (n, 11) where n is the number of objects in the world

            [xs, ys, xe, ye, ts, Distance, angle from agent, vx, vy, yaw, object_type]
        '''
        xs = object_info[:-1, 0]
        ys = object_info[:-1, 1]
        xe = object_info[1:, 0]
        ye = object_info[1:, 1]
        ts = np.arange(0, object_info.shape[0]-1)

        distances = np.linalg.norm(object_info[:, :2] - player_loc, axis=1).reshape(-1, 1)
        distances = (distances[:-1, 0] + distances[:1, 0])/2

        angles = np.arctan2(object_info[:, 1] - player_loc[1], object_info[:, 0] - player_loc[0]).reshape(-1, 1)
        angles = (angles[:-1, 0] + angles[1:, 0])/2

        vx = object_info[:, 3].reshape(-1, 1)
        vx = (vx[:-1, 0] + vx[1:, 0])/2

        vy = object_info[:, 4].reshape(-1, 1)
        vy = (vy[:-1, 0] + vy[1:, 0])/2

        yaw = object_info[:, 2].reshape(-1, 1)
        yaw = (yaw[:-1, 0] + yaw[1:, 0])/2

        object_type = object_info[:-1, 5]

        return np.vstack((xs, ys, xe, ye, ts, distances, angles, vx, vy, yaw, object_type)).T
    
    def calc_distance(self, object_vectors) :
        '''
        Calculate the distance between the objects over time
        Parametrs: 
        ----------
        object_vectors : np.array
            Array of shape (n, T, 11) where n is the number of objects in the world

            [xs, ys, xe, ye, ts, Distance, angle from agent, vx, vy, yaw, object_type]

        Returns
        -------
        np.array
            Array of shape (n, n, T) where n is the number of objects in the world
            and T is the number of time steps

        ''' 
        object_vectors = object_vectors[:, :, :2]
        T = object_vectors.shape[1]
        n = object_vectors.shape[0]
        DIST = np.zeros((n, n, T))
        for i in range(n) :
            DIST[i, :, :] = np.linalg.norm(object_vectors - object_vectors[i], axis=2)
        
        return DIST
    

    def vectorize_objects (self, object_ids, player_loc) : 
        '''
        Vectorize the objects in the world

        Parameters
        ----------
        object_ids : np.array
            Array of object ids that are nearby the player

        Returns
        -------
        np.array
            Array of shape (n, 60, 11) where n is the number of objects in the world

            [xs, ys, xe, ye, ts, Distance, angle from agent, vx, vy, yaw, object_type]
        '''
        objects_vectors = np.empty((0, OBJECT_TRACK_LENGTH-1, 11))
        for key in object_ids:

            if len(self.tracks[key]) < OBJECT_TRACK_LENGTH:
                continue

            player_loc = self.parent.get_transform().location
            player_loc = np.array([player_loc.x, player_loc.y])

            _vector = self._vectorize_obj(self.tracks[key], player_loc)[None, ...]
            objects_vectors = np.vstack((objects_vectors, _vector))


        # calculate Distance matrix
        distance_matrix = self.calc_distance(objects_vectors)
        
        return objects_vectors, distance_matrix
        

if __name__ == '__main__':
    import sys
    import os
    import glob
    import random

    try:
        sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass

    print('Testing get_bboxes')
    import carla
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
    blueprint = world.get_blueprint_library().filter("model3")[0]
    player = world.try_spawn_actor(blueprint, spawn_point)

    obj_types = []
    for ob_typ, bp in carla_objects_bp.items() : 
        obj_types += [(x, ob_typ) for x in bp]

    Object_tracker = Object_Tracking(world, player, object_types = obj_types)

    i=0
    while True : 
        world.tick(1)

        Object_tracker.track_obj_info()
        object_ids = Object_tracker.nearby_objects()

        player_loc = Object_tracker.parent.get_transform().location
        player_loc = np.array([player_loc.x, player_loc.y])
        objects_vectors = Object_tracker.vectorize_objects(object_ids, player_loc)
        print('Shape of objects_vectors:', objects_vectors.shape)
        
        i+=1

        if i>70:
            break
    # print("Shape of bboxes:", bboxes.shape)
    # print('bboxes:', bboxes)
    # print('types: ', bboxes[:, -1])
    # print('Done testing get_bboxes')
    