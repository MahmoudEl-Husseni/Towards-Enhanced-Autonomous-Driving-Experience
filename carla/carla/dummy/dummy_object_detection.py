import numpy as np
from utils import get_transform_matrix, get_inverse_matrix, get_image_point
    

def get_bboxes(world, parent) : 
    bboxes = np.empty((0, 5))
    # bboxes = np.empty((0, 8))
    for npc in world.get_actors().filter('*vehicle*'):
        if npc.id != parent.id:
            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(parent.get_transform().location)
            # if dist < 50:
            if True: 
                forward_vec = parent.get_transform().get_forward_vector()
                forward_vec = np.array([forward_vec.x, forward_vec.y, forward_vec.z])
                ray = npc.get_transform().location - parent.get_transform().location
                ray = np.array([ray.x, ray.y, ray.z])

                # if forward_vec.dot(ray) > 1:
                if True:
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    x_max = -1
                    x_min = 100000
                    y_max = -1
                    y_min = 100000
                    for vert in verts:

                        p = [vert.x, vert.y, vert.z]
                        if p[0] > x_max:
                            x_max = p[0]
                        # Find the leftmost vertex
                        if p[0] < x_min:
                            x_min = p[0]
                        # Find the highest vertex
                        if p[1] > y_max:
                            y_max = p[1]
                        # Find the lowest  vertex
                        if p[1] < y_min:
                            y_min = p[1]
                            
                    w = x_max - x_min
                    h = y_max - y_min

                    bbox = [x_min, y_min, w, h, 0]

                    bboxes = np.vstack((bboxes, bbox))

    return np.array(bboxes)