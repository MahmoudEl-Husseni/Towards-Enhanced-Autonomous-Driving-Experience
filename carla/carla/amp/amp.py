from lanes import *
from objects import *
from amp_config import *
from VectorNet.Argoverse2.Vectornet import VectorNet
from VectorNet.Argoverse2.dataset import Vectorset, custom_collate
from VectorNet.Argoverse2.utils.ckpt_utils import load_checkpoint
from torch.utils.data import DataLoader
import torch




class AMP:

    def __init__(self, world, parent, object_types = None) :
        self.world = world
        self.parent = parent


        if object_types is None :
            object_types = []
            for ob_typ, bp in carla_objects_bp.items() : 
                object_types += [(x, ob_typ) for x in bp]

        self.object_types = object_types
        self.lanes = get_lane_vectors(world)
        self.object_tracker = Object_Tracking(world, parent, self.object_types)
        self.objects_vectors = None

        # prepare VectorNet
        self.vectornet = VectorNet('Argo-1')
        optimiizer = torch.optim.Adam(self.vectornet.parameters(), lr=1e-3)
        schuduler = torch.optim.lr_scheduler.StepLR(optimiizer, step_size=10, gamma=0.1)
        load_checkpoint(CKPT_PATH, self.vectornet, optimiizer, schuduler)
        
        self.centers = None


    def get_agent_vectors (self, object_vectors, candidate_density, distance_matrix) -> np.ndarray: 
        '''
        Get the agent vectors

        args : 
            object_vectors : np.ndarray [59, 1] : object vectors [xs, ys, xe, ye, ts, distance, angle, vx, vy, yaw, object_type]
            candidate_density : int : candidate density
        returns :
            agent_vectors : np.ndarray [59, 8] : agent vectors [xs, ys, xe, ye, vx, vy, yaw, candidate_density]
        '''
        xs = object_vectors[:, 0]
        ys = object_vectors[:, 1]
        xe = object_vectors[:, 2,]
        ye = object_vectors[:, 3]
        vx = object_vectors[:, 7]
        vy = object_vectors[:, 8]
        yaw = object_vectors[:, 9]
        cd = (distance_matrix < CANDIDATE_DENSITY_RANGE).sum(axis=0)

        return np.vstack((xs, ys, xe, ye, vx, vy, yaw, cd)).T


    def get_amp_inputs (self, ) : 

        self.centers = np.empty((0, 2))
        batch = []
        # extract lane vectors 
        player_loc = np.array([self.parent.get_transform().location.x, 
                               self.parent.get_transform().location.y, 
                               self.parent.get_transform().location.z])

        nearby_lane_vectors = get_nearby_lane_vectors(self.lanes, player_loc)


        # extract Agent and Object vectors
        try:
            # Your code that operates on the actor
            self.object_tracker.track_obj_info()
        except RuntimeError as e:
            print(f"RuntimeError: {e}")

        object_ids = self.object_tracker.nearby_objects()
        self.objects_vectors, self.distance_matrix = self.object_tracker.vectorize_objects(object_ids, player_loc)

        # calculate three dimensional matrix of distance between objects over time
        for obj_idx in range(len(self.objects_vectors)) : 

            # get agent vectors 
            agent_vectors = self.get_agent_vectors(self.objects_vectors[obj_idx], len(self.objects_vectors), self.distance_matrix[obj_idx])[None, ...]
            center = np.array(agent_vectors[0, -1, 2:4])
            self.centers = np.vstack([self.centers, center])


            # get Object Vectors
            object_vectors = np.delete(self.objects_vectors, obj_idx, axis=0)
            object_vectors[:, :, :2] -= center
            object_vectors[:, :, 2:4] -= center
            agent_vectors[:, :, :2] -= center
            agent_vectors[:, :, 2:4] -= center


            _batch = [torch.Tensor(agent_vectors)[:, 1:], object_vectors, nearby_lane_vectors, torch.rand(1, 1, 1)]
            
            batch.append(_batch)
        
        return batch
    
    def get_amp_outputs (self, batch) : 
        outputs = []
        dataloader = DataLoader(batch, batch_size=len(batch), collate_fn=custom_collate, shuffle=False)

        for i, data in enumerate(dataloader) : 
            centers = data[0][:, -1, :2:4]
            out = self.vectornet(data)
            y_pred, confidences = out[:, :-N_TRAJ], out[:, -N_TRAJ:]
            y_pred = y_pred.view(-1, N_TRAJ, N_FUTURE, 2)

            # sort accroding to confidence
            y_pred = y_pred.detach().numpy()
            confidences = confidences.detach().numpy()
            idx = np.argsort(confidences, axis=1)[:, :DISPLAY_TRAJ]
            y_pred = y_pred[:, idx]
            
            outputs.append(y_pred[:DISPLAY_TRAJ])

        outputs = np.array(outputs)[0, 0, :, 0]
        centers = self.centers[:, None, ...]
        outputs += centers
        return outputs
    
if __name__ == '__main__' : 
    world = carla.World
    parent = carla.Actor
    amp = AMP(world, parent)
    batch = amp.get_amp_inputs()
    if len(batch) > 0: 
        outputs = amp.get_amp_outputs(batch)