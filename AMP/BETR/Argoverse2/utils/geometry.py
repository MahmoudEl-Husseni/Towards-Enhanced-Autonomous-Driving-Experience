from config import *

import numpy as np
from itertools import combinations
from scipy.interpolate import UnivariateSpline

import torch
from torch_geometric.data import Batch, Data

def normalize(vector, point):
    """
    Normalizes a vector by subtracting a given point.

    Args:
        vector (np.ndarray): The vector to normalize.
        point (np.ndarray): The point to subtract from the vector.

    Returns:
        np.ndarray: The normalized vector.
    """
    return vector - point

def interpolate_x(timesteps, z):
    """
    Interpolates missing values in a sequence of data points using UnivariateSpline.

    Args:
        timesteps (np.ndarray): The timesteps corresponding to the data points.
        z (np.ndarray): The data points to interpolate.

    Returns:
        np.ndarray: Interpolated data points.
    """
    n_ts = (timesteps.max() - timesteps.min() + 1)
    x = timesteps
    yx = z
    if n_ts > len(x) and len(x) > 3:
        interp_func_X = UnivariateSpline(x, yx)

        yx_ = []
        it = 0
        for i in range(x.min(), x.max() + 1):
            if i not in x:
                yx_.append(interp_func_X(i))
            else:
                yx_.append(yx[it])
                it += 1
    else:
        return yx

    return np.array(yx_)

def get_interpolated_xy(timesteps, x_coord, y_coord):
    """
    Interpolates missing x and y coordinates in a sequence of data points using UnivariateSpline.

    Args:
        timesteps (np.ndarray): The timesteps corresponding to the data points.
        x_coord (np.ndarray): The x coordinates to interpolate.
        y_coord (np.ndarray): The y coordinates to interpolate.

    Returns:
        tuple: Interpolated x and y coordinates.
    """
    n_ts = (timesteps.max() - timesteps.min() + 1)
    x = timesteps
    yx = x_coord
    yy = y_coord
    try:
        if n_ts > len(x) and len(x) > 3:
            interp_func_X = UnivariateSpline(x, yx)
            interp_func_Y = UnivariateSpline(x, yy)

            yx_ = []
            yy_ = []
            it = 0
            for i in range(x.min(), x.max() + 1):
                if i not in x:
                    yx_.append(interp_func_X(i))
                    yy_.append(interp_func_Y(i))
                else:
                    yx_.append(yx[it])
                    yy_.append(yy[it])
                    it += 1
        else:
            return yx, yy
    except:
        return np.array(x_coord), np.array(y_coord)

    return np.array(yx_), np.array(yy_)

def Angle_Distance_from_agent(df, loader):
    """
    Computes the angle and distance of other tracks from a focal track.

    Args:
        df (pd.DataFrame): DataFrame containing tracking data.
        loader (object): DataLoader object with attribute `focal_track_id`.

    Modifies:
        df: Adds columns 'displacement_from_agent' and 'angle_to_agent'.
    """
    track_id = loader.focal_track_id
    positions = df.loc[df['track_id'] == track_id, ['position_x', 'position_y']].values
    t_ids = df.loc[df['track_id'] != track_id, 'track_id'].unique()
    df['displacement_from_agent'] = np.zeros(len(df))
    df['angle_to_agent'] = np.zeros(len(df))

    for id in t_ids:
        dd = df.loc[df['track_id'] == id]
        t = dd['timestep'].values
        agent_p = positions[t - t.min()]
        diff = agent_p - dd[['position_x', 'position_y']].values

        angles = np.arctan(diff[:, 1] / diff[:, 0])
        dd['angle_to_agent'] = angles

        disp = np.linalg.norm(diff, axis=1)
        dd['displacement_from_agent'] = disp

        df[df['track_id'] == id] = dd.values

def n_candidates(df, loader, distance):
    """
    Computes the number of candidate tracks within a certain distance from the focal track at each timestep.

    Args:
        df (pd.DataFrame): DataFrame containing tracking data.
        loader (object): DataLoader object with attribute `focal_track_id`.
        distance (float): The maximum distance to consider a track as a candidate.

    Returns:
        np.ndarray: Array of candidate counts for each timestep.
    """
    Angle_Distance_from_agent(df, loader)
    df['is_candidate'] = df['displacement_from_agent'].apply(lambda x: x <= distance)

    return df.groupby('timestep')['is_candidate'].sum().values

def calc_direction(xyz):
    """
    Calculates the direction between consecutive 3D points.

    Args:
        xyz (np.ndarray): Array of 3D points.

    Returns:
        np.ndarray: Array of direction values between consecutive points.
    """
    direction = []
    lastx, lasty, lastz = xyz[0]
    for d in xyz[1:]:
        x, y, z = d
        dir = (y - lasty) / (x - lastx)
        lastx, lasty, lastz = [x, y, z]
        direction.append(dir)

    direction.append(dir)
    return np.array(direction)

def get_avg_vectors(data, col, n_frames_per_vector, n_past=N_PAST):
    """
    Computes average vectors from the data over specified frame intervals.

    Args:
        data (pd.DataFrame): DataFrame containing vector data.
        col (str): Column name containing vector values.
        n_frames_per_vector (int): Number of frames to average over.
        n_past (int, optional): Number of past frames to consider. Defaults to N_PAST.

    Returns:
        np.ndarray: Array of averaged vectors.
    """
    start_x = data[col][:N_PAST][:-n_frames_per_vector:n_frames_per_vector]
    end_x = data[col][:N_PAST][n_frames_per_vector::n_frames_per_vector]
    x_avg = (start_x + end_x) / 2.0
    return x_avg

def fc_graph(num_nodes):
    """
    Creates a fully connected graph for a given number of nodes.

    Args:
        num_nodes (int): Number of nodes in the graph.

    Returns:
        torch.Tensor: Edge index tensor for the fully connected graph.
    """
    edges = np.array(list(combinations(range(num_nodes), 2)))
    edges2 = edges[:, ::-1]

    edge_index = torch.tensor(np.vstack([edges, edges2]), dtype=torch.long).t().contiguous()

    return edge_index

def progress_bar(i, train_set_len, train_bs, length=70):
    """
    Displays a progress bar indicating the completion of a training step.

    Args:
        i (int): The current training step.
        train_set_len (int): Total length of the training set.
        train_bs (int): Batch size.
        length (int, optional): The length of the progress bar in characters. Defaults to 70.

    Returns:
        str: String representing the progress bar.
    """
    train_steps = (train_set_len / train_bs).__ceil__()

    progress = (i + 1) / train_steps
    eq = '='
    progress_bar = f"{red}{'progress:'}{res} {(f'{(progress * 100):.2f}' + ' %').ljust(7)} [{f'{eq * int(i * length / train_steps)}>'.ljust(length)}]"
    return progress_bar

def create_agent_graph_data(batch_data, num_nodes):
    """
    Creates a batch of graph data for agents.

    Args:
        batch_data (list of np.ndarray): List of agent data arrays.
        num_nodes (int): Number of nodes in each graph.

    Returns:
        Batch: Batch object containing graph data.
    """
    batch_data_ls = []
    batches = []

    for i, raw_data in enumerate(batch_data):
        data = Data()
        data.x = raw_data

        edge_index = fc_graph(num_nodes)

        data.edge_index = edge_index

        batch_data_ls.append(data)
        batches += [*[i] * num_nodes]

    base_data = Batch.from_data_list(batch_data_ls)
    base_data.batch = torch.tensor(batches)

    return base_data

def create_obj_graph(obj, num_nodes):
    """
    Creates a batch of graph data for objects.

    Args:
        obj (list of np.ndarray): List of object data arrays.
        num_nodes (int): Number of nodes in each graph.

    Returns:
        Batch: Batch object containing graph data.
    """
    graph_list = []
    batches = []
    for i, raw_data in enumerate(obj):

        data = Data()
        data.x = raw_data

        edge_index = fc_graph(num_nodes)

        data.edge_index = edge_index

        graph_list.append(data)
        batches += [*[i] * num_nodes]

    base_data = Batch.from_data_list(graph_list)
    base_data.batch = torch.Tensor(batches).to(int)

    return base_data
