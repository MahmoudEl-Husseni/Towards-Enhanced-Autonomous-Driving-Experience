# ------------------------------------------
# This file is part of waymo rasterize.
# File: rawTOinterm.py

# Autor: Mahmoud ElHusseni
# Created on 2024/02/01.
# Github: https://github.com/MahmoudEl-Husseni
# Email: mahmoud.a.elhusseni@gmail.com
# ------------------------------------------
from config import N_PAST
from utils import Angle_Distance_from_agent, normalize, interpolate_x

import sys
import numpy as np

sys.path.append("av2-api/src")
import warnings
warnings.simplefilter('ignore')


from av2.datasets.motion_forecasting import scenario_serialization as ss
# =================================================================


def extract_agent_features(loader): # Time Complexity -> O(1) // discarding numpy orperations & Loading from Disk
  df = ss._convert_tracks_to_tabular_format(loader.tracks)
  track_id_ = loader.focal_track_id
  agent_df = df[df['track_id']==track_id_]
  cur_df = df[df['timestep']==N_PAST-1]

  # Past XY
  vec = agent_df[agent_df['timestep']<N_PAST][['position_x', 'position_y']].values
  center = cur_df.loc[cur_df['track_id']==track_id_, ['position_x', 'position_y']].values.reshape(-1)

  XY_past = normalize(vec, center)

  # Candidates Denisity
  fl_points = agent_df.sort_values(by='timestep').iloc[[0, N_PAST]].loc[:, ['position_x', 'position_y']].values
  radius = np.linalg.norm(fl_points[1]-fl_points[0])
  
  # Ground Truth Trajectory
  gt = agent_df.loc[agent_df['timestep']>=N_PAST, ['position_x', 'position_y']].values

  # Normalized GT
  gt_normalized = normalize(gt, center)

  data = {
      'XY_past' : XY_past,
      'gt_normalized' : gt_normalized
  }
  return data, center, radius



def extract_obj_features(df, loader, radius, distance_ratio=1.5): # Time Complexity -> O(n) (n: Number of objects)

  focal_track_id = loader.focal_track_id

  norm_vec = df.loc[(df['track_id']==focal_track_id) & (df['timestep']==N_PAST), ['position_x', 'position_y']].values.reshape(-1)
  obj_track_ids = df.loc[df['track_id']!=focal_track_id, 'track_id'].unique()

  XYTs = np.empty((0, 5))

  mask_tovectors = []
  end = 0
  p_id = 0

  Angle_Distance_from_agent(df, loader)
  # return df


  agent_avg_past_velocity = np.linalg.norm(df.loc[(df['track_id']==focal_track_id) & (df['timestep']<N_PAST), ['velocity_x', 'velocity_y']].values, axis=1).mean()
  df = df[df['displacement_from_agent'] < agent_avg_past_velocity * distance_ratio]
  

  for t_id in obj_track_ids:

    obj_df = df[(df['track_id']==t_id) & (df['timestep']<N_PAST)]

    # XY_past
    t = obj_df['timestep'].values
    yx = obj_df['position_x'].values
    yy = obj_df['position_y'].values
    
    if len(t) < 3:
      continue
	
    # yx_, yy_ = get_interpolated_xy(t, yx, yy)
    yx_ = interpolate_x(t, yx)
    yy_ = interpolate_x(t, yy)
    yx_norm = normalize(yx_, norm_vec[0]).reshape(-1, 1)
    yy_norm = normalize(yy_, norm_vec[1]).reshape(-1, 1)

    # timestep
    timesteps = np.arange(t.min(), t.min()+len(yx_))

    # Object type
    XYTs = np.vstack((XYTs, np.hstack((yx_.reshape(-1, 1), yy_.reshape(-1, 1), yx_norm, yy_norm, timesteps.reshape(-1, 1)))))
    
    mask_tovectors.append(slice(end, len(XYTs)))
    end = len(XYTs)

    p_id += 1
  data = {
      'XYTs' : XYTs,
      'mask_tovectors' : mask_tovectors
  }
  return data