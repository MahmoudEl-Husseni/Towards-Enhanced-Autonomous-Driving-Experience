# ------------------------------------------
# This file is part of waymo rasterize.
# File: utils.py

# Autor: Mahmoud ElHusseni
# Created on 2024/02/01.
# Github: https://github.com/MahmoudEl-Husseni
# Email: mahmoud.a.elhusseni@gmail.com
# ------------------------------------------
from scipy.interpolate import UnivariateSpline
import numpy as np

def normalize(vector, point):
  return vector - point


def interpolate_x(timesteps, z):
    n_ts = (timesteps.max() - timesteps.min() + 1)
    x = timesteps
    yx = z
    if n_ts > len(x) and len(x) > 3:
      interp_func_X = UnivariateSpline(x, yx)

      yx_ = []
      it = 0
      for i in range(x.min(), x.max()+1):
        if i not in x:
          yx_.append(interp_func_X(i))
        else:
          yx_.append(yx[it])
          it+=1
    else :
      return yx

    return np.array(yx_)

def Angle_Distance_from_agent(df, loader):
  track_id = loader.focal_track_id
  positions = df.loc[df['track_id']==track_id, ['position_x', 'position_y']].values
  t_ids = df.loc[df['track_id']!=track_id, 'track_id'].unique()
  df['displacement_from_agent'] = np.zeros(len(df))
  df['angle_to_agent'] = np.zeros(len(df))

  for id in t_ids:
    dd = df.loc[df['track_id']==id]
    t = dd['timestep'].values
    agent_p = positions[t - t.min()]
    diff = agent_p - dd[['position_x', 'position_y']].values

    angles = np.arctan(diff[:, 1] / diff[:, 0])
    dd['angle_to_agent'] = angles

    disp = np.linalg.norm(diff, axis=1)
    dd['displacement_from_agent'] = disp

    df[df['track_id']==id] = dd.values
