import os 
import numpy as np

from config import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class WaymoDataset(Dataset):
  """
  A PyTorch Dataset class for loading Waymo Open Dataset data.

  Args:
    data_path (str): The path to the Waymo Open Dataset directory.
    type (str): The type of Waymo Open Dataset data to load, either 'train' or 'val'.
    device (str): The device to load the data to, either 'cpu' or 'cuda'.

  Returns:
    A PyTorch Dataset object containing the Waymo Open Dataset data.
  """
  
  def __init__(self, data_path=DIR.RENDER_DIR, type='train', device=config.DEVICE, 
               vis=False):
    self.path = data_path
    self.type = type
    self.vis = vis

    self.path = os.path.join(self.path, self.type)
    self.data_paths = os.listdir(self.path)

  def __len__(self):
    return len(os.listdir(self.path))

  def __getitem__(self, idx):
    serial_path = self.data_paths[idx]

    if not isinstance(idx, slice):
      serial_path = [serial_path]

    rasters = []
    trajs = []
    is_available = []
    DATA = []
    for p in serial_path:

      serial_path = os.path.join(self.path, p)
      data = np.load(serial_path, allow_pickle=True)
      DATA.append(data)
      raster = data['raster']
      raster = raster.transpose(2, 1, 0)

      trajectory = data["gt_marginal"]
      is_available = data["future_val_marginal"]

      rasters.append(raster)
      trajs.append(trajectory)

    trajs = torch.Tensor(np.array(trajs))
    trajs = torch.flatten(trajs, start_dim=1)
    is_available = torch.Tensor(np.array(is_available))
    
    if self.vis:
      return DATA
    
    return torch.Tensor(rasters), trajs, is_available