# ------------------------------------------
# This file is part of waymo rasterize.
# File: render.py

# Autor: Mahmoud ElHusseni
# Created on 2024/02/01.
# Github: https://github.com/MahmoudEl-Husseni
# Email: mahmoud.a.elhusseni@gmail.com
# ------------------------------------------
import sys 
if 'av2-api/src' not in sys.path: 
  sys.path.append('av2-api/src')


import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from matplotlib import pyplot as plt

from av2.datasets.motion_forecasting import scenario_serialization as ss
from av2.map.map_api import ArgoverseStaticMap


from config import RADIUS_OFFSET, road_colors, map_object_type, AGENT_COLOR, OBJ_COLOR
from rawTOinterm import extract_agent_features, extract_obj_features

RADIUS_OFFSET = 1.5

def extract_raster(scene, out_path):
  pref = scene.split('/')[-1]
  file_name = scene + "/scenario_" + pref + ".parquet"
  loader = ss.load_argoverse_scenario_parquet(file_name)

  df = ss._convert_tracks_to_tabular_format(loader.tracks)

  log_map_dirpath = Path(scene)
  avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath=log_map_dirpath, build_raster=False)

  agent_data, center, radius = extract_agent_features(loader)
  obj_data = extract_obj_features(df, loader, radius)

  polylines = avm.get_nearby_lane_segments(center, radius*RADIUS_OFFSET)

  fig, axs = plt.subplots(1, 1, figsize=(20, 12))
  for poly in polylines: 
    pts = poly.polygon_boundary[:, :2]
    axs.plot(pts[:, 0], pts[:, 1], c=road_colors[map_object_type[poly.lane_type.value.lower()]])

  xy = agent_data['XY_past']
  xy_ = xy + center

  xyts = obj_data['XYTs']
  for mask in obj_data['mask_tovectors']: 
    xy_o = xyts[mask][:, :2]
    
    axs.scatter(xy_o[:, 0], xy_o[:, 1], c=OBJ_COLOR)

  axs.scatter(xy_[:, 0], xy_[:, 1], c=AGENT_COLOR)

  axs.set_xticks([])
  axs.set_yticks([])
  fig.canvas.draw()
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  map = data[150:1040, 250:1750]

  data = {
      'raster' : map, 
      'gt_marginal' : agent_data['gt_normalized'], 
      'future_val_marginal' : np.ones(len(agent_data['gt_normalized']))
  }

  np.savez(f'{out_path}/{pref}.npz', **data)


def render(data, out_path): 
  os.makedirs(out_path, exist_ok=True)
  scenes = os.listdir(data)
  scenes = [os.path.join(data, scene) for scene in scenes]

  p = multiprocessing.Pool(20)
  res = []
  for scene in scenes: 
    res.append(p.apply_async(extract_raster, kwds=dict(scene=scene, out_path=out_path)))

  for r in tqdm(res):
    r.get()