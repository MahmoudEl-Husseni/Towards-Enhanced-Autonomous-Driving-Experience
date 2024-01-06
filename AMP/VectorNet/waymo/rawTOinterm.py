import os
import numpy as np
import tensorflow as tf

from config import *
from utils.data import *
from utils.utils import *

import multiprocessing as mp


def process(scene_file, OUT_DIR):
  os.makedirs(f'{OUT_DIR}/agents', exist_ok=True)
  os.makedirs(f'{OUT_DIR}/obj', exist_ok=True)
  os.makedirs(f'{OUT_DIR}/lanes', exist_ok=True)
  os.makedirs(f'{OUT_DIR}/gt', exist_ok=True)

  data = load_data(scene_file)

  for record in data:
    XY, Velocity, Current_valid, Agents_val, Agent_type, YAWS, \
            GT_XY, Future_valid, Tracks_to_predict, lane_xyz, \
            lane_valid, lane_dir, lane_id, lane_type = record


    sm = get_lane_splitter(lane_id)
    DISP = agent_agent_disp(XY)

    for i in range(len(XY)):
      ret = extract_features(i, XY, Velocity, Current_valid, Agents_val, Agent_type, YAWS, GT_XY, Future_valid,
                            Tracks_to_predict, lane_xyz, lane_valid, lane_dir, lane_id, lane_type,
                            sm, DISP)

      if ret == -1:
        continue

      _agent_data, _obj_data, _lane_data = ret

      agent_vectors = vectorize_agent(_agent_data)
      obj_vectors = vectorize_object(_obj_data)
      lane_vectors = vectorize_lanes(_lane_data)

      pref = generate_pref()
      pref = scene_file[-14:-9] + '_' + pref
      save_agents(agent_vectors, f'{OUT_DIR}/agents', pref)
      save_objects(obj_vectors, f'{OUT_DIR}/obj', pref)
      save_lanes(lane_vectors, f'{OUT_DIR}/lanes', pref)
      save_gt(_agent_data['Normalized_GT'], f'{OUT_DIR}/gt', pref)


def process_scene(record_data, ):

  record = load_scene(record_data)

  XY, Velocity, Current_valid, Agents_val, Agent_type, YAWS, \
          GT_XY, Future_valid, Tracks_to_predict, lane_xyz, \
          lane_valid, lane_dir, lane_id, lane_type = record


  sm = get_lane_splitter(lane_id)
  DISP = agent_agent_disp(XY)

  for i in range(len(XY)):
    ret = extract_features(i, XY, Velocity, Current_valid, Agents_val, Agent_type, YAWS, GT_XY, Future_valid,
                          Tracks_to_predict, lane_xyz, lane_valid, lane_dir, lane_id, lane_type,
                          sm, DISP)

    if ret == -1:
      continue

    _agent_data, _obj_data, _lane_data = ret

    agent_vectors = vectorize_agent(_agent_data)
    obj_vectors = vectorize_object(_obj_data)
    lane_vectors = vectorize_lanes(_lane_data)

    pref = generate_pref()
    pref = scene_file[-14:-9] + '_' + pref
    save_agents(agent_vectors, f'{OUT_DIR}/agents', pref)
    save_objects(obj_vectors, f'{OUT_DIR}/obj', pref)
    save_lanes(lane_vectors, f'{OUT_DIR}/lanes', pref)
    save_gt(_agent_data['Normalized_GT'], f'{OUT_DIR}/gt', pref)


if __name__=="__main__": 

  files = os.listdir(DATA_DIR)
  scene_files = [os.path.join(DATA_DIR, file) for file in files]

  os.makedirs(f'{OUT_DIR}/agents', exist_ok=True)
  os.makedirs(f'{OUT_DIR}/obj', exist_ok=True)
  os.makedirs(f'{OUT_DIR}/lanes', exist_ok=True)
  os.makedirs(f'{OUT_DIR}/gt', exist_ok=True)



  for scene_file in scene_files:    

    dataset = tf.data.TFRecordDataset(
            [scene_file], num_parallel_reads=1
        )

    dataset = tf.data.TFRecordDataset(
          [scene_file], num_parallel_reads=1
      )
    if N_SHARDS > 1:
        dataset = dataset.shard(N_SHARDS, 0)
    p = mp.Pool(16)
    res = []
    for record_data in dataset:
      res.append(p.apply_async(process_scene, args=(record_data,)))
    p.close()

    for r in res:
      r.get()
    p.join()
    
