import os 
import numpy as np
from tqdm import tqdm
import multiprocessing

import tensorflow as tf

from render import *
from config import *

def render(out_path, 
           data,
           n_shards=8, 
           validate=True, 
           use_vectorize=False, 
           n_jobs=20, 
           each=0, 
           no_valid=False):
    """
    Render the data from the TFRecord files into images.

    Args:
        out_path (str): The path to the directory where the images will be saved.
        data (str): The path to the directory containing the TFRecord files.
        n_shards (int, optional): The number of shards to use. Defaults to 8.
        validate (bool, optional): Whether to validate the data. Defaults to True.
        use_vectorize (bool, optional): Whether to use vectorized rendering. Defaults to False.
        n_jobs (int, optional): The number of processes to use. Defaults to 20.
        each (int, optional): The shard to use. Defaults to 0.
        no_valid (bool, optional): Whether to skip validation. Defaults to False.
    """

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    files = os.listdir(data)
    dataset = tf.data.TFRecordDataset(
        [os.path.join(data, f) for f in files], num_parallel_reads=1
    )


    if n_shards > 1:
        dataset = dataset.shard(n_shards, each)

    p = multiprocessing.Pool(n_jobs)
    proc_id = 0
    res = []
    
    for data in tqdm(dataset):

        proc_id += 1
        example = tf.train.Example()
        example.ParseFromString(data.numpy())
        
        res.append(
            p.apply_async(
                merge,
                kwds=dict(
                    data=example,
                    proc_id=proc_id,
                    validate=not no_valid,
                    out_dir=out_path,
                    use_vectorize=use_vectorize,
                ),
            )
        )
    for r in tqdm(res):
        r.get()


if __name__=="__main__":

    # render data
    render(**RENDER_ARGS)

    # # make directories
    os.makedirs(os.path.join(DIR.RENDER_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(DIR.RENDER_DIR, 'val'), exist_ok=True)
    os.makedirs(os.path.join(DIR.RENDER_DIR, 'test'), exist_ok=True)


    # move files
    data_paths = os.listdir(DIR.RENDER_DIR)

    np.random.shuffle(data_paths)
    n = len((data_paths))

    n_train = int(config.TRAIN_SZ * n)
    n_val = int(config.VAL_SZ * n)
    n_test = int(config.TEST_SZ * n)

    for i in range(n_train):
        file = data_paths[i]
        train_path = os.path.join(DIR.RENDER_DIR, 'train', file)
        origin_path = os.path.join(DIR.RENDER_DIR, file)

        if file in ['train', 'test', 'val'] or not os.path.isfile(origin_path):
            continue
        os.rename(origin_path, train_path)
        # break

    for i in range(n_train, n_val + n_train):
        file = data_paths[i]
        val_path = os.path.join(DIR.RENDER_DIR, 'val', file)
        origin_path = os.path.join(DIR.RENDER_DIR, file)

        if file in ['train', 'test', 'val'] or not os.path.isfile(origin_path):
            continue

        os.rename(origin_path, val_path)
        # break

    for i in range(-n_test, n):
        file = data_paths[i]
        test_path = os.path.join(DIR.RENDER_DIR, 'test', file)
        origin_path = os.path.join(DIR.RENDER_DIR, file)

        if file in ['train', 'test', 'val'] or not os.path.isfile(origin_path):
            continue


        os.rename(origin_path, test_path)
        # break

    train_path = os.path.join(DIR.RENDER_DIR, 'train')
    val_path = os.path.join(DIR.RENDER_DIR, 'val')
    test_path = os.path.join(DIR.RENDER_DIR, 'test')

    print(n_train, ', ', len(os.listdir(train_path)))
    print(n_val, ', ', len(os.listdir(val_path)))
    print(n_test, ', ', len(os.listdir(test_path)))
