from __future__ import annotations

import os

import numpy as np
import polars as pl
import tensorflow as tf
from pymovements.gaze.transforms import pix2deg
from pymovements.gaze.transforms import pos2vel
from tqdm import tqdm


import config


def choose_gpu(gpu: str) -> None:
    # select graphic card
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1.
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)


def get_samples(
        df: pl.DataFrame,
        label_df: pl.DataFrame,
        window_in_ms: int,
        padding: bool = False,
) -> tuple[np.array, np.array]:
    arr = np.empty((0, window_in_ms, 3))
    label_arr = np.empty((0, len(label_df.columns)))
    book_names = df['book_name'].unique().to_list()
    screen_ids = list(range(1, 6))
    for book_name in book_names:
        book_df = df.filter(pl.col('book_name') == book_name)
        label_book_df = label_df.filter(pl.col('book') == book_name)
        for screen_id in screen_ids:
            tmp_book_df = book_df.filter(pl.col('screen_id') == screen_id)
            _window_id = 0
            # cut into windows
            while True:
                try:
                    tmp_df = tmp_book_df[_window_id*window_in_ms:(_window_id+1)*window_in_ms]
                    tmp_fix_coord = tmp_df[['x_left', 'y_left']]
                    tmp_fix = tmp_df[['x_left', 'y_left', 'screen_id']]
                    _coord_fix_arr = tmp_fix_coord.to_numpy()

                    # https://www.viewsonic.com/de/products/sheet/G90fB
                    deg_fix_arr = pix2deg(
                        _coord_fix_arr,
                        screen_px=(1024, 768),
                        screen_cm=(44.5, 42.4),
                        distance_cm=70,
                        origin='center',
                    )
                    vel_fix_arr = pos2vel(deg_fix_arr, sampling_rate=1000)
                    tmp_fix_arr = tmp_fix.to_numpy()
                    tmp_fix_arr[:, [0, 1]] = vel_fix_arr
                    arr = np.vstack([arr, np.expand_dims(tmp_fix_arr, axis=0)])
                    label_arr = np.vstack([label_arr, label_book_df.to_numpy()])
                except ValueError:
                    if not padding:
                        break
                    tmp_df = tmp_book_df[_window_id*window_in_ms:(_window_id+1)*window_in_ms]
                    tmp_fix_coord = tmp_df[['x_left', 'y_left']]
                    tmp_fix = tmp_df[['x_left', 'y_left', 'screen_id']]
                    _coord_fix_arr = tmp_fix_coord.to_numpy()

                    # https://www.viewsonic.com/de/products/sheet/G90fB
                    deg_fix_arr = pix2deg(
                        _coord_fix_arr,
                        screen_px=(768, 1024),
                        screen_cm=(42.4, 44.5),
                        distance_cm=70,
                        origin='center',
                    )
                    vel_fix_arr = pos2vel(deg_fix_arr, sampling_rate=1000)
                    tmp_fix_arr = tmp_fix.to_numpy()
                    tmp_fix_arr[:, [0, 1]] = vel_fix_arr
                    tmp_fix_arr_pad = np.expand_dims(
                        np.pad(
                            tmp_fix_arr,
                            ((0, window_in_ms - len(vel_fix_arr)), (0, 0)),
                        ), axis=0,
                    )
                    arr = np.vstack([arr, tmp_fix_arr_pad])
                    label_arr = np.vstack([label_arr, label_book_df.to_numpy()])
                    break
                _window_id += 1

    return arr, label_arr


def get_sb_sat_data(window_in_ms: int,
                    verbose: int = 0) -> tuple[np.array, np.array, dict[str, int]]:
    if verbose == 0:
        disable = True
    else:
        disable = False
        
    data_path = config.SB_SAT_DIR_PATH
    csv_dir_path = os.path.join(data_path, 'csvs')

    label_df = pl.read_csv(os.path.join(data_path, 'labels.csv'))
    X_arr = np.empty((0, window_in_ms, 3))
    label_dict = {
        'acc': 0, 'confidence': 1, 'difficulty': 2, 'familiarity': 3, 'recognition': 4,
        'interest': 5, 'pressured': 6, 'sleepiness': 7, 'subj_acc': 8, 'native': 9, 'book': 10,
        'subj': 11,
    }
    label_list = list(label_dict.keys())
    label_arr = np.empty((0, len(label_dict)))
    for file_name in tqdm(os.listdir(csv_dir_path), disable = disable):
        subj = file_name.split('.')[0]
        reader_label_df = label_df.filter(pl.col('subj') == subj).select(label_list)
        em_df = pl.read_csv(os.path.join(csv_dir_path, file_name), separator='\t')
        arr, tmp_label_arr = get_samples(em_df, reader_label_df, window_in_ms)
        X_arr = np.vstack([X_arr, arr])
        label_arr = np.vstack([label_arr, tmp_label_arr])

    return X_arr, label_arr, label_dict
