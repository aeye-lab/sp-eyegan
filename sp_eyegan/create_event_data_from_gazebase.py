from __future__ import annotations

import argparse
import math
import os
import random
import socket
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm

import config as config
from sp_eyegan.preprocessing import data_loader as data_loader
from sp_eyegan.preprocessing import event_detection as event_detection
from sp_eyegan.preprocessing import smoothing as smoothing

def list_string(s):
    split_string = s.split(',')
    out_list = []
    for split_elem in split_string:
        out_list.append(split_elem.strip().replace('\'','').replace('[','').replace(']',''))
    return out_list


def list_int(s):
    split_string = s.split(',')
    out_list = []
    for split_elem in split_string:
        out_list.append(int(split_elem.strip().replace('\'','').replace('[','').replace(']','')))
    return out_list

def main():
    # gloabal params
    sampling_rate = 1000

    # params
    pp_params=(None, 6)
    minDurFix=20
    minDurSac= 6
    input_velocity=True
    blink_threshold=0.6
    min_blink_duration=1
    velocity_threshold=0.1
    smoothing_window_length=0.007
    verbose = 0
    disable = False
    dispersion_threshold = 1.6
    min_degree_visual_angle = 5
    min_saccade_length = 30
    max_saccade_length = 90
    min_fixation_length = 30
    max_fixation_dispersion = 2.7
    min_saccade_velocity = 10

    max_vel = 500

    parser = argparse.ArgumentParser()
    parser.add_argument('-target_sampling_rate', '--target_sampling_rate', type=int, default=1000)
    parser.add_argument('-sac_window_size', '--sac_window_size', type=int, default=30)
    parser.add_argument('-fix_window_size', '--fix_window_size', type=int, default=100)
    parser.add_argument('-stimulus', '--stimulus', type=str, default='text')    # video | text | all | fxs | hss | ran

    args = parser.parse_args()
    target_sampling_rate = args.target_sampling_rate
    stimulus = args.stimulus
    if stimulus == 'video':
        use_trial_types = ['VD1','VD2']
    elif stimulus == 'text':
        use_trial_types = ['TEX']
    elif stimulus == 'fxs':
        use_trial_types = ['FXS']
    elif stimulus == 'hss':
        use_trial_types = ['HSS']
    elif stimulus == 'ran':
        use_trial_types = ['RAN']
    elif stimulus == 'all':
        use_trial_types = ['TEX','VD1','VD2','BLG','FXS','HSS','RAN']
    sac_window_size=args.sac_window_size
    fix_window_size=args.fix_window_size



    gaze_data_list, gaze_feature_dict, gaze_label_matrix, gaze_label_dict = data_loader.load_gazebase_data(
        gaze_base_dir=config.GAZE_BASE_DIR,
        use_trial_types=use_trial_types,
        number_train=-1,
        max_round=9,
        target_sampling_rate=target_sampling_rate,
        sampling_rate=sampling_rate,
    )


    event_df_list = []
    list_dicts_list = []
    for i in tqdm(np.arange(len(gaze_data_list)),disable = disable):
        x_dva   = gaze_data_list[i][:,gaze_feature_dict['x_dva_left']]
        y_dva   = gaze_data_list[i][:,gaze_feature_dict['y_dva_left']]
        x_pixel = gaze_data_list[i][:,gaze_feature_dict['x_left_px']]
        y_pixel = gaze_data_list[i][:,gaze_feature_dict['y_left_px']]
        corrupt = np.zeros([len(x_dva),])
        corrupt_ids = np.where(np.logical_or(np.isnan(x_pixel),
                                        np.isnan(y_pixel)))[0]
        corrupt[corrupt_ids] = 1

        # apply smoothing like in https://digital.library.txstate.edu/handle/10877/6874
        smooth_vals = smoothing.smooth_data(x_dva, y_dva,
                                           n=2, smoothing_window_length = smoothing_window_length,
                                           sampling_rate = target_sampling_rate)

        x_smo = smooth_vals['x_smo']
        y_smo = smooth_vals['y_smo']
        vel_x = smooth_vals['vel_x']
        vel_y = smooth_vals['vel_y']
        vel   = smooth_vals['vel']
        acc_x = smooth_vals['acc_x']
        acc_y = smooth_vals['acc_y']
        acc   = smooth_vals['acc']

        corrupt_vels = []
        corrupt_vels += list(np.where(vel_x > max_vel)[0])
        corrupt_vels += list(np.where(vel_x < -max_vel)[0])
        corrupt_vels += list(np.where(vel_y > max_vel)[0])
        corrupt_vels += list(np.where(vel_y < -max_vel)[0])

        corrupt[corrupt_vels] = 1


        # dispersion
        list_dicts, event_df = event_detection.get_sacc_fix_lists_dispersion(
                                                        x_smo, y_smo,
                                                        corrupt = corrupt,
                                                        sampling_rate = target_sampling_rate,
                                                        min_duration = min_fixation_length,
                                                        velocity_threshold = 20,
                                                        flag_skipNaNs = False,
                                                        verbose=0,
                                                        max_fixation_dispersion = max_fixation_dispersion,
                                                        )

        event_df_list.append(event_df)
        list_dicts_list.append(list_dicts)

    print('number of lists: ' + str(len(event_df_list)))


    fixation_list = []
    saccade_list  = []
    for i in tqdm(np.arange(len(event_df_list))):
        list_dicts = list_dicts_list[i]
        event_df   = event_df_list[i]
        fixations  = list_dicts['fixations']
        saccades   = list_dicts['saccades']
        x_dva      = gaze_data_list[i][:,gaze_feature_dict['x_dva_left']]
        y_dva      = gaze_data_list[i][:,gaze_feature_dict['y_dva_left']]
        x_pixel    = gaze_data_list[i][:,gaze_feature_dict['x_left_px']]
        y_pixel    = gaze_data_list[i][:,gaze_feature_dict['y_left_px']]

        # apply smoothing like in https://digital.library.txstate.edu/handle/10877/6874
        smooth_vals = smoothing.smooth_data(x_dva, y_dva,
                                           n=2, smoothing_window_length = smoothing_window_length,
                                           sampling_rate = target_sampling_rate)

        x_smo = smooth_vals['x_smo']
        y_smo = smooth_vals['y_smo']
        vel_x = smooth_vals['vel_x']
        vel_y = smooth_vals['vel_y']
        vel   = smooth_vals['vel']
        acc_x = smooth_vals['acc_x']
        acc_y = smooth_vals['acc_y']
        acc   = smooth_vals['acc']

        for f_i in range(len(fixations)):
            fixation_list.append(np.concatenate([
                    np.expand_dims(x_smo[fixations[f_i]], axis=1),
                    np.expand_dims(y_smo[fixations[f_i]], axis=1),
                    np.expand_dims(x_pixel[fixations[f_i]], axis=1),
                    np.expand_dims(y_pixel[fixations[f_i]], axis=1),
                    np.expand_dims(vel_x[fixations[f_i]], axis=1)/target_sampling_rate,
                    np.expand_dims(vel_y[fixations[f_i]], axis=1)/target_sampling_rate,
                                          ],axis=1))
        for s_i in range(len(saccades)):
            saccade_list.append(np.concatenate([
                    np.expand_dims(x_smo[saccades[s_i]], axis=1),
                    np.expand_dims(y_smo[saccades[s_i]], axis=1),
                    np.expand_dims(x_pixel[saccades[s_i]], axis=1),
                    np.expand_dims(y_pixel[saccades[s_i]], axis=1),
                    np.expand_dims(vel_x[saccades[s_i]], axis=1)/target_sampling_rate,
                    np.expand_dims(vel_y[saccades[s_i]], axis=1)/target_sampling_rate,
                                          ],axis=1))


    print('number of fixations: ' + str(len(fixation_list)))
    print('number of saccades: ' + str(len(saccade_list)))

    filtered_fixation_list = []
    for f_i in tqdm(np.arange(len(fixation_list))):
        cur_x_dva = fixation_list[f_i][:,0]
        cur_y_dva = fixation_list[f_i][:,1]
        x_amp = np.abs(np.max(cur_x_dva) - np.min(cur_x_dva))
        y_amp = np.abs(np.max(cur_y_dva) - np.min(cur_y_dva))
        cur_dispersion = x_amp + y_amp
        if cur_dispersion >= max_fixation_dispersion:
            continue
        if len(cur_x_dva) <= fix_window_size:
            continue
        filtered_fixation_list.append(fixation_list[f_i])
    print('number of fixations after filtering: ' + str(len(filtered_fixation_list)))


    filtered_saccade_list = []
    for s_i in tqdm(np.arange(len(saccade_list))):
        cur_len = saccade_list[s_i].shape[0]
        if cur_len >= sac_window_size:
            x_vels = saccade_list[s_i][:,4]
            y_vels = saccade_list[s_i][:,5]
            x_dva  = saccade_list[s_i][:,0]
            y_dva  = saccade_list[s_i][:,1]
            filtered_saccade_list.append(saccade_list[s_i])
    print('number of saccades after filtering: ' + str(len(filtered_saccade_list)))

    print('number of fixations: ' + str(len(filtered_fixation_list)))
    print('number of saccades:  ' + str(len(filtered_saccade_list)))


    # store fixations and saccades
    column_dict = {'x_dva': 0,
                   'y_dva':1,
                   'x_px':2,
                   'y_px':3,
                   'x_dva_vel':4,
                   'y_dva_vel':5,
                  }

    joblib.dump(column_dict,'data/column_dict.joblib',compress=3,protocol=2)

    fix_lens = [filtered_fixation_list[a].shape[0] for a in range(len(filtered_fixation_list))]
    sac_lens = [filtered_saccade_list[a].shape[0] for a in range(len(filtered_saccade_list))]
    print('fix_lens: ' + str(np.max(fix_lens)))
    print('sacc lens: ' + str(np.max(sac_lens)))

    max_fix_len = fix_window_size
    max_sac_len = sac_window_size
    fixation_matrix = np.ones([len(filtered_fixation_list),max_fix_len,len(column_dict)]) * -1
    saccade_matrix = np.ones([len(filtered_saccade_list),max_sac_len,len(column_dict)]) * -1

    for i in tqdm(np.arange(len(filtered_fixation_list))):
        cur_fix_len = np.min([max_fix_len,filtered_fixation_list[i].shape[0]])
        fixation_matrix[i,0:cur_fix_len,:] = filtered_fixation_list[i][0:cur_fix_len,:]

    for i in tqdm(np.arange(len(filtered_saccade_list))):
        cur_sac_len = np.min([max_sac_len,filtered_saccade_list[i].shape[0]])
        saccade_matrix[i,0:cur_sac_len,:] = filtered_saccade_list[i][0:cur_sac_len,:]


    np.save('data/fixation_matrix_gazebase_vd_' + stimulus,fixation_matrix)
    np.save('data/saccade_matrix_gazebase_vd_' + stimulus,saccade_matrix)

if __name__ == '__main__':
    # execute only if run as a script
    raise SystemExit(main())
