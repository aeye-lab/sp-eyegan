from __future__ import annotations

import os
import pickle
import random
import sys

import joblib
import numpy as np
import pandas as pd
import pymovements as pm
from scipy import interpolate
from tqdm import tqdm


# global params
class_name_mapping = {0:'High Arousal High Valence',
                     1:'Low Arousal High Valence',
                     2:'Low Arousal Low Valence',
                     3:'High Arousal Low Valence',
                     }

arousal_class_mapping = {0:0,
                        1:1,
                        2:1,
                        3:0}
arousal_class_name_mapping = {0:'High Arousal',
                             1:'Low Arousal'}

valence_class_mapping = {0:0,
                        1:0,
                        2:1,
                        3:1}
valence_class_name_mapping = {0:'High Valence',
                             1:'Low Valence'}

channels = {0:'Time stamp',
            1:'Left X',
            2:'Left Y',
            3:'Left Blink',
            4:'Right X',
            5:'Right Y',
            6:'Right Blink'}

rev_channels = {channels[a]:a for a in channels}


# convert pixel to deg
def pix2deg(pix, screenPX,screenCM,distanceCM, adjust_origin=True):
    return pm.transforms.pix2deg(
        arr=pix,
        screen_px=screenPX,
        screen_cm=screenCM,
        distance_cm=distanceCM,
        center_origin=adjust_origin,
    )

# convert pixel to deg
def pix2deg_xy(data, screen_config = {'resolution':[1680,1050],
                     'screen_size':[47.4,29.7],
                     'distance':55.5,
                     'sampling_rate':1000},
                     adjust_origin=True):
    # Converts pixel screen coordinate to degrees of visual angle
    data_copy = data.copy()
    deg = np.zeros(data_copy.shape)
    for i in range(data_copy.shape[2]):
        deg[:,:,i] = pix2deg(data_copy[:,:,i],screen_config['resolution'][i],
                                           screen_config['screen_size'][i],
                                           screen_config['distance'],
                                           adjust_origin = adjust_origin)
    return deg


# convert deg to px
def deg2pix(deg, screenPX,screenCM,distanceCM):
    from math import atan2, degrees
    # Converts degrees of visual angle to pixel screen coordinate
    # screenPX is the number of pixels that the monitor has in the horizontal
    # axis (for x coord) or vertical axis (for y coord)
    # screenCM is the width of the monitor in centimeters
    # distanceCM is the distance of the monitor to the retina
    # pix: screen coordinate in pixels
    # adjust origin: if origin (0,0) of screen coordinates is in the corner of the screen rather than in the center, set to True to center coordinates
    deg=np.array(deg)

    deg_per_px = degrees(atan2(.5*screenCM, distanceCM)) / (.5*screenPX)
    return deg / deg_per_px


# convert deg to px
def deg2pix_xy(data, screen_config = {'resolution':[1680,1050],
                     'screen_size':[47.4,29.7],
                     'distance':55.5,
                     'sampling_rate':1000}):
    from math import atan2, degrees
    # Converts degrees of visual angle to pixel screen coordinate
    data_copy = data.copy()
    pix = np.zeros(data_copy.shape)
    for i in range(data_copy.shape[2]):
        pix[:,:,i] = deg2pix(data_copy[:,:,i],screen_config['resolution'][i],
                                           screen_config['screen_size'][i],
                                           screen_config['distance'])
    return pix

# converts pix to velocities
def pix2vel(data,
           screen_config = {'resolution':[1680,1050],
                     'screen_size':[47.4,29.7],
                     'distance':55.5,
                     'sampling_rate':1000},
            max_vel = None):
    data_copy = data.copy()
    vel = np.zeros(data_copy.shape)
    for i in range(data_copy.shape[0]):
        data_copy[i,:,0] = pix2deg(data_copy[i,:,0],screen_config['resolution'][0],
                                           screen_config['screen_size'][0],
                                           screen_config['distance'])
        data_copy[i,:,1] = pix2deg(data_copy[i,:,1],screen_config['resolution'][1],
                                           screen_config['screen_size'][1],
                                           screen_config['distance'])
        vel[i] = vecvel(data_copy[i],screen_config['sampling_rate'])
    if max_vel is not None:
        vel[vel >= max_vel] = max_vel
        vel[vel <= -max_vel] = -max_vel
    return vel


# conversts deg to velocities
def deg2vel(data,
            screen_config = {'resolution':[1680,1050],
                     'screen_size':[47.4,29.7],
                     'distance':55.5,
                     'sampling_rate':1000},
            smooth = True):

    data_copy = data.copy()
    vel = np.zeros(data_copy.shape)
    for i in range(data_copy.shape[0]):
        vel[i] = vecvel(data_copy[i],screen_config['sampling_rate'])
    return vel

# get all csv files recursively
def get_csvs(start_dir):
    return_list = []
    list_dir = os.listdir(start_dir)
    for i in range(len(list_dir)):
        cur_dir = start_dir + '/' + list_dir[i]
        if os.path.isdir(cur_dir):
            cur_dir += '/'
            return_list += get_csvs(cur_dir)
        else:
            if cur_dir.endswith('.csv'):
                return_list.append(cur_dir)
    return return_list


def transform_to_new_seqlen_length(X, new_seq_len,skip_padded = False):
    """
    Example: if old seq len was 7700, new_seq_len=1000:
    Input X has: 144 x 7700 x n_channels
    Output X has: 144*8 x 1000 x n_channels
    The last piece of each trial 7000-7700 gets padded with first 300 of this piece to be 1000 long
    :param X:
    :param new_seq_len:
    :return:
    """
    n, rest = np.divmod(X.shape[1], new_seq_len)

    if rest > 0 and not skip_padded:
        n_rows = X.shape[0]*(n+1)
    else:
        n_rows = X.shape[0]*n

    X_new = np.nan * np.ones((n_rows, new_seq_len, X.shape[2]))

    idx = 0
    for t in range(0, X.shape[0]):
        for i in range(0, n):
            # cut out 1000 ms piece of trial t
            X_tmp = np.expand_dims(X[t, i*new_seq_len: (i+1)*new_seq_len, :], axis=0)

            # concatenate pieces
            X_new[idx, :, :] = X_tmp

            idx = idx + 1

        if rest > 0 and not skip_padded:
            # concatenate last one with pad
            start_idx_last_piece = new_seq_len*(n)
            len_pad_to_add = new_seq_len-rest
            # piece to pad:
            X_incomplete = np.expand_dims(X[t, start_idx_last_piece:X.shape[1], :], axis=0)
            # padding piece:
            start_idx_last_piece = new_seq_len*(n-1)
            X_pad = np.expand_dims(X[t, start_idx_last_piece:start_idx_last_piece+len_pad_to_add, :], axis=0)


            X_tmp = np.concatenate((X_incomplete, X_pad), axis=1)

            # concatenate last piece of original row t
            X_new[idx, :, :] = X_tmp

            idx = idx + 1

    seq_len = new_seq_len
    #print(X_new.shape)
    assert np.sum(np.isnan(X_new[:, :, 0])) == 0, 'Cutting into pieces failed, did not fill each position of new matrix.'

    return X_new


def get_data_for_user(
    data_path, max_vel = 500,delete_nans = True,
    sampling_rate = 1000, smooth = True,
    delete_high_velocities = False,
    output_length = None,
    transform_deg_to_px = False,
    min_max_transform_px = True,
    screen_config = {'resolution':[1680,1050],
                     'screen_size':[47.4,29.7],
                     'distance':55.5},
    target_sampling_rate = None,
    ):


    if output_length is None:
        output_length = sampling_rate
    cur_data = pd.read_csv(data_path)
    if target_sampling_rate is not None:
        x_vals  = np.array(cur_data['x'])
        y_vals  = np.array(cur_data['y'])
        times   = np.array(cur_data['n'])
        val     = np.array(cur_data['val'])
        if delete_nans:
            x_vals[val != 0] = np.nan
            y_vals[val != 0] = np.nan
            not_nan_ids = np.logical_and(~np.isnan(x_vals),~np.isnan(y_vals))
            x_vals = x_vals[not_nan_ids]
            y_vals = y_vals[not_nan_ids]
            times = times[not_nan_ids]

        cur_interpolate_x = interpolate.interp1d(times, x_vals)
        cur_interpolate_y = interpolate.interp1d(times, y_vals)

        step_size = sampling_rate / target_sampling_rate
        use_time_stamps = np.arange(np.min(times),np.max(times),step_size)
        x_dva_gazebase = cur_interpolate_x(use_time_stamps)
        y_dva_gazebase = cur_interpolate_y(use_time_stamps)
        X = np.array([
            x_dva_gazebase,
            y_dva_gazebase,
        ]).T
        sampling_rate = target_sampling_rate
    else:
        X = np.array([
            cur_data['x'],
            cur_data['y'],
        ]).T

        X[np.array(cur_data['val']) != 0,:] = np.nan
        if delete_nans:
            not_nan_ids = np.logical_and(~np.isnan(X[:,0]),~np.isnan(X[:,1]))
            X = X[not_nan_ids,:]



    # transform deg to pix
    if transform_deg_to_px:
        X_px = X.copy()
        X_px[:,0] = deg2pix(X_px[:,0],
                    screen_config['resolution'][0],
                    screen_config['screen_size'][0],
                    screen_config['distance'])
        # adjust origin
        X_px[:,0] += screen_config['resolution'][0]/2

        X_px[:,1] = deg2pix(X_px[:,1],
                    screen_config['resolution'][1],
                    screen_config['screen_size'][1],
                    screen_config['distance'])
        # adjust origin
        X_px[:,1] += screen_config['resolution'][1]/2
    else:
        X_px = np.zeros(X.shape)

    # transform to velocities
    vel_left = vecvel(X, sampling_rate)
    vel_left[vel_left > max_vel] = max_vel
    vel_left[vel_left < -max_vel] = -max_vel
    if delete_high_velocities:
        not_high_velocity_ids = np.logical_or(
            np.abs(vel_left[:,0]) >= max_vel,
            np.abs(vel_left[:,1]) >= max_vel,
        )
        X = X[not_high_velocity_ids]
        vel_left = vel_left[not_high_velocity_ids]
    X_vel = vel_left

    X_vel_transformed = transform_to_new_seqlen_length(
        X = np.reshape(X_vel,[1,X_vel.shape[0],X_vel.shape[1]]),
        new_seq_len = output_length,
        skip_padded=True,
    )

    X_deg_transformed = transform_to_new_seqlen_length(
        X = np.reshape(X,[1,X.shape[0],X.shape[1]]),
        new_seq_len = output_length,
        skip_padded=True,
    )

    X_px_transformed = transform_to_new_seqlen_length(
        X = np.reshape(X_px,[1,X_px.shape[0],X_px.shape[1]]),
        new_seq_len = output_length,
        skip_padded=True,
    )

    user_dict = {
        'X':X,
        'X_deg':X,
        'X_deg_transformed':X_deg_transformed,
        'X_px':X_px,
        'X_px_transformed':X_px_transformed,
        'X_vel':X_vel,
        'X_vel_transformed':X_vel_transformed,
        'path':data_path,

    }
    return user_dict

def load_gazebase_data(gaze_base_dir = 'path_to_gazebase_data',
                            use_trial_types = ['TEX','RAN'],#,'BLG','FXS','VD1','VD2','HSS']
                            number_train = -1,
                            max_round = 9,
                            target_sampling_rate = 60,
                            sampling_rate = 1000
                            ):
    if os.path.exists(gaze_base_dir + 'Round_9/'):
        gaze_base_raw_data_dirs = [gaze_base_dir + 'Round_9/',
                        gaze_base_dir + 'Round_8/',
                        gaze_base_dir + 'Round_7/',
                        gaze_base_dir + 'Round_6/',
                        gaze_base_dir + 'Round_5/',
                        gaze_base_dir + 'Round_4/',
                        gaze_base_dir + 'Round_3/',
                        gaze_base_dir + 'Round_2/',
                        gaze_base_dir + 'Round_1/']
    else:
        gaze_base_raw_data_dirs = [gaze_base_dir + '/raw/Round_9/',
                        gaze_base_dir + '/raw/Round_8/',
                        gaze_base_dir + '/raw/Round_7/',
                        gaze_base_dir + '/raw/Round_6/',
                        gaze_base_dir + '/raw/Round_5/',
                        gaze_base_dir + '/raw/Round_4/',
                        gaze_base_dir + '/raw/Round_3/',
                        gaze_base_dir + '/raw/Round_2/',
                        gaze_base_dir + '/raw/Round_1/']

    csv_files = []
    for cur_dir in gaze_base_raw_data_dirs:
        csv_files += get_csvs(cur_dir)

    round_list    = []
    subject_list  = []
    session_list  = []
    trial_list    = []
    path_list     = []
    use_for_train = []
    for csv_file in csv_files:
        #print(csv_file)
        file_name = csv_file.split('/')[-1]
        #print(file_name)
        file_name_split = file_name.replace('.csv','').split('_')
        cur_round = file_name_split[1][0]
        cur_subject = int(file_name_split[1][1:])
        cur_session = file_name_split[2]
        cur_trial = file_name_split[3]
        if cur_trial not in use_trial_types:
            continue
        if int(cur_round) > max_round:
            continue
        use_for_train.append(1)
        round_list.append(cur_round)
        subject_list.append(cur_subject)
        session_list.append(cur_session)
        trial_list.append(cur_trial)
        path_list.append(csv_file)

    data_csv = pd.DataFrame({'round':round_list,
                        'subject':subject_list,
                        'session':session_list,
                        'trial':trial_list,
                        'path':path_list})

    user_data_list = []
    sub_id_list    = []
    for i in tqdm(range(len(data_csv))):
        cur_line = data_csv.iloc[i]
        cur_path = cur_line['path']
        try:
            cur_sub_id = int(cur_path.split('/')[-1].split('_')[1][1:])
            cur_data = get_data_for_user(cur_path,
                                                    smooth = True,
                                                    output_length = None,
                                                    transform_deg_to_px = True,
                                                    min_max_transform_px = True,
                                                    screen_config = {'resolution':[1680,1050],
                                                                     'screen_size':[47.4,29.7],
                                                                     'distance':55.5},
                                                    target_sampling_rate = target_sampling_rate,
                                                    sampling_rate = sampling_rate,
                                                    )
            add_data_dict = {'x_vel_dva_left':cur_data['X_vel'][:,0],
                             'y_vel_dva_left':cur_data['X_vel'][:,1],
                             'x_left_px':cur_data['X_px'][:,0],
                             'y_left_px':cur_data['X_px'][:,1],
                             'x_dva_left':cur_data['X_deg'][:,0],
                             'y_dva_left':cur_data['X_deg'][:,1],
                            }

            feature_dict = {'x_vel_dva_left':0,
                             'y_vel_dva_left':1,
                             'x_left_px':2,
                             'y_left_px':3,
                             'x_dva_left':4,
                             'y_dva_left':5,
                           }

            cur_data_matrix = np.zeros([cur_data['X_vel'].shape[0],
                                       len(add_data_dict.keys())])

            counter = 0
            for key in add_data_dict.keys():
                cur_data_matrix[:,counter] = add_data_dict[key]
                counter += 1
            user_data_list.append(cur_data_matrix)
            sub_id_list.append(cur_sub_id)
        except:
            print('error with file: ' + str(cur_path))

    Y = np.zeros([len(sub_id_list),1])
    Y[:,0] = np.array(sub_id_list)
    y_column_dict = {'subject_id':0,
                    }
    return user_data_list, feature_dict, Y, y_column_dict




# Compute velocity times series from 2D position data
# returns velocity in deg/sec or pix/sec
def vecvel(x, sampling_rate = 1):
    # sanity check: horizontal and vertical gaze coordinates missing values at the same time (Eyelink eyetracker never records only one coordinate)
    assert np.array_equal(np.isnan(x[:,0]), np.isnan(x[:,1]))
    N = x.shape[0]
    v = np.zeros((N,2)) # first column for x-velocity, second column for y-velocity
    v[1:N,] = sampling_rate * (x[1:N,:] - x[0:N-1,:])
    return v

def vecvel_vector(x, sampling_rate = 1):
    N = len(x)
    v = np.zeros([N,])
    v[1:N] = sampling_rate * (x[1:N] - x[0:N-1])
    return v


from scipy import interpolate

# data: vector containing the values
# timestamp: vector containing the time stamps
# channel_name: channel that should be interpolated
# sampling_rate: sampling rate we want to interpolate
# flag_0_to_nan: flag, whether we want to set all 0 values to np.nan
def interpolate_channel(data,timestamp,sampling_rate,flag_0_to_nan = True):
    # set values of 0 to nan
    if flag_0_to_nan:
        data[np.isnan(data)] = np.nan
    inter_data = interpolate.interp1d(timestamp,data)
    return inter_data(np.arange(timestamp[0],timestamp[-1],1000./sampling_rate))


# data: dictionary containing the data
# sampling_rate: sampling rate of data
def preprocess_data(data,sampling_rate):
    sample_time = data[rev_channels['Time stamp'],:]
    x_coord = data[rev_channels['Left X'],:]
    x_coord[x_coord == 0] = np.nan
    y_coord = data[rev_channels['Left Y'],:]
    y_coord[y_coord == 0] = np.nan

    cur_inter_x = interpolate.interp1d(sample_time,x_coord)
    cur_inter_y = interpolate.interp1d(sample_time,y_coord)



    left_x = data[rev_channels['Left X'],:]
    left_x[np.isnan(left_x)] = np.nan
    left_y = data[rev_channels['Left Y'],:]
    left_y[np.isnan(left_y)] = np.nan
    right_x = data[rev_channels['Right X'],:]
    right_x[np.isnan(right_x)] = np.nan
    right_y = data[rev_channels['Right Y'],:]
    right_y[np.isnan(right_y)] = np.nan
    time_stamp = data[rev_channels['Time stamp'],:]

    x_left_coord_inter = interpolate_channel(left_x,time_stamp,sampling_rate,flag_0_to_nan = True)
    x_left_vel = vecvel_vector(x_left_coord_inter) * sampling_rate
    y_left_coord_inter = interpolate_channel(left_y, time_stamp,sampling_rate,flag_0_to_nan = True)
    y_left_vel = vecvel_vector(y_left_coord_inter) * sampling_rate
    x_right_coord_inter = interpolate_channel(right_x,time_stamp,sampling_rate,flag_0_to_nan = True)
    x_right_vel = vecvel_vector(x_right_coord_inter) * sampling_rate
    y_right_coord_inter = interpolate_channel(right_y,time_stamp,sampling_rate,flag_0_to_nan = True)
    y_right_vel = vecvel_vector(y_right_coord_inter) * sampling_rate
    x_pix_diff  = np.abs(x_left_coord_inter - x_right_coord_inter)
    y_pix_diff  = np.abs(y_left_coord_inter - y_right_coord_inter)


    x_dva_left, y_dva_left = convert_gaze_to_dva(x_left_coord_inter,y_left_coord_inter)
    x_vel_dva_left, y_vel_dva_left = vecvel_vector(x_dva_left) * sampling_rate, vecvel_vector(y_dva_left) * sampling_rate
    x_dva_right, y_dva_right = convert_gaze_to_dva(x_right_coord_inter,y_right_coord_inter)
    x_vel_dva_right, y_vel_dva_right = vecvel_vector(x_dva_right) * sampling_rate, vecvel_vector(y_dva_right) * sampling_rate




    corrupt = np.zeros(x_left_coord_inter.shape)
    corrupt[x_left_coord_inter == 0] = 1
    corrupt[x_right_coord_inter == 0] = 1
    corrupt[y_left_coord_inter == 0] = 1
    corrupt[y_right_coord_inter == 0] = 1

    return (np.swapaxes(np.vstack([x_left_vel,y_left_vel,
                x_right_vel,y_right_vel,
                x_pix_diff,y_pix_diff,
                x_left_coord_inter,  y_left_coord_inter,
                x_right_coord_inter, y_right_coord_inter,
                x_dva_left, y_dva_left,
                x_dva_right,y_dva_right,
                x_vel_dva_left,y_vel_dva_left,
                x_vel_dva_right, y_vel_dva_right,
                corrupt]),axis1=0,axis2=1),
                {'x_left_vel':0,
                 'y_left_vel':1,
                 'x_right_vel':2,
                 'y_right_vel':3,
                 'x_pix_diff':4,
                 'y_pix_diff':5,
                 'x_left_px':6,
                 'y_left_px':7,
                 'x_right_px':8,
                 'y_right_px':9,
                 'x_dva_left':10,
                 'y_dva_left':11,
                 'x_dva_right':12,
                 'y_dva_right':13,
                 'x_vel_dva_left':14,
                 'y_vel_dva_left':15,
                 'x_vel_dva_right':16,
                 'y_vel_dva_right':17,
                 'corrupt':18,
                }
           )




# function to cut data matrix into pieces
#    inputs:
#       data: a 'm x n' matrix
#       window_size: seconds for each window (assuming times to be milliseconds)
#       padding: flag, whether to pad the sequences or not
#       padding_value: value used to pad sequences
#    output:
#       output_data: a matrix of '? x window_size x n'
def cut_into_sequences(data,
                        window_size = 600,
                        padding = False,
                        padding_value = 0):
    if padding == False:
        num_instances = int(np.floor((data.shape[0]) / window_size))
        out_matrix    = np.ones([num_instances,window_size,data.shape[1]]) * 0
    else:
        num_instances = int(np.ceil((data.shape[0]) / window_size))
        out_matrix    = np.ones([num_instances,window_size,data.shape[1]]) * padding_value
    for f_i in range(num_instances):
        out_matrix[f_i,
                   0:data[f_i * window_size:(f_i+1)*window_size].shape[0],:] = data[f_i * window_size:(f_i+1)*window_size]
    return out_matrix



# standard normalize data
#   inputs:
#      data: data matrix of shape: (#instances x #time-steps x #num-features)
def standard_normalize_data(data_train,
                           data_test = None):
    out_data_train = np.zeros(data_train.shape)
    if data_test is not None:
        out_data_test = np.zeros(data_test.shape)
    if len(data_train.shape) == 3:
        for i in range(data_train.shape[2]):
            mean_val = np.mean(data_train[:,:,i])
            std_val = np.std(data_train[:,:,i])
            out_data_train[:,:,i] = (data_train[:,:,i] - mean_val) / (np.finfo(np.float32).eps + (std_val))
            if data_test is not None:
                out_data_test[:,:,i] = (data_test[:,:,i] - mean_val) / (np.finfo(np.float32).eps + (std_val))
    elif len(data_train.shape) == 2:
        for i in range(data_train.shape[1]):
            mean_val = np.mean(data_train[:,i])
            std_val = np.std(data_train[:,i])
            out_data_train[:,i] = (data_train[:,i] - mean_val) / (np.finfo(np.float32).eps + (std_val))
            if data_test is not None:
                out_data_test[:,i] = (data_test[:,i] - mean_val) / (np.finfo(np.float32).eps + (std_val))
    if data_test is None:
        return out_data_train
    else:
        return out_data_train, out_data_test
