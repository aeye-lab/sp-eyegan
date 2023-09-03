from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd


# get list of saccade-ids and fixation-ids with dispersion algorithm
def get_sacc_fix_lists_dispersion(x_deg, y_deg,
                        corrupt = None,
                        sampling_rate = 1000,
                        min_duration = 80,
                        velocity_threshold = 20,
                        min_event_duration_fixation = 50,
                        min_event_duration_saccade = 10,
                        flag_skipNaNs = True,
                        verbose=0,
                        max_fixation_dispersion = None,
                                 ):

    #  fix=1, saccade=2, corrupt=3
    events = get_i_dt(x_deg, y_deg,
                        corrupt = corrupt,
                        sampling = sampling_rate,
                        min_duration = min_duration,
                        velocity_threshold = velocity_threshold,
                        min_event_duration_fixation = min_event_duration_fixation,
                        min_event_duration_saccade = min_event_duration_saccade,
                        flag_skipNaNs = flag_skipNaNs,
                        verbose=0,
                        max_fixation_dispersion = None,
                                     )

    #  fix=1, saccade=2, corrupt=3
    event_list = np.array(events['event'])
    prev_label = -1
    fixations = []
    saccades = []
    errors = []
    for i in range(len(event_list)):
        cur_label = event_list[i]
        if cur_label != prev_label:
            if prev_label != -1:
                if prev_label == 1:
                    fixations.append(cur_list)
                elif prev_label == 2:
                    saccades.append(cur_list)
                else:
                    errors.append(cur_list)
            cur_list = [i]
        else:
            cur_list.append(i)
        prev_label = cur_label

    if len(cur_list) > 0:
        if prev_label == 1:
            fixations.append(cur_list)
        elif prev_label == 2:
            saccades.append(cur_list)
        else:
            errors.append(cur_list)
    return {'fixations': fixations,
            'saccades': saccades,
            'errors': errors}, events







'''
Eye movements were processed with the biometric
framework described in Section 2, with eye movement
classification thresholds: velocity threshold of 20°/sec,
micro-saccade threshold of 0.5°, and micro-fixation
threshold of 100 milliseconds. Feature extraction was
performed across all eye movement recordings, while
matching and information fusion were performed
according to the methods described in Section 3"

source: https://www.researchgate.net/publication/220811146_Identifying_fixations_and_saccades_in_eye-tracking_protocols

The I-DT algorithm requires two parameters, the dispersionthreshold
and the duration threshold.  Like the velocitythreshold for I-VT,
 the dispersion threshold can be set toinclude 1/2° to 1° of visual
 angle if the distance from eye toscreen is known.  Otherwise, the
 dispersion threshold can beestimated from exploratory analysis of
 the data.  The durationthreshold is typically set to a value between
 100 and 200 ms[21], depending on task processing demands.

Identifying fixations and saccades in eye-tracking protocols
Dario Salvucci, H. Goldberg

'''
#
# input:
#           x_coordinates: degrees of visual angle in x-axis
#           y_coordinates: degrees of visual angle in y-axis
#
# output:
#           d: data-frame containing the saccade label
def get_i_dt(x_coordinates,y_coordinates,
            corrupt = None,
            sampling = 1000,
            min_duration = 80,
            velocity_threshold = 20,
            min_event_duration_fixation = 50,
            min_event_duration_saccade = 10,
            flag_skipNaNs = True,
            verbose=0,
            max_fixation_dispersion = None,
            ):

    duration_threshold = int(np.floor((min_duration / 1000.) * sampling))
    min_duration_threshold_fixation = np.max([1,int(np.floor((min_event_duration_fixation / 1000.) * sampling))])
    min_duration_threshold_saccade = np.max([1,int(np.floor((min_event_duration_saccade / 1000.) * sampling))])
    if max_fixation_dispersion is None:
        dispersion_threshold = (velocity_threshold / 1000. * min_duration)
    else:
        dispersion_threshold = max_fixation_dispersion

    d = { 'x_deg':x_coordinates,
          'y_deg':y_coordinates}
    d = pd.DataFrame(d)

    sacc = np.ones([len(x_coordinates),])
    start_id = 0
    end_id   = start_id + duration_threshold
    previous_dispension = 100 * dispersion_threshold
    counter = 0
    while start_id <= len(x_coordinates):
        cur_x_window = x_coordinates[start_id:end_id]
        cur_y_window = y_coordinates[start_id:end_id]
        # skip NaNs
        if flag_skipNaNs:
            cur_use_ids = np.logical_and(np.isnan(cur_x_window) == False,
                                        np.isnan(cur_y_window) == False)
            cur_x_window = cur_x_window[cur_use_ids]
            cur_y_window = cur_y_window[cur_use_ids]
        else:
            cur_use_ids = np.logical_and(np.isnan(cur_x_window),
                                        np.isnan(cur_y_window))
            cur_x_window[cur_use_ids] = 100 * dispersion_threshold
            cur_y_window[cur_use_ids] = 100 * dispersion_threshold
        if len(cur_x_window) > 0:
            cur_dispersion = (np.max(cur_x_window) - np.min(cur_x_window)) +\
                            (np.max(cur_y_window) - np.min(cur_y_window))
        else:
            cur_dispersion = 100* dispersion_threshold
        #print('x_coordintes: ' + str(cur_x_window))
        #print('y_coordintes: ' + str(cur_y_window))
        #print('cur_dispersion: ' + str(cur_dispersion))

        if cur_dispersion <= dispersion_threshold and end_id <= len(x_coordinates):
            end_id += 1
            #print('start_id: ' + str(start_id))
            #print('end_id: ' + str(end_id))
            #print(allo)
        else:
            if previous_dispension <= dispersion_threshold:
                sacc[start_id:end_id-1] = 0
                start_id = end_id
                end_id = start_id + duration_threshold
            else:
                start_id += 1
                end_id += 1
        previous_dispension = cur_dispersion
        counter += 1
        if verbose:
            if counter % 1000 == 0:
                print(counter)
    sacc[np.isnan(d['x_deg'])] = 3

    d['sac'] = sacc

    if corrupt is None:
        nan_ids_x = list(np.where(np.isnan(x_coordinates))[0])
        nan_ids_y = list(np.where(np.isnan(y_coordinates))[0])
        corruptIdx = list(set(nan_ids_x + nan_ids_y))
    else:
        corruptIdx = list(np.where(corrupt == 1)[0])
        nan_ids_x = list(np.where(np.isnan(x_coordinates))[0])
        nan_ids_y = list(np.where(np.isnan(y_coordinates))[0])
        corruptIdx = list(set(corruptIdx + nan_ids_x + nan_ids_y))

    # corrupt samples
    d['corrupt'] = d.index.isin(corruptIdx)
    d['event'] = np.where(d.corrupt, 3, np.where(d.sac,2,1))
    return d
