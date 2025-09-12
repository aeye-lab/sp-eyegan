# written by Paul Prasse

import numpy as np
from scipy.spatial import distance as distance
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import warnings
from sp_eyegan.model.preprocessing import event_detection as event_detection
from sp_eyegan.model.preprocessing import curve_metrics as curve_metrics
from sp_eyegan.model.preprocessing import smoothing as smoothing


# get list of saccade-ids and fixation-ids
def get_sacc_fix_lists(x_deg, y_deg,
                       corrupt = None,
                       pp_params=(None, 6),
                       minDurFix=20, minDurSac=6,
                       sampling=1000,
                       input_velocity=False):
    events = event_detection.detect_saccades(x_deg, y_deg, corrupt = corrupt,
                                             pp_params=pp_params, minDurFix=minDurFix,
                                             minDurSac=minDurSac, sampling=sampling,
                                             input_velocity=input_velocity)

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
    events = event_detection.get_i_dt(x_deg, y_deg,
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


## get blink events for the list of eye closures
# params: 
#           eye_closures: 1-dimensional vector of eye closures
#           blink_theshold :consider value above as blink (only important if flag_use_eye_state_label is False)
# returns:
#   list of lists, where each sublist contains the indexes of the blink
def get_blink_events_from_eye_closures(eye_closures, blink_threshold=0.6):
    blink_events = []
    prev_label = False
    cur_blink_event = []
    for i in range(len(eye_closures)):
        cur_label = False
        if eye_closures[i] >= blink_threshold:
            cur_label = True

        if cur_label == True:
            cur_blink_event.append(i)
        elif prev_label == False and cur_label == False:
            continue
        elif prev_label == True and cur_label == False:
            blink_events.append(cur_blink_event)
            cur_blink_event = []
        prev_label = cur_label
    if len(cur_blink_event) > 0:
        blink_events.append(cur_blink_event)
    return blink_events


## get blink events for the list of eye states
# params: 
#           eye_states: 1-dimensional vector of eye states
#           close_labels : list of labels considered to be a closed eye
# returns:
#   list of lists, where each sublist contains the indexes of the blink
def get_blink_events_from_eye_states(eye_states, close_labels=[1]):
    blink_events = []
    prev_label = False
    cur_blink_event = []
    for i in range(len(eye_states)):
        cur_label = False
        if eye_states[i] in close_labels:
            cur_label = True

        if cur_label == True:
            cur_blink_event.append(i)
        elif prev_label == False and cur_label == False:
            continue
        elif prev_label == True and cur_label == False:
            blink_events.append(cur_blink_event)
            cur_blink_event = []
        prev_label = cur_label
    if len(cur_blink_event) > 0:
        blink_events.append(cur_blink_event)
    return blink_events


# creates a feature for a list of values (e.g. mean or standard deviation of values in list)
# params:
#       values: list of values
#       aggregation_function: name of function to be applied to list
# returns:
#       aggregated value
def get_feature_from_list(values, aggregation_function):
    if np.sum(np.isnan(values)) == len(values):
        return np.nan
    if aggregation_function == 'mean':
        return np.nanmean(values)
    elif aggregation_function == 'std':
        return np.nanstd(values)
    elif aggregation_function == 'median':
        return np.nanmedian(values)
    elif aggregation_function == 'skew':
        not_nan_values = np.array(values)[~np.isnan(values)]
        return skew(not_nan_values)
    elif aggregation_function == 'kurtosis':
        not_nan_values = np.array(values)[~np.isnan(values)]
        return kurtosis(not_nan_values)
    else:
        return np.nan


###################################################################
#
#   Implementation of features from 'The Accuracy of Eyelid Movement Parameters for Drowsiness Detection' by Wilkinson et. al
#           https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3836343/
#
###################################################################

## compute %LC
# Percent Long Closures (%LC): proportion of time eyes are fully closed > 10 ms.
# params: 
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]
#           blink_threshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           min_duration: minimal duration for a long blink
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           close_labels : list of labels considered to be a closed eye
def get_percentage_long_closures(eye_closures, eye_states,
                                 blink_threshold=0.6,
                                 min_duration=10,
                                 flag_use_eye_state_label=True,
                                 close_labels=[1]):
    if flag_use_eye_state_label:
        blink_events = get_blink_events_from_eye_states(eye_states, close_labels=close_labels)
    else:
        blink_events = get_blink_events_from_eye_closures(eye_closures, blink_threshold=blink_threshold)

    long_closure_sum = 0
    for blink_event in blink_events:
        if len(blink_event) >= min_duration:
            long_closure_sum += len(blink_event)
    return float(long_closure_sum / len(eye_closures))


## compute IED
# Inter-Event Duration (IED): blink duration measured from the point of maximum closing velocity to maximum opening velocity of the eyelid measured in seconds.
# params: 
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]
#           blink_threshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           window_size: number of time steps before and after blink to consider to get maximum velocity
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           close_labels : list of labels considered to be a closed eye
# return:
#           list of inter event durations
def get_inter_event_durations(eye_closures, eye_states,
                              blink_threshold=0.6,
                              window_size=100,
                              flag_use_eye_state_label=True,
                              close_labels=[1]):
    if flag_use_eye_state_label:
        blink_events = get_blink_events_from_eye_states(eye_states, close_labels=close_labels)
    else:
        blink_events = get_blink_events_from_eye_closures(eye_closures, blink_threshold=blink_threshold)

    # compute velocities
    prev_eye_closure = np.zeros(len(eye_closures))
    prev_eye_closure[1:] = eye_closures[:-1]
    velocities = eye_closures - prev_eye_closure
    velocities[0] = 0

    inter_event_durations = []
    for blink_event in blink_events:
        blink_close_velocities = velocities[np.max([0, blink_event[0] - window_size + 1]):blink_event[0] + 1]
        blink_open_velocities = velocities[blink_event[-1] + 1:blink_event[-1] + window_size + 1]
        if len(blink_close_velocities) == 0 or len(blink_open_velocities) == 0:
            inter_event_duration = np.nan
        else:
            max_close_vel_id = np.argmax(blink_close_velocities)
            max_open_vel_id = np.argmin(blink_open_velocities)
            inter_event_duration = (window_size - max_close_vel_id - 1) + len(blink_event) + max_open_vel_id

        inter_event_durations.append(inter_event_duration)
    return inter_event_durations


## compute TEC
#    * Percent Time with Eyes Closed (%TEC): Percentage of time that the eyes are deemed closed in each minute. The eyes are deemed closed when the velocity of the eyelid movement following the closing of the eyelid drops below the velocity threshold, and is deemed closed until the velocity increases back above this threshold indicating the beginning of the re-opening of the eyelids.
# params:    
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]blink_theshold :consider value above as blink
#           blink_theshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           window_size: number of time steps before and after blink to consider to get maximum velocity
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           close_labels : list of labels considered to be a closed eye
#           velocity_threshold: specify the threshold for the velocity, where the blink starts / ends
def get_percent_time_with_eyes_closed(eye_closures, eye_states,
                                      blink_threshold=0.6, velocity_threshold=0.1,
                                      window_size=100,
                                      flag_use_eye_state_label=True,
                                      close_labels=[1]):
    if flag_use_eye_state_label:
        blink_events = get_blink_events_from_eye_states(eye_states, close_labels=close_labels)
    else:
        blink_events = get_blink_events_from_eye_closures(eye_closures, blink_threshold=blink_threshold)

    # compute velocities
    prev_eye_closure = np.zeros(len(eye_closures))
    prev_eye_closure[1:] = eye_closures[:-1]
    velocities = eye_closures - prev_eye_closure
    velocities[0] = 0

    time_eyes_closed = []
    for blink_event in blink_events:
        start_id = np.max([blink_event[0] - window_size, 0])
        end_id = np.min([blink_event[-1] + window_size + 1, len(eye_closures)])
        blink_ids = np.arange(start_id, end_id, 1)

        blink_velocities = velocities[blink_ids]
        try:
            start_id = np.min(np.where(blink_velocities >= velocity_threshold)[0])
        except:
            start_id = blink_event[0]
        try:
            end_id = np.max(np.where(blink_velocities <= -velocity_threshold)[0])
        except:
            end_id = blink_event[-1]
        cur_time = end_id - start_id
        time_eyes_closed.append(cur_time)
    return np.sum(time_eyes_closed) / (len(eye_closures) + np.finfo(np.float32).eps)


## compute BTD
# *Blink Total Duration (BTD): duration of blinks measured in seconds from the start of closing to complete re-opening.
# params:    
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]blink_threshold :consider value above as blink
#           blink_threshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           close_labels : list of labels considered to be a closed eye
# returns:
#           list of total blink durations
def get_blink_total_duration(eye_closures, eye_states,
                             blink_threshold=0.6,
                             flag_use_eye_state_label=True,
                             close_labels=[1]):
    return get_blink_durations(eye_closures, eye_states, blink_threshold, flag_use_eye_state_label, close_labels)


# compute –AVR
# * Negative Amplitude-Velocity Ratio (–AVR): the ratio of the maximum amplitude to maximum velocity of eyelid movement for the reopening phase of blinks.
# params:    
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]
#           blink_threshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           window_size: number of time steps before and after blink to consider to get maximum velocity
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           close_labels : list of labels considered to be a closed eye
# returns:
#           list of negative amplitude velocities
def get_negative_amplitude_velocity_ratio(eye_closures, eye_states,
                                          blink_threshold=0.6, window_size=100,
                                          flag_use_eye_state_label=True,
                                          close_labels=[1]):
    if flag_use_eye_state_label:
        blink_events = get_blink_events_from_eye_states(eye_states, close_labels=close_labels)
    else:
        blink_events = get_blink_events_from_eye_closures(eye_closures, blink_threshold=blink_threshold)

    prev_eye_closure = np.zeros(len(eye_closures))
    prev_eye_closure[1:] = eye_closures[:-1]
    velocities = eye_closures - prev_eye_closure
    velocities[0] = 0

    negative_amplitude_velocity_ratios = []
    for blink_event in blink_events:
        reopening_ids = np.arange(np.min([blink_event[-1], len(eye_closures) - 1]),
                                  np.min([blink_event[-1] + window_size, len(eye_closures) - 1]), 1) + 1
        try:
            max_amplitude = np.max(eye_closures[reopening_ids]) - np.min(eye_closures[reopening_ids])
            min_velocities = np.min(velocities[reopening_ids])
            amp_vel_ratio = max_amplitude / (np.abs(min_velocities) + np.finfo(np.float32).eps)
            negative_amplitude_velocity_ratios.append(amp_vel_ratio)
        except:
            negative_amplitude_velocity_ratios.append(np.nan)
    return negative_amplitude_velocity_ratios


# compute +AVR
# * Positive Amplitude-Velocity Ratio (+AVR): the ratio of the maximum amplitude to maximum velocity of eyelid movement for the closing phase of blinks.
# params:    
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]blink_theshold :consider value above as blink
#           blink_threshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           window_size: number of time steps before and after blink to consider to get maximum velocity
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           close_labels : list of labels considered to be a closed eye
# returns:
#           list of positive amplitude velocities
def get_positive_amplitude_velocity_ratio(eye_closures, eye_states,
                                          blink_threshold=0.6, window_size=100,
                                          flag_use_eye_state_label=True,
                                          close_labels=[1]):
    if flag_use_eye_state_label:
        blink_events = get_blink_events_from_eye_states(eye_states, close_labels=close_labels)
    else:
        blink_events = get_blink_events_from_eye_closures(eye_closures, blink_threshold=blink_threshold)

    prev_eye_closure = np.zeros(len(eye_closures))
    prev_eye_closure[1:] = eye_closures[:-1]
    velocities = eye_closures - prev_eye_closure
    velocities[0] = 0

    positive_amplitude_velocity_ratios = []
    for blink_event in blink_events:
        closing_ids = np.arange(blink_event[0] - window_size + 1, blink_event[0] + 1, 1)

        max_amplitude = np.max(eye_closures[closing_ids]) - np.min(eye_closures[closing_ids])
        max_velocities = np.max(velocities[closing_ids])
        amp_vel_ratio = max_amplitude / (np.abs(max_velocities) + np.finfo(np.float32).eps)
        positive_amplitude_velocity_ratios.append(amp_vel_ratio)
    return positive_amplitude_velocity_ratios


##################################################################
#
#   Implementation of features presented in 'Blinks and saccades as indicators of fatigue in sleepiness warnings: looking tired?' by Schleicher et. al
#       https://www.tandfonline.com/doi/full/10.1080/00140130701817062
#
##################################################################

# compute blink durations
# params:    
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]blink_theshold :consider value above as blink
#           blink_threshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           close_labels : list of labels considered to be a closed eye
# returns:
#           list of blink durations
def get_blink_durations(eye_closures, eye_states,
                        blink_threshold=0.6,
                        flag_use_eye_state_label=True,
                        close_labels=[1]):
    if flag_use_eye_state_label:
        blink_events = get_blink_events_from_eye_states(eye_states, close_labels=close_labels)
    else:
        blink_events = get_blink_events_from_eye_closures(eye_closures, blink_threshold=blink_threshold)

    blink_durations = []
    for blink_event in blink_events:
        blink_durations.append(len(blink_event))
    return blink_durations


# compute standardised blink duration = 100*(duration/(0.862*amplitude + 121))
# params:    
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]blink_theshold :consider value above as blink
#           blink_threshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           window_size: number of time steps before and after blink to consider to get maximum velocity
#           close_labels : list of labels considered to be a closed eye
# returns:
#           list of standardized blink durations
def get_standard_blink_durations(eye_closures, eye_states,
                                 blink_threshold=0.6,
                                 window_size=100,
                                 flag_use_eye_state_label=True,
                                 close_labels=[1]):
    if flag_use_eye_state_label:
        blink_events = get_blink_events_from_eye_states(eye_states, close_labels=close_labels)
    else:
        blink_events = get_blink_events_from_eye_closures(eye_closures, blink_threshold=blink_threshold)

    standard_blink_durations = []
    for blink_event in blink_events:
        # get amplitude
        start_id = np.max([0, blink_event[0] - window_size + 1])
        before_blink_eye_closures = eye_closures[start_id:blink_event[0]+1]
        amplitude = np.max(before_blink_eye_closures) - np.min(before_blink_eye_closures)
        standard_blink_durations.append(100 * (len(blink_event) / (0.862 * amplitude + 121)))
    return standard_blink_durations


# compute blink intervals
# interval as the time from the end of the previous event to the beginning of the current event
# params:    
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]blink_theshold :consider value above as blink
#           blink_threshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           close_labels : list of labels considered to be a closed eye
# returns:
#           list of blink intervals
def get_blink_intervals(eye_closures, eye_states,
                        blink_threshold=0.6,
                        flag_use_eye_state_label=True,
                        close_labels=[1]):
    if flag_use_eye_state_label:
        blink_events = get_blink_events_from_eye_states(eye_states, close_labels=close_labels)
    else:
        blink_events = get_blink_events_from_eye_closures(eye_closures, blink_threshold=blink_threshold)

    blink_intervals = []
    prev_event = None
    for blink_event in blink_events:
        if prev_event is not None:
            cur_interval = blink_event[0] - prev_event[-1]
            blink_intervals.append(cur_interval)
        prev_event = blink_event
    return blink_intervals


# compute delay of lid reopening
# params:    
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]
#           blink_threshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           window_size: number of time steps before and after blink to consider to get maximum velocity
#           close_labels : list of labels considered to be a closed eye
# returns:
#           list of lid reopening durations
def get_delay_lid_reopening(eye_closures, eye_states,
                            blink_threshold=0.6,
                            flag_use_eye_state_label=True,
                            close_labels=[1]):
    if flag_use_eye_state_label:
        blink_events = get_blink_events_from_eye_states(eye_states, close_labels=close_labels)
    else:
        blink_events = get_blink_events_from_eye_closures(eye_closures, blink_threshold=blink_threshold)

    delay_lid_reopenings = []
    for blink_event in blink_events:
        event_closures = eye_closures[blink_event]
        # get id of first full closure:
        start_max_idx = 0
        for i in range(1, len(event_closures)):
            if event_closures[i] > event_closures[i - 1]:
                start_max_idx = i
            elif event_closures[i] < event_closures[i - 1]:
                break
        # get id of end of full closure:
        end_max_idx = len(event_closures) - 1
        for i in range(len(event_closures) - 2, -1, -1):
            if event_closures[i] > event_closures[i + 1]:
                end_max_idx = i
            elif event_closures[i] < event_closures[i + 1]:
                break
        # compute duration
        cur_delay = end_max_idx - start_max_idx
        delay_lid_reopenings.append(cur_delay)

    return delay_lid_reopenings



# compute standardized lid reopening duration = 100*(duration/(0.862*amplitude + 121))
# params:    
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]blink_theshold :consider value above as blink
#           blink_threshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           window_size: number of time steps before and after blink to consider to get maximum velocity
#           close_labels : list of labels considered to be a closed eye
# returns:
#           list of standardized lid reopening durations
def standard_duration_lid_reopening(eye_closures, eye_states,
                                    blink_threshold=0.6,
                                    window_size=100,
                                    flag_use_eye_state_label=True,
                                    close_labels=[1]):
    if flag_use_eye_state_label:
        blink_events = get_blink_events_from_eye_states(eye_states, close_labels=close_labels)
    else:
        blink_events = get_blink_events_from_eye_closures(eye_closures, blink_threshold=blink_threshold)

    standard_lid_reopenings = []
    for blink_event in blink_events:
        event_closures = eye_closures[blink_event]
        # get id of first full closure:
        start_max_idx = 0
        for i in range(1, len(event_closures)):
            if event_closures[i] > event_closures[i - 1]:
                start_max_idx = i
            elif event_closures[i] < event_closures[i - 1]:
                break
        # get id of end of full closure:
        end_max_idx = len(event_closures) - 1
        for i in range(len(event_closures) - 2, -1, -1):
            if event_closures[i] > event_closures[i + 1]:
                end_max_idx = i
            elif event_closures[i] < event_closures[i + 1]:
                break
        # compute delay for lid reopening
        cur_delay = end_max_idx - start_max_idx
        # compute amplitude in eye_closure before event
        start_id = np.max([0, blink_event[0] - window_size + 1])
        before_blink_eye_closures = eye_closures[start_id:blink_event[0] + 1]
        amplitude = np.max(before_blink_eye_closures) - np.min(before_blink_eye_closures)
        # standardized lid reopening
        standard_lid_reopenings.append(100 * (cur_delay / (0.862 * amplitude + 121)))
    return standard_lid_reopenings


# compute lid closure speed
# params:    
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]
#           blink_threshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           window_size: number of time steps before and after blink to consider to get maximum velocity
#           close_labels : list of labels considered to be a closed eye
#           aggregation: function to aggregate the speeds
#           standardize: flag indicating if we want to standardize the values (100*(speed/(7.186*amplitude + 60.55)))
# returns:
#           list of closure speeds
def closure_speed(eye_closures, eye_states,
                  blink_threshold=0.6,
                  window_size=100,
                  flag_use_eye_state_label=True,
                  close_labels=[1],
                  aggregation='max',
                  standardize=False):
    if flag_use_eye_state_label:
        blink_events = get_blink_events_from_eye_states(eye_states, close_labels=close_labels)
    else:
        blink_events = get_blink_events_from_eye_closures(eye_closures, blink_threshold=blink_threshold)

    prev_eye_closure = np.zeros(len(eye_closures))
    prev_eye_closure[1:] = eye_closures[:-1]
    velocities = eye_closures - prev_eye_closure
    velocities[0] = 0

    closure_speeds = []
    for blink_event in blink_events:
        # max opening id
        try:
            start_id = np.max([0, blink_event[0] - window_size])
            max_opening_id = np.argmax(eye_closures[start_id:blink_event[0]])
            cur_speeds = velocities[start_id + max_opening_id:blink_event[0]+1]
            if len(cur_speeds) == 0:
                cur_speeds = [0]
            if aggregation == 'max':
                cur_speed = np.max(cur_speeds)
            elif aggregation == 'mean':
                # Potential Problem: We have a lot of velocity values with 0.
                cur_speed = np.mean(cur_speeds)
        except:
            cur_speed = np.nan
            closure_speeds.append(cur_speed)
            continue
        if standardize:
            start_id = np.max([0, blink_event[0] - window_size + 1])
            before_blink_eye_closures = eye_closures[start_id:blink_event[0]+1]
            amplitude = np.max(before_blink_eye_closures) - np.min(before_blink_eye_closures)
            closure_speeds.append(100 * (cur_speed / (7.186 * amplitude + 60.55)))
        else:
            closure_speeds.append(cur_speed)
    return closure_speeds


# compute saccade durations
# params:    
#           saccade_lists: list of list of saccade indexes
#           x_angles: list of angles in x direction
#           y_angles: list of angles in x direction
#           standardize: flag indicating if we want to standardize the values (100*(saccadic duration/(amplitude*2.07 + 26)))
# returns:
#           list of saccade durations
def get_saccadic_durations(saccade_lists, x_angles=None,
                           y_angles=None,
                           standardize=False):
    if standardize:
        vals = []
        for sacc_list in saccade_lists:
            if x_angles is not None:
                amplitude = distance.euclidean([x_angles[sacc_list[0]], y_angles[sacc_list[0]]],
                                               [x_angles[sacc_list[-1]], y_angles[sacc_list[-1]]])
                vals.append(100 * (len(sacc_list) / (amplitude * 2.07 + 26)))
            else:
                vals.append(len(sacc_list))
        return vals
    else:
        return [len(a) for a in saccade_lists]


# compute saccade amplitudes
# params:    
#           saccade_lists: list of list of sccade indexes
#           x_angles: list of angles in x direction
#           y_angles: list of angles in x direction
# returns:
#           list of saccade amplitudes
def get_saccadic_amplitudes(saccade_lists, x_angles=None,
                            y_angles=None):
    vals = []
    for sacc_list in saccade_lists:
        if x_angles is not None:
            amplitude = float(distance.euclidean([x_angles[sacc_list[0]], y_angles[sacc_list[0]]],
                                                    [x_angles[sacc_list[-1]], y_angles[sacc_list[-1]]]))
            vals.append(amplitude)
        else:
            vals.append(np.nan)
    return vals


# compute saccade velocities
# params:    
#           saccade_lists: list of list of sccade indexes
#           x_angles: list of angles in x direction
#           y_angles: list of angles in x direction
#           x_vels: list of angular velocities in x direction
#           y_vels: list of angular velocities in y direction
#           aggregation_function: 'max' or 'mean'
#           standardize: flag indicating if we want to standardize the values 100*(speed/(445.9*(1–exp(–0.04844*amplitude–0.1121)))) for mean   
#                                                                             100*(max. speed/(580.4*(1–exp(–0.06771*amplitude–0.1498)))) for max velocity
# returns:
#           list of saccade durations
def get_saccadic_velocities(saccade_lists, x_angles=None,
                            y_angles=None,
                            x_vels=None,
                            y_vels=None,
                            aggregation_function='max',
                            standardize=False):
    vals = []
    for sacc_list in saccade_lists:
        if x_vels is not None:
            velocities = [np.abs(x_vels[sacc_list[i]]) + np.abs(x_vels[sacc_list[i]]) for i in range(len(sacc_list))]
            if aggregation_function == 'max':
                velocity = np.max(velocities)
            elif aggregation_function == 'mean':
                velocity = np.mean(velocities)
            if standardize:
                if x_angles is not None:
                    amplitude = float(distance.euclidean([x_angles[sacc_list[0]], y_angles[sacc_list[0]]],
                                                            [x_angles[sacc_list[-1]], y_angles[sacc_list[-1]]]))
                    if aggregation_function == 'max':
                        val = 100 * (velocity / (580.4 * (1 - np.exp(-0.06771 * (amplitude - 0.1498)))))
                        vals.append(val)
                    elif aggregation_function == 'mean':
                        velocity = np.mean(velocities)
                        val = 100 * (velocity / (445.9 * (1 - np.exp(-0.04844 * amplitude - 0.1121))))
                        vals.append(val)
                else:
                    vals.append(np.nan)
            else:
                vals.append(velocity)
        else:
            vals.append(np.nan)
    return vals


## compute vector containing all the saccadic features
# params:    
#           saccade_lists: list of list of sccade indexes
#           x_angles: list of angles in x direction
#           y_angles: list of angles in x direction
#           x_vels: list of angular velocities in x direction
#           y_vels: list of angular velocities in y direction
#           feature_aggregations: list of aggregation functions to apply to list of values
#           feature_prefix: prefix for the featurename
# returns:
#           list of saccade durations
def compute_saccadic_features(saccade_lists, x_angles=None,
                              y_angles=None,
                              x_vels=None, y_vels=None,
                              feature_prefix='',
                              feature_aggregations=['mean', 'std', 'median']):
    if len(feature_prefix) > 0:
        feature_prefix = feature_prefix + '_'

    feature_names = []
    features = []

    # saccadic duration features
    aggregations = ['mean']
    standards = [True, False]
    for aggregation in aggregations:
        for standardize in standards:
            durations = get_saccadic_durations(saccade_lists=saccade_lists, x_angles=x_angles,
                                               y_angles=y_angles,
                                               standardize=standardize)
            for feature_aggregation in feature_aggregations:
                cur_features_suffix = feature_prefix + '_' + feature_aggregation
                if standardize:
                    cur_features_suffix += '_standard'
                cur_features_suffix += '_saccadic_duration_' + aggregation
                feature_names.append(cur_features_suffix)
                features.append(get_feature_from_list(durations, feature_aggregation))

                # saccadic duration features
    aggregations = ['mean']
    standards = [False]
    for aggregation in aggregations:
        for standardize in standards:
            amplitudes = get_saccadic_amplitudes(saccade_lists=saccade_lists, x_angles=x_angles,
                                                 y_angles=y_angles)
            for feature_aggregation in feature_aggregations:
                cur_features_suffix = feature_prefix + '_' + feature_aggregation
                if standardize:
                    cur_features_suffix += '_standard'
                cur_features_suffix += '_saccadic_amplitude_' + aggregation
                feature_names.append(cur_features_suffix)
                features.append(get_feature_from_list(amplitudes, feature_aggregation))

    # saccadic duration velocities
    aggregations = ['mean', 'max']
    standards = [True, False]
    for aggregation in aggregations:
        for standardize in standards:
            velocities = get_saccadic_velocities(saccade_lists=saccade_lists, x_angles=x_angles,
                                                 y_angles=y_angles,
                                                 x_vels=x_vels,
                                                 y_vels=y_vels,
                                                 aggregation_function=aggregation,
                                                 standardize=standardize)
            for feature_aggregation in feature_aggregations:
                cur_features_suffix = feature_prefix + '_' + feature_aggregation
                if standardize:
                    cur_features_suffix += '_standard'
                cur_features_suffix += '_saccadic_velocity_' + aggregation
                feature_names.append(cur_features_suffix)
                features.append(get_feature_from_list(velocities, feature_aggregation))
    return np.array(features), feature_names


## compute vector containing all the eye closure features
# params:    
#           eye_closures: 1-dimensional vector of eye closures
#           eye_states: 1-dimensional vector of eye states [a 1 in the vector means closed]blink_theshold :consider value above as blink
#           blink_theshold :consider value above as blink (only important if flag_use_eye_state_label is False)
#           window_size: number of time steps before and after blink to consider to get maximum velocity
#           flag_use_eye_state_label: indicates if we want to use the eye_state or the eye_status to define blinks
#           close_labels : list of labels considered to be a closed eye
#           min_duration: minimal duration for a long blink
#           velocity_threshold: specify the threshold for the velocity, where the blink starts / ends
#           feature_aggregations: list of aggregation functions to apply to list of values
#           feature_prefix: prefix for the featurename
# returns:
#           list of features and featurenames
def compute_eye_closure_features(eye_closures, eye_states,
                                 blink_threshold=0.6, window_size=100,
                                 flag_use_eye_state_label=True,
                                 min_duration=1,
                                 close_labels=[1],
                                 velocity_threshold=0.1,
                                 feature_prefix='',
                                 feature_aggregations=['mean', 'std', 'median']):
    if len(feature_prefix) > 0:
        feature_prefix = feature_prefix + '_'

    feature_names = []
    features = []

    # features of 1. paper: "The Accuracy of Eyelid Movement Parameters for Drowsiness Detection"
    # (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3836343/)
    # LC
    feature_names.append(feature_prefix + 'LC')
    features.append(get_percentage_long_closures(eye_closures, eye_states,
                                                 blink_threshold=blink_threshold,
                                                 min_duration=min_duration,
                                                 flag_use_eye_state_label=flag_use_eye_state_label,
                                                 close_labels=close_labels))
    # IED
    inter_event_durations = get_inter_event_durations(eye_closures, eye_states,
                                                      blink_threshold=blink_threshold,
                                                      window_size=window_size,
                                                      flag_use_eye_state_label=flag_use_eye_state_label,
                                                      close_labels=close_labels)

    for feature_aggregation in feature_aggregations:
        feature_names.append(feature_prefix + '_' + feature_aggregation + '_IED')
        features.append(get_feature_from_list(inter_event_durations, feature_aggregation))

    # TEC    
    feature_names.append(feature_prefix + 'TEC')
    features.append(get_percent_time_with_eyes_closed(eye_closures, eye_states,
                                                      blink_threshold=blink_threshold,
                                                      window_size=window_size,
                                                      velocity_threshold=velocity_threshold,
                                                      flag_use_eye_state_label=flag_use_eye_state_label,
                                                      close_labels=close_labels))

    # BTD
    blink_total_durations = get_blink_total_duration(eye_closures, eye_states,
                                                     blink_threshold=blink_threshold,
                                                     flag_use_eye_state_label=flag_use_eye_state_label,
                                                     close_labels=close_labels)

    for feature_aggregation in feature_aggregations:
        feature_names.append(feature_prefix + '_' + feature_aggregation + '_BTD')
        features.append(get_feature_from_list(blink_total_durations, feature_aggregation))

    # -AVR
    neg_amp_vec = get_negative_amplitude_velocity_ratio(eye_closures, eye_states,
                                                        blink_threshold=blink_threshold,
                                                        window_size=window_size,
                                                        flag_use_eye_state_label=flag_use_eye_state_label,
                                                        close_labels=close_labels)
    for feature_aggregation in feature_aggregations:
        feature_names.append(feature_prefix + '_' + feature_aggregation + '_-AVR')
        features.append(get_feature_from_list(neg_amp_vec, feature_aggregation))

    # +AVR
    pos_amp_vec = get_positive_amplitude_velocity_ratio(eye_closures, eye_states,
                                                        blink_threshold=blink_threshold,
                                                        window_size=window_size,
                                                        flag_use_eye_state_label=flag_use_eye_state_label,
                                                        close_labels=close_labels)
    for feature_aggregation in feature_aggregations:
        feature_names.append(feature_prefix + '_' + feature_aggregation + '_+AVR')
        features.append(get_feature_from_list(pos_amp_vec, feature_aggregation))

    # features of 2. paper: "Blinks and saccades as indicators of fatigue in sleepiness warnings: looking tired?"
    # (https://www.tandfonline.com/doi/pdf/10.1080/00140130701817062)
    # blink duration
    blink_durations = get_blink_durations(eye_closures, eye_states,
                                          blink_threshold=blink_threshold,
                                          flag_use_eye_state_label=flag_use_eye_state_label,
                                          close_labels=close_labels)
    for feature_aggregation in feature_aggregations:
        feature_names.append(feature_prefix + '_' + feature_aggregation + '_blink_duration')
        features.append(get_feature_from_list(blink_durations, feature_aggregation))

    # standard blink duration
    standard_blink_durations = get_standard_blink_durations(eye_closures, eye_states,
                                                            blink_threshold=blink_threshold,
                                                            window_size=window_size,
                                                            flag_use_eye_state_label=flag_use_eye_state_label,
                                                            close_labels=close_labels)
    for feature_aggregation in feature_aggregations:
        feature_names.append(feature_prefix + '_' + feature_aggregation + '_standard_blink_duration')
        features.append(get_feature_from_list(standard_blink_durations, feature_aggregation))

    # blink duration
    blink_intervals = get_blink_intervals(eye_closures, eye_states,
                                          blink_threshold=blink_threshold,
                                          flag_use_eye_state_label=flag_use_eye_state_label,
                                          close_labels=close_labels)
    for feature_aggregation in feature_aggregations:
        feature_names.append(feature_prefix + '_' + feature_aggregation + '_blink_intervals')
        features.append(get_feature_from_list(blink_intervals, feature_aggregation))

    # delay reopening
    delay_reopenings = get_delay_lid_reopening(eye_closures, eye_states,
                                               blink_threshold=blink_threshold,
                                               flag_use_eye_state_label=flag_use_eye_state_label,
                                               close_labels=close_labels)
    for feature_aggregation in feature_aggregations:
        feature_names.append(feature_prefix + '_' + feature_aggregation + '_delay_reopening')
        features.append(get_feature_from_list(delay_reopenings, feature_aggregation))

    # get standard_duration_lid_reopening
    standard_lid_durations = standard_duration_lid_reopening(eye_closures, eye_states,
                                                             blink_threshold=blink_threshold,
                                                             window_size=window_size,
                                                             flag_use_eye_state_label=flag_use_eye_state_label,
                                                             close_labels=close_labels)
    for feature_aggregation in feature_aggregations:
        feature_names.append(feature_prefix + '_' + feature_aggregation + '_standard_lid_reopening_durations')
        features.append(get_feature_from_list(standard_lid_durations, feature_aggregation))

    # closure speed features
    aggregations = ['max', 'mean']
    standards = [True, False]
    for aggregation in aggregations:
        for standardize in standards:
            speeds = closure_speed(eye_closures, eye_states,
                                   blink_threshold=blink_threshold,
                                   window_size=window_size,
                                   flag_use_eye_state_label=flag_use_eye_state_label,
                                   close_labels=close_labels,
                                   aggregation=aggregation,
                                   standardize=standardize)

            for feature_aggregation in feature_aggregations:
                cur_features_suffix = feature_prefix + '_' + feature_aggregation
                if standardize:
                    cur_features_suffix += '_standard'
                cur_features_suffix += '_closure_speed_' + aggregation
                feature_names.append(cur_features_suffix)
                features.append(get_feature_from_list(speeds, feature_aggregation))

    return np.array(features), feature_names



def get_pupil_features(pupil,
                       feature_prefix='pupil',
                       feature_aggregations = ['mean', 'std', 'median', 'skew', 'kurtosis'],
                      ):
    
    pupil_mean = pupil
    tmp_2 = np.zeros(pupil_mean.shape)
    tmp_2[1:] = pupil_mean[:-1]
    gradient = pupil_mean - tmp_2
    
    feature_names = []
    features = []
    
    # pupil mean diameter
    for feature_aggregation in feature_aggregations:
        feature_names.append(feature_prefix + '_' + feature_aggregation + '_pupil_diameter')
        features.append(get_feature_from_list(pupil_mean, feature_aggregation))
        
    # pupil mean diameter
    for feature_aggregation in feature_aggregations:
        feature_names.append(feature_prefix + '_' + feature_aggregation + '_pupil_gradient')
        features.append(get_feature_from_list(gradient, feature_aggregation))
    
    return np.array(features), feature_names

########################################################################################################
#
# compute saccadic feature from paper 'Study of an Extensive Set of Eye Movement Features: Extraction Methods and Statistical Analysis' by Rigas et. al
#             https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7722561/
#
########################################################################################################

#   computes features for a saccade
# params:
#   saccade_list: indexes of saccade
#   pos_h_raw : degrees of visual angle in x-axis (raw)
#   pos_v_raw : degrees of visual angle in y-axis (raw)
#   pos_h : degrees of visual angle in x-axis (smooted)
#   pos_v : degrees of visual angle in y-axis (smooted)
#   vel_h : velocity in degrees of visual angle in x-axis (smooted)
#   vel_v : velocity in degrees of visual angle in y-axis (smooted)
#   acc_h : acceleration in degrees of visual angle in x-axis (smooted)
#   acc_v : acceleration in degrees of visual angle in y-axis (smooted)
#   sampling_rate: sampling rate
#   smoothing_window_length: smoothing window length in seconds
#   min_fixation_duration: minimal fixation duration (in s)
def get_features_for_smoothed_saccade(saccade_list,
                                      pos_h_raw,
                                      pos_v_raw,
                                      pos_h,
                                      pos_v,
                                      vel_h,
                                      vel_v,
                                      acc_h,
                                      acc_v,
                                      sampling_rate=1000,
                                      smoothing_window_length=0.007,
                                      min_fixation_duration=0.030):
    # select saccade
    pos_h_raw = pos_h_raw[saccade_list]
    pos_v_raw = pos_v_raw[saccade_list]
    pos_h = pos_h[saccade_list]
    pos_v = pos_v[saccade_list]
    vel_h = vel_h[saccade_list]
    vel_v = vel_v[saccade_list]
    acc_h = acc_h[saccade_list]
    acc_v = acc_v[saccade_list]

    out_dict = dict()
    # duration (ms)
    out_dict['duration'] = np.max([np.finfo(np.float32).eps, 1000 * (saccade_list[-1] - saccade_list[0]) / sampling_rate])

    # amplitude H, V, R (deg)
    out_dict['amp_v'] = np.abs(pos_h[-1] - pos_h[0])
    out_dict['amp_h'] = np.abs(pos_v[-1] - pos_v[0])
    out_dict['amp_r'] = np.sqrt(out_dict['amp_h'] ** 2 + out_dict['amp_v'] ** 2)

    # saccade horizontal direction
    # +1 -> right, -1 -> left
    out_dict['direction_h'] = np.sign(pos_h[-1] - pos_h[0])

    # saccade travelled distance
    out_dict['trav_dist'] = np.sum(np.sqrt(np.diff(pos_h) ** 2 + np.diff(pos_v) ** 2))

    # saccade efficiancy
    out_dict['efficiency'] = out_dict['amp_r'] / (out_dict['trav_dist'] + np.finfo(np.float32).eps)

    # saccade tail efficiency
    # tail = last 'smoothing_window_length' ms
    if len(pos_h_raw) > smoothing_window_length:
        tail_h = pos_h_raw[len(pos_h_raw) - int(np.round(smoothing_window_length * sampling_rate)):]
        tail_v = pos_h_raw[len(pos_v_raw) - int(np.round(smoothing_window_length * sampling_rate)):]
    else:
        tail_h = pos_h_raw
        tail_v = pos_h_raw
    
    # if we have really short saccade or really low sampling rate
    if len(tail_h) == 0:
        tail_h = pos_h_raw
        tail_v = pos_h_raw
    
    
    tail_amp = np.sqrt((tail_h[-1] - tail_h[0]) ** 2 + (tail_v[-1] - tail_v[0]) ** 2)
    tail_trav_dist = np.sum(np.sqrt(np.diff(tail_h) ** 2 + np.diff(tail_v) ** 2))
    out_dict['tail_efficiency'] = tail_amp / (np.finfo(float).eps + tail_trav_dist)

    # saccade tail percent incosistent
    # tail last 'smoothing_window_length' ms
    consistent_array = np.zeros([len(tail_h) - 1, 1])
    for m in range(len(tail_h) - 1):
        v1 = np.array([tail_h[m + 1], tail_v[m + 1]]) - np.array([tail_h[m], tail_v[m]])
        v2 = np.array([pos_h_raw[-1], pos_v_raw[-1]]) - np.array([pos_h_raw[0], pos_v_raw[0]])
        Angle = np.arctan2(np.abs(np.linalg.det(np.array([np.transpose(v1), np.transpose(v2)]))),
                           np.dot(np.transpose(v1), np.transpose(v2))) * 180 / np.pi
        if np.abs(Angle) < 60:
            consistent_array[m] = 1

    out_dict['tail_pr_inconsist'] = 100 * (1 - np.sum(consistent_array) / (np.finfo(float).eps + (len(tail_h) - 1)))

    # saccade trajectory curvature features
    if len(pos_h) >= 4:
        metrics = curve_metrics.curve_metrics(pos_h, pos_v, sampling_rate)
    else:
        metrics = dict()
    if 'direction' in metrics:
        out_dict['direction'] = metrics['direction']
    else:
        out_dict['direction'] = np.nan

    if 'IniDev' in metrics:
        out_dict['IniDev'] = metrics['IniDev']
    else:
        out_dict['IniDev'] = np.nan

    if 'IniAD' in metrics:
        out_dict['IniAD'] = metrics['IniAD']
    else:
        out_dict['IniAD'] = np.nan

    if 'RawDev' in metrics:
        out_dict['RawDev'] = metrics['RawDev']
    else:
        out_dict['RawDev'] = np.nan

    if 'RawPOC' in metrics:
        out_dict['RawPOC'] = metrics['RawPOC']
    else:
        out_dict['RawPOC'] = np.nan

    if 'CurveArea' in metrics:
        out_dict['CurveArea'] = metrics['CurveArea']
    else:
        out_dict['CurveArea'] = np.nan

    if 'pol2[0]' in metrics:
        out_dict['pol2[0]'] = metrics['pol2[0]']
    else:
        out_dict['pol2[0]'] = np.nan

    if 'curve3[0]' in metrics:
        out_dict['curve3[0]'] = metrics['curve3[0]']
    else:
        out_dict['curve3[0]'] = np.nan

    if 'POC3[0]' in metrics:
        out_dict['POC3[0]'] = metrics['POC3[0]']
    else:
        out_dict['POC3[0]'] = np.nan

    if 'curve3[1]' in metrics:
        out_dict['curve3[1]'] = metrics['curve3[1]']
    else:
        out_dict['curve3[1]'] = np.nan

    if 'POC3[1]' in metrics:
        out_dict['POC3[1]'] = metrics['POC3[1]']
    else:
        out_dict['POC3[1]'] = np.nan

    if 'curve3[MaxIndex]' in metrics:
        out_dict['curve3[MaxIndex]'] = metrics['curve3[MaxIndex]']
    else:
        out_dict['curve3[MaxIndex]'] = np.nan

    if 'curve3[MaxIndex]' in metrics:
        out_dict['curve3[MaxIndex]'] = metrics['curve3[MaxIndex]']
    else:
        out_dict['curve3[MaxIndex]'] = np.nan

    if 'POC3[MaxIndex]' in metrics:
        out_dict['POC3[MaxIndex]'] = metrics['POC3[MaxIndex]']
    else:
        out_dict['POC3[MaxIndex]'] = np.nan

    # number of velocity local minima
    SVel = np.sqrt(vel_h ** 2 + vel_v ** 2)
    N_localmin = 0;
    for k in range(len(SVel) - 2):
        if SVel[k] > SVel[k + 1] and SVel[k + 2] > SVel[k + 1]:
            N_localmin = N_localmin + 1
    out_dict['num_vel_loc_min'] = N_localmin

    #  peak velocity H, V, R (deg/s)
    out_dict['peak_vel_h'] = np.max(np.abs(vel_h))
    out_dict['peak_vel_v'] = np.max(np.abs(vel_v));
    out_dict['peak_vel_r'] = np.max(np.sqrt(vel_h ** 2 + vel_v ** 2))

    #  mean velocity H, V, R (deg/s)
    out_dict['mean_vel_h'] = out_dict['amp_h'] / (out_dict['duration'] / sampling_rate)
    out_dict['mean_vel_v'] = out_dict['amp_v'] / (out_dict['duration'] / sampling_rate)
    out_dict['mean_vel_r'] = np.sqrt(out_dict['mean_vel_h'] ** 2 + out_dict['mean_vel_v'] ** 2)

    #  velocity profile mean H, V, R (deg/s)
    out_dict['vel_profile_mean_h'] = np.mean(np.abs(vel_h))
    out_dict['vel_profile_mean_v'] = np.mean(np.abs(vel_v))
    out_dict['vel_profile_mean_r'] = np.mean(np.sqrt(vel_h ** 2 + vel_v ** 2))

    #  velocity profile median H, V, R (deg/s)
    out_dict['vel_profile_median_h'] = np.median(np.abs(vel_h))
    out_dict['vel_profile_median_v'] = np.median(np.abs(vel_v))
    out_dict['vel_profile_median_r'] = np.median(np.sqrt(vel_h ** 2 + vel_v ** 2))

    #  velocity profile std H, V, R (deg/s)
    out_dict['vel_profile_std_h'] = np.std(np.abs(vel_h))
    out_dict['vel_profile_std_v'] = np.std(np.abs(vel_v))
    out_dict['vel_profile_std_r'] = np.std(np.sqrt(vel_h ** 2 + vel_v ** 2))

    #  velocity profile skewness H, V, R (deg/s)
    out_dict['vel_profile_skew_h'] = skew(np.abs(vel_h))
    out_dict['vel_profile_skew_v'] = skew(np.abs(vel_v))
    out_dict['vel_profile_skew_r'] = skew(np.sqrt(vel_h ** 2 + vel_v ** 2))

    #  velocity profile kurtosis H, V, R (deg/s)
    out_dict['vel_profile_kurtosis_h'] = kurtosis(np.abs(vel_h))
    out_dict['vel_profile_kurtosis_v'] = kurtosis(np.abs(vel_v))
    out_dict['vel_profile_kurtosis_r'] = kurtosis(np.sqrt(vel_h ** 2 + vel_v ** 2))

    # find liit of acceleration-deceleration phases (via peak velocity for less effect noise)
    vel = np.sqrt(vel_h ** 2 + vel_v ** 2)
    pIdx = np.nanargmax(vel)

    # peak acceleration H, V, R (deg/s^2)
    if pIdx == 0:
        out_dict['peak_acc_h'] = np.max(np.abs(acc_h[0:pIdx + 1]))
        out_dict['peak_acc_v'] = np.max(np.abs(acc_v[0:pIdx + 1]))
        out_dict['peak_acc_r'] = np.max(np.sqrt(acc_h[0:pIdx + 1] ** 2 + acc_v[0:pIdx + 1] ** 2))
    else:
        out_dict['peak_acc_h'] = np.max(np.abs(acc_h[0:pIdx + 1]))
        out_dict['peak_acc_v'] = np.max(np.abs(acc_v[0:pIdx + 1]))
        out_dict['peak_acc_r'] = np.max(np.sqrt(acc_h[0:pIdx + 1] ** 2 + acc_v[0:pIdx + 1] ** 2))

    # peak deceleration H, V, R (deg/s^2)
    if pIdx == len(acc_h) - 1:
        out_dict['peak_deacc_h'] = np.max(np.abs(acc_h[pIdx - 1:]))
        out_dict['peak_deacc_v'] = np.max(np.abs(acc_v[pIdx - 1:]))
        out_dict['peak_deacc_r'] = np.max(np.sqrt(acc_h[pIdx - 1:] ** 2 + acc_v[pIdx - 1:] ** 2))
    else:
        out_dict['peak_deacc_h'] = np.max(np.abs(acc_h[pIdx:]))
        out_dict['peak_deacc_v'] = np.max(np.abs(acc_v[pIdx:]))
        out_dict['peak_deacc_r'] = np.max(np.sqrt(acc_h[pIdx:] ** 2 + acc_v[pIdx:] ** 2))

    # acceleration profile mean H, V, R (deg/s^2)
    out_dict['acc_profile_mean_h'] = np.mean(np.abs(acc_h))
    out_dict['acc_profile_mean_v'] = np.mean(np.abs(acc_v))
    out_dict['acc_profile_mean_r'] = np.mean(np.sqrt(acc_h ** 2 + acc_v ** 2))

    # acceleration profile median H, V, R (deg/s^2)
    out_dict['acc_profile_median_h'] = np.median(np.abs(acc_h))
    out_dict['acc_profile_median_v'] = np.median(np.abs(acc_v))
    out_dict['acc_profile_median_r'] = np.median(np.sqrt(acc_h ** 2 + acc_v ** 2))

    # acceleration profile std H, V, R (deg/s^2)
    out_dict['acc_profile_std_h'] = np.std(np.abs(acc_h))
    out_dict['acc_profile_std_v'] = np.std(np.abs(acc_v))
    out_dict['acc_profile_std_r'] = np.std(np.sqrt(acc_h ** 2 + acc_v ** 2))

    # acceleration profile skew H, V, R (deg/s^2)
    out_dict['acc_profile_skew_h'] = skew(np.abs(acc_h))
    out_dict['acc_profile_skew_v'] = skew(np.abs(acc_v))
    out_dict['acc_profile_skew_r'] = skew(np.sqrt(acc_h ** 2 + acc_v ** 2))

    # acceleration profile kurtosis H, V, R (deg/s^2)
    out_dict['acc_profile_kurtosis_h'] = kurtosis(np.abs(acc_h))
    out_dict['acc_profile_kurtosis_v'] = kurtosis(np.abs(acc_v))
    out_dict['acc_profile_kurtosis_r'] = kurtosis(np.sqrt(acc_h ** 2 + acc_v ** 2))

    # amplitude-duration ratio H, V, R (deg/s)
    out_dict['amp_duration_ratio_h'] = out_dict['amp_h'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['amp_duration_ratio_v'] = out_dict['amp_v'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['amp_duration_ratio_r'] = out_dict['amp_r'] / (1 / sampling_rate * out_dict['duration'])

    # peak velocity-amplitude ratio H, V, R (deg/s/deg)
    out_dict['peak_vel_amp_ratio_h'] = out_dict['amp_h'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['peak_vel_amp_ratio_v'] = out_dict['amp_v'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['peak_vel_amp_ratio_r'] = out_dict['amp_r'] / (1 / sampling_rate * out_dict['duration'])

    # peak velocity-duration ratio (a.k.a. 'saccadic ratio') H, V, R (deg/s^2)
    out_dict['peak_vel_duration_ratio_h'] = out_dict['peak_vel_h'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['peak_vel_duration_ratio_v'] = out_dict['peak_vel_v'] / (1 / sampling_rate * out_dict['duration'])
    out_dict['peak_vel_duration_ratio_r'] = out_dict['peak_vel_r'] / (1 / sampling_rate * out_dict['duration'])

    # peak velocity-mean velocity ratio (a.k.a. 'q-ratio') H, V, R
    out_dict['peak_vel_duration_ratio_h'] = out_dict['peak_vel_h'] / (out_dict['mean_vel_h'] + np.finfo(np.float32).eps)
    out_dict['peak_vel_duration_ratio_v'] = out_dict['peak_vel_v'] / (out_dict['mean_vel_v'] + np.finfo(np.float32).eps)
    out_dict['peak_vel_duration_ratio_r'] = out_dict['peak_vel_r'] / (out_dict['mean_vel_r'] + np.finfo(np.float32).eps)

    # peak velocity-local noise ratio Ratio
    v = np.sqrt(vel_h ** 2 + vel_v ** 2)
    '''
    window_start_idx = int(np.round(saccade_list[0]))
    window_end_idx = int(np.max([0, np.ceil(window_start_idx - min_fixation_duration * sampling_rate)]))
    VelLocNoise = v[window_start_idx:window_end_idx]
    VelLocNoise = np.mean(VelLocNoise) + 3 * np.std(VelLocNoise)
    '''
    VelLocNoise = np.nanmean(v) + 3 * np.nanstd(v)
    out_dict['peak_velocity_loc_noise_ratio_r'] = out_dict['peak_vel_r'] / (VelLocNoise + np.finfo(np.float32).eps)

    # acceleration-deceleration duration ratio
    out_dict['acc_dec_duration_ratio'] = pIdx / (len(vel) - pIdx)

    # peak acceleration-peak deceleration ratio H, V, R
    out_dict['peak_acc_peak_dec_ratio_h'] = out_dict['peak_acc_h'] / (out_dict['peak_deacc_h'] + np.finfo(np.float32).eps)
    out_dict['peak_acc_peak_dec_ratio_v'] = out_dict['peak_acc_v'] / (out_dict['peak_deacc_v'] + np.finfo(np.float32).eps)
    out_dict['peak_acc_peak_dec_ratio_r'] = out_dict['peak_acc_r'] / (out_dict['peak_deacc_r'] + np.finfo(np.float32).eps)

    # ADDITIONAL LOHR FEATURES
    out_dict['pos_trace_sd_h'] = np.std(pos_h)
    out_dict['pos_trace_sd_v'] = np.std(pos_v)
    out_dict['dispersion'] = np.max(pos_h) - np.min(pos_h) + np.max(pos_v) - np.min(pos_v)
    out_dict['sac_angle'] = np.arctan(pos_v[-1] - pos_v[0]) / ((pos_h[-1] - pos_h[0]) + np.finfo(np.float32).eps)
    out_dict['centroid_h'] = np.mean(pos_h)
    out_dict['centroid_v'] = np.mean(pos_v)

    return out_dict


#   computes features for a saccades
# params:
#   saccade_list: list of indices of saccades
#   pos_h_raw : degrees of visual angle in x-axis (raw)
#   pos_v_raw : degrees of visual angle in y-axis (raw)
#   pos_h : degrees of visual angle in x-axis (smooted)
#   pos_v : degrees of visual angle in y-axis (smooted)
#   vel_h : velocity in degrees of visual angle in x-axis (smooted)
#   vel_v : velocity in degrees of visual angle in y-axis (smooted)
#   acc_h : acceleration in degrees of visual angle in x-axis (smooted)
#   acc_v : acceleration in degrees of visual angle in y-axis (smooted)
#   sampling_rate: sampling rate
#   smoothing_window_length: smoothing window length in seconds
#   min_fixation_duration: minimal fixation duration (in s)
def get_features_for_smoothed_saccades(saccade_lists,
                                       pos_h_raw,
                                       pos_v_raw,
                                       pos_h,
                                       pos_v,
                                       vel_h,
                                       vel_v,
                                       acc_h,
                                       acc_v,
                                       sampling_rate=1000,
                                       smoothing_window_length=0.007,
                                       min_fixation_duration=0.030,
                                       feature_prefix='saccade',
                                       feature_aggregations=['mean', 'std', 'median']):
    # print('number of saccades:  ' + str(len(saccade_lists)))

    counter = 0
    for saccade_list in saccade_lists:
        if np.sum(np.isnan(pos_h[saccade_list])) > 0:
            continue
        if counter == 0:
            features = get_features_for_smoothed_saccade(saccade_list,
                                                         pos_h_raw,
                                                         pos_v_raw,
                                                         pos_h,
                                                         pos_v,
                                                         vel_h,
                                                         vel_v,
                                                         acc_h,
                                                         acc_v,
                                                         sampling_rate=sampling_rate,
                                                         smoothing_window_length=smoothing_window_length,
                                                         min_fixation_duration=min_fixation_duration)
            for key in features:
                features[key] = [features[key]]
        else:
            tmp_features = get_features_for_smoothed_saccade(saccade_list,
                                                             pos_h_raw,
                                                             pos_v_raw,
                                                             pos_h,
                                                             pos_v,
                                                             vel_h,
                                                             vel_v,
                                                             acc_h,
                                                             acc_v,
                                                             sampling_rate=sampling_rate,
                                                             smoothing_window_length=smoothing_window_length,
                                                             min_fixation_duration=min_fixation_duration)
            for key in features:
                features[key].append(tmp_features[key])
        counter += 1
    out_features = []
    feature_names = []
    try:
        for key in features:
            cur_list = features[key]
            for aggregation_function in feature_aggregations:
                cur_feature = get_feature_from_list(cur_list, aggregation_function)
                out_features.append(cur_feature)
                feature_names.append(feature_prefix + '_texas_' + aggregation_function + '_' + key)
    except:
        warnings.warn('Warning:no saccades found!')
        out_features = np.array([np.nan for a in range(85*len(feature_aggregations))])
        #raise RuntimeError('no saccades found to process')
    return np.array(out_features), feature_names
    
    
    
########################################################################################################
#
# compute saccadic feature from paper 'Study of an Extensive Set of Eye Movement Features: Extraction Methods and Statistical Analysis' by Rigas et. al
#             https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7722561/
#
########################################################################################################

#   computes features for a fixation
# params:
#   fixation_list: indexes of fixations
#   pos_h_raw : degrees of visual angle in x-axis (raw)
#   pos_v_raw : degrees of visual angle in y-axis (raw)
#   pos_h : degrees of visual angle in x-axis (smooted)
#   pos_v : degrees of visual angle in y-axis (smooted)
#   vel_h : velocity in degrees of visual angle in x-axis (smooted)
#   vel_v : velocity in degrees of visual angle in y-axis (smooted)
#   acc_h : acceleration in degrees of visual angle in x-axis (smooted)
#   acc_v : acceleration in degrees of visual angle in y-axis (smooted)
#   sampling_rate: sampling rate
def get_features_for_smoothed_fixation(fixation_list,
                                      pos_h_raw,
                                      pos_v_raw,
                                      pos_h,
                                      pos_v,
                                      vel_h,
                                      vel_v,
                                      acc_h,
                                      acc_v,
                                      sampling_rate=1000):
    from sklearn import linear_model
    from sklearn.metrics import r2_score                                  
    
    # select saccade
    pos_h_raw = pos_h_raw[fixation_list]
    pos_v_raw = pos_v_raw[fixation_list]
    pos_h = pos_h[fixation_list]
    pos_v = pos_v[fixation_list]
    vel_h = vel_h[fixation_list]
    vel_v = vel_v[fixation_list]
    acc_h = acc_h[fixation_list]
    acc_v = acc_v[fixation_list]
    
    v = np.sqrt(vel_h ** 2 + vel_v ** 2)
    Fix_Vel_P90 = np.percentile(v,90)
    
    out_dict = dict()
    # duration (ms)
    out_dict['duration'] = np.max([np.finfo(np.float32).eps, 1000 * (fixation_list[-1] - fixation_list[0]) / sampling_rate])
    
    # Position Centroid H, V (deg)
    out_dict['PosCentroid_H'] = np.mean(pos_h)
    out_dict['PosCentroid_V'] = np.mean(pos_v)
    
    # Drift Displacement H, V, R (deg)
    out_dict['DriftDisp_H'] = np.abs(pos_h[-1] - pos_h[0])
    out_dict['DriftDisp_V'] = np.abs(pos_v[-1] - pos_v[0])
    out_dict['DriftDisp_R'] = np.sqrt((pos_h[-1]-pos_h[0])**2 + (pos_v[-1]-pos_v[0])**2)
    
    
    # Drift Distance H, V, R (deg)
    out_dict['DriftDist_H'] = np.sum(np.abs(np.diff(pos_h)))
    out_dict['DriftDist_V'] = np.sum(np.abs(np.diff(pos_v)))
    out_dict['DriftDist_R'] = np.sum(np.sqrt(np.diff(pos_h)**2 + np.diff(pos_v)**2))
    
    # Drift Distance H, V, R (deg)
    out_dict['DriftDist_H'] = np.sum(np.abs(np.diff(pos_h)))
    out_dict['DriftDist_V'] = np.sum(np.abs(np.diff(pos_v)))
    out_dict['DriftDist_R'] = np.sum(np.sqrt(np.diff(pos_h)**2 + np.diff(pos_v)**2))
    
    # Drift mean Velocity H, V, R (deg/s)
    out_dict['DriftAvgSpeed_H'] = out_dict['DriftDist_H']/(0.001*out_dict['duration']);
    out_dict['DriftAvgSpeed_V'] = out_dict['DriftDist_V']/(0.001*out_dict['duration']);
    out_dict['DriftAvgSpeed_R'] = out_dict['DriftDist_R']/(0.001*out_dict['duration']); 
    
    # Drift 1-order (linear) fit Slope and R^2 H, V 
    timeData = np.arange(len(pos_h)) / sampling_rate
    lr = linear_model.LinearRegression()
    x = timeData.reshape(-1, 1)
    y = pos_h
    model = lr.fit(x, y)
    pred = model.predict(x)
    r2 = r2_score(pred,y)
    slope = model.coef_[0]
    out_dict['DriftFitLn_Slope_H'] = slope
    out_dict['DriftFitLn_R2_H'] = r2
    x = timeData.reshape(-1, 1)
    y = pos_v
    model = lr.fit(x, y)
    pred = model.predict(x)
    r2 = r2_score(pred,y)
    slope = model.coef_[0]
    out_dict['DriftFitLn_Slope_V'] = slope
    out_dict['DriftFitLn_R2_V'] = r2
    
    
    # Drift 2-order (quadratic) fit R^2 H, V 
    timeData = np.arange(len(pos_h)) / sampling_rate
    timeData = np.array([timeData, np.power(timeData,2)]).T
    x = timeData
    y = pos_h
    model = lr.fit(x, y)
    pred = model.predict(x)
    r2 = r2_score(pred,y)
    out_dict['DriftFitQd_R2_H'] = r2
    x = timeData
    y = pos_v
    model = lr.fit(x, y)
    pred = model.predict(x)
    r2 = r2_score(pred,y)
    out_dict['DriftFitQd_R2_V'] = r2
    
    ''' 
    % Drift step-wise fit parameter percentages H, V
    timeData = (Fix_Start(i):Fix_End(i))'/[Params.samplingFreq];
    Tstep = [timeData power(timeData,2)];
    [~, ~, ~, Xinmodel, ~, ~, ~] = stepwisefit(Tstep, pos_h, 'display', 'off');
    [~, ~, ~, Yinmodel, ~, ~, ~] = stepwisefit(Tstep, pos_v, 'display', 'off');
    Fix_StepLQParam_H(i, 1) = Xinmodel[0]; Fix_StepLQParam_H(i, 2) = Xinmodel(2);
    Fix_StepLQParam_V(i, 1) = Yinmodel[0]; Fix_StepLQParam_V(i, 2) = Yinmodel(2);
   '''
    
    # Velocity Profile Mean H, V, R (deg/s)
    out_dict['VelProfMn_H'] = np.mean(np.abs(vel_h))
    out_dict['VelProfMn_V'] = np.mean(np.abs(vel_v))
    out_dict['VelProfMn_R'] = np.mean(np.sqrt(vel_h**2 + vel_v**2))
    
    # Velocity Profile Median H, V, R (deg/s)
    out_dict['VelProfMd_H'] = np.median(np.abs(vel_h))
    out_dict['VelProfMd_V'] = np.median(np.abs(vel_v))
    out_dict['VelProfMd_R'] = np.median(np.sqrt(vel_h**2 + vel_v**2))
    
    # Velocity Profile Std H, V, R (deg/s)
    out_dict['VelProfSd_H'] = np.std(abs(vel_h))
    out_dict['VelProfSd_V'] = np.std(np.abs(vel_v))
    out_dict['VelProfSd_R'] = np.std(np.sqrt(vel_h**2 + vel_v**2))
    
    # Velocity Profile Skewness H, V, R (deg/s)
    out_dict['VelProfSk_H'] = skew(np.abs(vel_h))
    out_dict['VelProfSk_V'] = skew(np.abs(vel_v))
    out_dict['VelProfSk_R'] = skew(np.sqrt(vel_h**2 + vel_v**2))
    
    # Velocity Profile Kurtosis H, V, R (deg/s)
    out_dict['VelProfKu_H'] = kurtosis(np.abs(vel_h))
    out_dict['VelProfKu_V'] = kurtosis(np.abs(vel_v))
    out_dict['VelProfKu_R'] = kurtosis(np.sqrt(vel_h**2 + vel_v**2))
    
    # Percent Above 90-percentile Velocity Threshold R
    out_dict['PrAbP90VelThr_R'] = 100*np.sum(v > Fix_Vel_P90)/len(v)
    
    '''
    # Percent Crossing 90-percentile Velocity Threshold R
    cross_idx = crossing(v(Fix_Start(i):Fix_End(i)), [], Fix_Vel_P90);
    out_dict['PrCrP90VelThr_R'] = 100*length(cross_idx)/length(v(Fix_Start(i):Fix_End(i)));
    '''
    	
    # Acceleration Profile Mean H, V, R (deg/s^2)
    out_dict['AccProfMn_H'] = np.mean(np.abs(acc_h));
    out_dict['AccProfMn_V'] = np.mean(np.abs(acc_v));
    out_dict['AccProfMn_R'] = np.mean(np.sqrt(acc_h**2 + acc_v**2));
        
    # Acceleration Profile Median H, V, R (deg/s^2)
    out_dict['AccProfMd_H'] = np.median(np.abs(acc_h));
    out_dict['AccProfMd_V'] = np.median(np.abs(acc_v));
    out_dict['AccProfMd_R'] = np.median(np.sqrt(acc_h**2 + acc_v**2));
        
    # Acceleration Profile Std H, V, R (deg/s^2)
    out_dict['AccProfSd_H'] = np.std(np.abs(acc_h));
    out_dict['AccProfSd_V'] = np.std(np.abs(acc_v));
    out_dict['AccProfSd_R'] = np.std(np.sqrt(acc_h**2 + acc_v**2));
        
    # Acceleration Profile Skewness H, V, R (deg/s^2)
    out_dict['AccProfSk_H'] = skew(np.abs(acc_h));
    out_dict['AccProfSk_V'] = skew(np.abs(acc_v));
    out_dict['AccProfSk_R'] = skew(np.sqrt(acc_h**2 + acc_v**2));
        
    # Acceleration Profile Kurtosis H, V, R (deg/s^2)
    out_dict['AccProfKu_H'] = kurtosis(np.abs(acc_h));
    out_dict['AccProfKu_V'] = kurtosis(np.abs(acc_v));
    out_dict['AccProfKu_R'] = kurtosis(np.sqrt(acc_h**2 + acc_v**2));

    return out_dict


#   computes features for a saccades
# params:
#   fixation_lists: list of indices of saccades
#   pos_h_raw : degrees of visual angle in x-axis (raw)
#   pos_v_raw : degrees of visual angle in y-axis (raw)
#   pos_h : degrees of visual angle in x-axis (smooted)
#   pos_v : degrees of visual angle in y-axis (smooted)
#   vel_h : velocity in degrees of visual angle in x-axis (smooted)
#   vel_v : velocity in degrees of visual angle in y-axis (smooted)
#   acc_h : acceleration in degrees of visual angle in x-axis (smooted)
#   acc_v : acceleration in degrees of visual angle in y-axis (smooted)
#   sampling_rate: sampling rate
def get_features_for_smoothed_fixations(fixation_lists,
                                       pos_h_raw,
                                       pos_v_raw,
                                       pos_h,
                                       pos_v,
                                       vel_h,
                                       vel_v,
                                       acc_h,
                                       acc_v,
                                       sampling_rate=1000,
                                       feature_prefix='fixation',
                                       feature_aggregations=['mean', 'std', 'median']):
    # print('number of saccades:  ' + str(len(fixation_list)))

    counter = 0
    for fixation_list in fixation_lists:
        if np.sum(np.isnan(pos_h[fixation_list])) > 0:
            continue
        if counter == 0:
            features = get_features_for_smoothed_fixation(fixation_list,
                                                         pos_h_raw,
                                                         pos_v_raw,
                                                         pos_h,
                                                         pos_v,
                                                         vel_h,
                                                         vel_v,
                                                         acc_h,
                                                         acc_v,
                                                         sampling_rate=sampling_rate)
            for key in features:
                features[key] = [features[key]]
        else:
            tmp_features = get_features_for_smoothed_fixation(fixation_list,
                                                             pos_h_raw,
                                                             pos_v_raw,
                                                             pos_h,
                                                             pos_v,
                                                             vel_h,
                                                             vel_v,
                                                             acc_h,
                                                             acc_v,
                                                             sampling_rate=sampling_rate)
            for key in features:
                features[key].append(tmp_features[key])
        counter += 1
    out_features = []
    feature_names = []
    try:
        for key in features:
            cur_list = features[key]
            for aggregation_function in feature_aggregations:
                cur_feature = get_feature_from_list(cur_list, aggregation_function)
                out_features.append(cur_feature)
                feature_names.append(feature_prefix + '_texas_' + aggregation_function + '_' + key)
    except:
        warnings.warn('Warning:no saccades found!')
        out_features = np.array([np.nan for a in range(49*len(feature_aggregations))])
        #raise RuntimeError('no saccades found to process')
    return np.array(out_features), feature_names



#Gaze entropy measures detect alcohol-induced driver impairment - ScienceDirect
# https://www.sciencedirect.com/science/article/abs/pii/S0376871619302789
# computes the gaze entropy features
# params:
#    fixation_list: list of fixation idx (e.g. by calling get_sacc_fix_lists_dispersion)
#    x_pixel: x-coordinates of data
#    y_pixel: y coordinata of data
#    x_dim: screen horizontal pixels
#    y_dim: screen vertical pixels
#    patch_size: size of patches to use
def get_gaze_entropy_features(fixation_list,
                             x_pixel,
                             y_pixel,
                             x_dim = 1280,
                             y_dim = 1024,
                             patch_size = 64):


    def calc_patch(patch_size,mean):
        return int(np.floor(mean / patch_size))



    def entropy(value):
        return value * (np.log(value) / np.log(2))


    # dictionary of visited patches
    patch_dict = dict()
    # dictionary for patch transitions
    trans_dict = dict()
    pre = None
    for fix_list in fixation_list:
        x_mean = np.mean(x_pixel[fix_list])
        y_mean = np.mean(y_pixel[fix_list])
        patch_x = calc_patch(patch_size,x_mean)
        patch_y = calc_patch(patch_size,y_mean)
        cur_point = str(patch_x) + '_' + str(patch_y)
        if cur_point not in patch_dict:
            patch_dict[cur_point] = 0
        patch_dict[cur_point] += 1
        if pre is not None:
            if pre not in trans_dict:
                trans_dict[pre] = []
            trans_dict[pre].append(cur_point)
        pre = cur_point


    # stationary gaze entropy
    # SGE
    sge = 0
    x_max = int(x_dim / patch_size)
    y_max = int(y_dim / patch_size)
    fix_number = len(fixation_list)
    for i in range(x_max):
        for j in range(y_max):
            cur_point = str(i) + '_' + str(j)
            if cur_point in patch_dict:
                cur_prop = patch_dict[cur_point] / fix_number
                sge += entropy(cur_prop)
    sge = sge * -1
    
    # gaze transition entropy
    # GTE
    gte = 0
    for patch in trans_dict:
        cur_patch_prop = patch_dict[patch] / fix_number
        cur_destination_list = trans_dict[patch]
        (values,counts) = np.unique(cur_destination_list,return_counts = True)
        inner_sum = 0
        for i in range(len(values)):
            cur_val = values[i]
            cur_count = counts[i]
            cur_prob = cur_count / np.sum(counts)
            cur_entropy = entropy(cur_prob)
            inner_sum += cur_entropy
        #print('cur_patch_prop: ' + str(cur_patch_prop))
        #print('inner_sum: ' + str(inner_sum))
        gte += (cur_patch_prop * inner_sum)
    gte = gte * -1
    
    return (np.array([sge,gte],),['fixation_feature_SGE',
                                  'fixation_feature_GTE'])

# computes all features
#
# params:
#    data: matrix containing the data
#    data_format: dictionary describing the data
#    screenPX_x: screen horizontal pixels
#    screenPX_y: screen vertical pixels
#    screenCM_x: screen horizontal length (cm)
#    screenCM_y: screen vertical length (cm)
#    distanceCM: distance from head to screen (cm)
#    minDurFix: minDurFix parameter for event detection (minimal fixation duration in ms)
#    sampling_rate: sampling rate
#    blink_threshold: threshold what counts as blink
#    window_size: window of ms used to determine start and end of a blink
#    min_blink_duration: minimal blink duration in ms
#    velocity_threshold: velocity threshold for the detection of the beginning and ending of a blink
#    smoothing_window_length: smoothing window length for smoothing the velocities (Texas algorithm)
#    feature_aggregations: list of aggregations performed on list of values
def compute_combined_features(data,
                              data_format,
                              screenPX_x=1280,
                              screenPX_y=1024,
                              screenCM_x=38,
                              screenCM_y=30.2,
                              distanceCM=68,
                              minDurFix=20,
                              sampling_rate=1000,
                              blink_threshold=0.6,
                              window_size=100,
                              min_blink_duration=1,
                              velocity_threshold=0.1,
                              smoothing_window_length=0.007,
                              feature_aggregations=['mean', 'std', 'median', 'skew', 'kurtosis'],
                              verbose = 0):
    if verbose == 0:
        disable = True
    else:
        disable = False
    for i in tqdm(np.arange(data.shape[0]),disable = disable):
        x_pixel = data[i, :, data_format['x_pixel']]
        y_pixel = data[i, :, data_format['y_pixel']]
        if 'eye_closure' in data_format:            
            eye_closures = data[i, :, data_format['eye_closure']]
        else:
            eye_closures = np.zeros(x_pixel.shape)
        if 'blink' in data_format:
            eye_blink = data[i, :, data_format['blink']]
        else:
            eye_blink = np.zeros(x_pixel.shape)
        corrupt = data[i, :, data_format['corrupt']]
        try:
            pupil  = data[i,:,data_format['pupil']]
        except:
            pupil  = np.zeros(x_pixel.shape)

        # get degrees of visual angle
        x_dva = event_detection.pix2deg(x_pixel, screenPX_x, screenCM_x, distanceCM, adjust_origin=True)
        y_dva = event_detection.pix2deg(y_pixel, screenPX_y, screenCM_y, distanceCM, adjust_origin=True)

        # apply smoothing like in https://digital.library.txstate.edu/handle/10877/6874
        smooth_vals = smoothing.smooth_data(x_dva, y_dva)
        x_smo = smooth_vals['x_smo']
        y_smo = smooth_vals['y_smo']
        vel_x = smooth_vals['vel_x']
        vel_y = smooth_vals['vel_y']
        vel = smooth_vals['vel']
        acc_x = smooth_vals['acc_x']
        acc_y = smooth_vals['acc_y']
        acc = smooth_vals['acc']
        
        # get eye movement events (fixations, saccades)
        list_dicts, event_df = get_sacc_fix_lists_dispersion(x_dva, y_dva,
                                                      corrupt = corrupt,
                                                      sampling_rate = sampling_rate)
       
        
        # add pupil features
        features_pupils, feature_names_pupils = get_pupil_features(pupil,
                                                                   feature_prefix='pupil',
                                                                   feature_aggregations=feature_aggregations)
        
        
        # add gaze entropy features
        features_gaze_entropy, feature_name_gaze_entropy = get_gaze_entropy_features(fixation_list=list_dicts['fixations'],
                                             x_pixel=x_pixel,
                                             y_pixel=y_pixel,
                                             patch_size=64)
        
        
        # add count features
        # add blink count
        blink_events = get_blink_events_from_eye_states(eye_blink, close_labels=[1])        
        count_features = [len(blink_events)]
        count_feature_names = ['count_blinks']
        count_features.append(len(np.where(corrupt == 1)[0]))
        count_feature_names.append('count_corrupt')
        
        # add event counts
        count_features.append(len(list_dicts['fixations']))
        count_feature_names.append('count_fixations')
        count_features.append(len(list_dicts['errors']))
        count_feature_names.append('count_errors')
        count_features.append(len(list_dicts['saccades']))
        count_feature_names.append('count_saccades')
                
        count_features = np.array(count_features)

        # features for eye closures
        eye_closure_features, eye_closure_feature_names = compute_eye_closure_features(eye_closures, eye_blink,
                                                                                       blink_threshold=blink_threshold,
                                                                                       window_size=window_size,
                                                                                       flag_use_eye_state_label=True,
                                                                                       min_duration=min_blink_duration,
                                                                                       close_labels=[1],
                                                                                       velocity_threshold=velocity_threshold,
                                                                                       feature_prefix='eye_closure',
                                                                                       feature_aggregations=feature_aggregations)

        # saccadic features
        features_sacc, feature_names_sacc = compute_saccadic_features(list_dicts['saccades'],
                                                                      x_angles=x_dva,
                                                                      y_angles=y_dva,
                                                                      x_vels=vel_x, y_vels=vel_y,
                                                                      feature_prefix='saccade',
                                                                      feature_aggregations=feature_aggregations)

        # Texas features for saccads
        texas_features, texas_feature_names = get_features_for_smoothed_saccades(list_dicts['saccades'],
                                                                                 x_dva,
                                                                                 y_dva,
                                                                                 x_smo,
                                                                                 y_smo,
                                                                                 vel_x,
                                                                                 vel_y,
                                                                                 acc_x,
                                                                                 acc_y,
                                                                                 sampling_rate=sampling_rate,
                                                                                 smoothing_window_length=smoothing_window_length,
                                                                                 min_fixation_duration=minDurFix / 1000.,
                                                                                 feature_prefix='saccade',
                                                                                 feature_aggregations=feature_aggregations)
                                                                                 
        # Texas features for fixations
        texas_features_fixations, texas_feature_names_fixations = get_features_for_smoothed_fixations(list_dicts['fixations'],
                                                                                 x_dva,
                                                                                 y_dva,
                                                                                 x_smo,
                                                                                 y_smo,
                                                                                 vel_x,
                                                                                 vel_y,
                                                                                 acc_x,
                                                                                 acc_y,
                                                                                 sampling_rate=sampling_rate,
                                                                                 feature_prefix='fixation',
                                                                                 feature_aggregations=feature_aggregations)
                                                                          
                                                                          
        tmp_combined_features = np.concatenate([features_pupils,
                                                features_gaze_entropy, count_features, 
                                                eye_closure_features, 
                                                features_sacc, texas_features,
                                                texas_features_fixations])
        if i == 0:        
            combined_features = np.zeros([data.shape[0],len(tmp_combined_features)])
        combined_features[i,:] = tmp_combined_features
        
    combined_feature_names = list(feature_names_pupils) +\
                             list(feature_name_gaze_entropy) +\
                             list(count_feature_names) +\
                             list(eye_closure_feature_names) +\
                             list(feature_names_sacc) +\
                             list(texas_feature_names) +\
                             list(texas_feature_names_fixations)
    return combined_features, combined_feature_names




# computes all features
#
# params:
#    pixel_data: intances x n x 2 matrix containing pixel data
#    dva_data: instances x n x 2 matrix containing dva data
#    vel_data: instances x n x 2 matrix containing velocity data
#    minDurFix: minDurFix parameter for event detection (minimal fixation duration in ms)
#    sampling_rate: sampling rate
#    window_size: window of ms used to determine start and end of a blink
#    velocity_threshold: velocity threshold for the detection of the beginning and ending of a blink
#    smoothing_window_length: smoothing window length for smoothing the velocities (Texas algorithm)
#    feature_aggregations: list of aggregations performed on list of values
def compute_velocity_features(pixel_data,
                              dva_data,
                              vel_data,
                              minDurFix=20,
                              sampling_rate=1000,
                              window_size=100,
                              velocity_threshold=0.1,
                              smoothing_window_length=0.007,
                              feature_aggregations=['mean', 'std', 'median', 'skew', 'kurtosis'],
                              verbose = 0):
    if verbose == 0:
        disable = True
    else:
        disable = False
        
    for i in tqdm(np.arange(pixel_data.shape[0]),disable = disable):
        x_pixel = pixel_data[i, :, 0]
        y_pixel = pixel_data[i, :, 1]
        
        x_dva = dva_data[i, :, 0]
        y_dva = dva_data[i, :, 1]
        
        corrupt = np.array(np.isnan(x_dva),dtype=np.int32)
        
        # apply smoothing like in https://digital.library.txstate.edu/handle/10877/6874
        smooth_vals = smoothing.smooth_data(x_dva, y_dva)
        x_smo = smooth_vals['x_smo']
        y_smo = smooth_vals['y_smo']
        vel_x = smooth_vals['vel_x']
        vel_y = smooth_vals['vel_y']
        vel = smooth_vals['vel']
        acc_x = smooth_vals['acc_x']
        acc_y = smooth_vals['acc_y']
        acc = smooth_vals['acc']
        
        # get eye movement events (fixations, saccades)
        list_dicts, event_df = get_sacc_fix_lists_dispersion(x_dva, y_dva,
                                                      corrupt = corrupt,
                                                      sampling_rate = sampling_rate)
        
        # add event counts
        count_features = []
        count_feature_names = []
        count_features.append(len(list_dicts['fixations']))
        count_feature_names.append('count_fixations')
        count_features.append(len(list_dicts['errors']))
        count_feature_names.append('count_errors')
        count_features.append(len(list_dicts['saccades']))
        count_feature_names.append('count_saccades')
                
        count_features = np.array(count_features)

        # saccadic features
        features_sacc, feature_names_sacc = compute_saccadic_features(list_dicts['saccades'],
                                                                      x_angles=x_dva,
                                                                      y_angles=y_dva,
                                                                      x_vels=vel_x, y_vels=vel_y,
                                                                      feature_prefix='saccade',
                                                                      feature_aggregations=feature_aggregations)

        # Texas features for saccads
        texas_features, texas_feature_names = get_features_for_smoothed_saccades(list_dicts['saccades'],
                                                                                 x_dva,
                                                                                 y_dva,
                                                                                 x_smo,
                                                                                 y_smo,
                                                                                 vel_x,
                                                                                 vel_y,
                                                                                 acc_x,
                                                                                 acc_y,
                                                                                 sampling_rate=sampling_rate,
                                                                                                                       smoothing_window_length=smoothing_window_length,
                                                                                 min_fixation_duration=minDurFix / 1000.,
                                                                                 feature_prefix='saccade',
                                                                                 feature_aggregations=feature_aggregations)
                                                                                 
        # Texas features for fixations
        texas_features_fixations, texas_feature_names_fixations = get_features_for_smoothed_fixations(list_dicts['fixations'],
                                                                                 x_dva,
                                                                                 y_dva,
                                                                                 x_smo,
                                                                                 y_smo,
                                                                                 vel_x,
                                                                                 vel_y,
                                                                                 acc_x,
                                                                                 acc_y,
                                                                                 sampling_rate=sampling_rate,
                                                                                 feature_prefix='fixation',
                                                                                 feature_aggregations=feature_aggregations)
                                                                          
                                                                          
        tmp_combined_features = np.concatenate([count_features, 
                                                features_sacc, texas_features,
                                                texas_features_fixations])
        if i == 0:        
            combined_features = np.zeros([pixel_data.shape[0],len(tmp_combined_features)])
        combined_features[i,:] = tmp_combined_features
        
    combined_feature_names = list(count_feature_names) +\
                             list(feature_names_sacc) +\
                             list(texas_feature_names) +\
                             list(texas_feature_names_fixations)
    return combined_features, combined_feature_names



# computes all features in prallel
#
# params:
#    data: matrix containing the data
#    data_format: dictionary describing the data
#    screenPX_x: screen horizontal pixels
#    screenPX_y: screen vertical pixels
#    screenCM_x: screen horizontal length (cm)
#    screenCM_y: screen vertical length (cm)
#    distanceCM: distance from head to screen (cm)
#    minDurFix: minDurFix parameter for event detection (minimal fixation duration in ms)
#    sampling_rate: sampling rate
#    blink_threshold: threshold what counts as blink
#    window_size: window of ms used to determine start and end of a blink
#    min_blink_duration: minimal blink duration in ms
#    velocity_threshold: velocity threshold for the detection of the beginning and ending of a blink
#    smoothing_window_length: smoothing window length for smoothing the velocities (Texas algorithm)
#    feature_aggregations: list of aggregations performed on list of values
#    velocity_saccade_detection: flag, what kind of saccade detection algorithm we should use (True-> velocity based; False -> dispersion based)
#    n_jobs: number of jobs to calculate the features
#    verbose: verbosity level of parallel computing
def compute_combined_features_parallel(data,
                              data_format,
                              screenPX_x=1280,
                              screenPX_y=1024,
                              screenCM_x=38,
                              screenCM_y=30.2,
                              distanceCM=68,
                              minDurFix=20,
                              sampling_rate=1000,
                              blink_threshold=0.6,
                              window_size=100,
                              min_blink_duration=1,
                              velocity_threshold=0.1,
                              smoothing_window_length=0.007,
                              feature_aggregations=['mean', 'std', 'median', 'skew', 'kurtosis'],
                              n_jobs = -1,
                              verbose = 10):
                              
    from joblib import Parallel, delayed
    n_jobs = n_jobs    

    data.shape[0]
    use_ids = []
    num_per_fold = int(np.ceil(data.shape[0] / n_jobs))
    for i in range(n_jobs):
        use_ids.append(np.arange(i*num_per_fold,np.min([(i+1)*num_per_fold,data.shape[0]]),1))
    
    # check for empty lists in use_ids
    t_use_ids = []
    for i in range(len(use_ids)):
        if len(use_ids[i]) > 0:
            t_use_ids.append(use_ids[i])
    use_ids = t_use_ids
    
    num_parallel = len(use_ids)
    
    X_features_list = Parallel(n_jobs=num_parallel,verbose=verbose)(delayed(compute_combined_features)(data[use_ids[i]],
                                                              data_format,
                                                              screenPX_x,
                                                              screenPX_y,
                                                              screenCM_x,
                                                              screenCM_y,
                                                              distanceCM,
                                                              minDurFix,
                                                              sampling_rate,
                                                              blink_threshold,
                                                              window_size,
                                                              min_blink_duration,
                                                              velocity_threshold,
                                                              smoothing_window_length,
                                                              feature_aggregations,
                                                              verbose) for i in range(num_parallel))

    res, feature_names = zip(*X_features_list)
    X_features = np.zeros([data.shape[0],res[0].shape[1]])
    start_id = 0
    for i in range(len(res)):
        cur_data = res[i]
        X_features[start_id:start_id + cur_data.shape[0]] = cur_data
        start_id += cur_data.shape[0]
    return X_features, feature_names[0]