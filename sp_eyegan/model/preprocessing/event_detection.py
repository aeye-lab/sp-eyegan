import pandas as pd
import numpy as np
import os
import re
import warnings




class Screen:  # properties of the screen
    def __init__(self, screenPX_x, screenPX_y,screenCM_x, screenCM_y, dist):
        self.px_x = screenPX_x # screen width in pixels
        self.px_y = screenPX_y
        self.cm_x = screenCM_x # screen width in cm
        self.cm_y = screenCM_y
        # maximal/minimal screen coordinates in degrees of visual angle:
        self.x_max = pix2deg(screenPX_x-1, screenPX_x, screenCM_x, dist)
        self.y_max = pix2deg(screenPX_y-1, screenPX_y, screenCM_y, dist)
        self.x_min = pix2deg(0, screenPX_x, screenCM_x, dist)
        self.y_min = pix2deg(0, screenPX_y, screenCM_y, dist)
        
    

class Experiment:
    def __init__(self, screenPX_x, screenPX_y,screenCM_x, screenCM_y, dist, sampling):        
        self.sampling = sampling # sampling rate in Hz
        self.dist = dist      # eye-to-screen distance in cm
        self.screen = Screen(screenPX_x, screenPX_y,screenCM_x, screenCM_y, dist)


def pix2deg(pix, screenPX,screenCM,distanceCM, adjust_origin=True):
    # Converts pixel screen coordinate to degrees of visual angle
    # screenPX is the number of pixels that the monitor has in the horizontal
    # axis (for x coord) or vertical axis (for y coord)
    # screenCM is the width of the monitor in centimeters
    # distanceCM is the distance of the monitor to the retina 
    # pix: screen coordinate in pixels
    # adjust origin: if origin (0,0) of screen coordinates is in the corner of the screen rather than in the center, set to True to center coordinates
    pix=np.array(pix)
    # center screen coordinates such that 0 is center of the screen:
    if adjust_origin: 
        pix = pix-(screenPX-1)/2 # pixel coordinates start with (0,0) 
    # eye-to-screen-distance in pixels
    distancePX = distanceCM*(screenPX/screenCM)
    return np.arctan2(pix,distancePX) * 180/np.pi #  *180/pi wandelt bogenmass in grad

#---------------------------------------------------
# Compute velocity times series from 2D position data
#---------------------------------------------------
# adapted from Engbert et al.  Microsaccade Toolbox 0.9
# x: array of shape (N,2) (x und y screen or visual angle coordinates of N samples in *chronological* order)
# returns velocity in deg/sec or pix/sec
def vecvel(x, sampling_rate=1000, smooth=True, sample_diff = False):
    N = x.shape[0]  
    v = np.zeros((N,2)) # first column for x-velocity, second column for y-velocity
    if smooth: # v based on mean of preceding 2 samples and mean of following 2 samples
        v[2:N-2, :] =  (sampling_rate/6)*(x[4:N,:] + x[3:N-1,:] - x[1:N-3,:] - x[0:N-4,:])
        # *SAMPLING => pixeldifferenz pro sec
        # v[n,:]: Differenz zwischen mean(sample_n-2, sample_n-1) und mean(sample_n+1, sample_n+2) (=> durch 2 teilen); jetzt ist aber Schrittweite 3 sample lang (von n-1.5 bis n+1.5) => durch 3 teilen => insgesamt durch 6 teilen
        # v based on preceding sample and following sample for second and penultimate sample
        v[1,:] = (sampling_rate/2)*(x[2,:] - x[0,:])
        v[N-2,:] = (sampling_rate/2)*(x[N-1,:] - x[N-3,:])
    else:
        if not sample_diff:
            v[1:N-1,] = (sampling_rate/2)*(x[2:N,:] - x[0:N-2,:]) # differenz Vorgänger sample und nachfolger sample; beginnend mit 2.sample bis N
            #/2 weil dx differenz zwischen voergänger und nachfolger sample ist => schrittweite ist also 2
        else:
            v[1:N,] = (sampling_rate)*(x[1:N,:] - x[0:N-1,:])
    return v

#! corruptSamples nach Sakkadenerkennung anwenden, damit Chronologie erhalten bleibt
#! es kann nach Anwendung von corruptSamples Fixationen und Sakkaden der Laenge 1 geben.
def corruptSamplesIdx(x,y, x_max, y_max, x_min, y_min, theta=0.6, samplingRate=1000):
    # x,y: x,y coordinates in degrees of visual anngle
    # x,y must be chronologically ordered sample sequences, missing values have NOT been removed yet
    # max_x... max/min coordinates for samples to fall on the screen; 
    # theta: velocity threshold in degrees/ms   
    x = np.array(x)
    y = np.array(y)
    ## Find Offending Samples
    # samples that exceed velocity threshold
    ## adjust theta depending on sampling rate
    theta=theta*1000/samplingRate
    x2 = np.append(x[0],x[:-1]) # x2 is sample that precedes x (numpy.roll not used here beacuse it inserts last element in first position)
    y2 = np.append(y[0],y[:-1]) 
    distTrv = np.sqrt(np.power((x-x2),2) + np.power((y-y2),2)) 
    fast_ix = np.where(distTrv>theta)[0] # too fast samples; np.where returns tuple
    # Missing samples
    mis_ix = np.where(np.isnan(x) | np.isnan(y))[0] # np.where returns tuple of len 1
    # samples outside the screen
    out_ix = np.where(np.greater(x, x_max) | np.greater(y, y_max) | np.greater(x_min,x) | np.greater(y_min,y))[0] 
    #RuntimeWarning: invalid value encountered in greater => is ok; comparison with nan yields nan
    # remove samples,where gaze is completely still => scheint auch recording Fehler zu sein 
    # TODO: macht das Sinn? Hiervon sind meist nur einzelne samples betroffen. erstmal raus. ggf wieder einkommentieren.
    #still_ix = np.where(np.equal(x,np.roll(x, shift=-1)) & np.equal(x,np.roll(x, shift=1)) & np.equal(y,np.roll(y, shift=-1)) & np.equal(y,np.roll(y, shift=1)))[0]
    # return all bad samples
    return np.sort(np.unique(np.concatenate((mis_ix, out_ix,fast_ix))))

# same as microsacc, but returns only issac
# params:
#   x: 2-d matrix with input
#   corrupt: 1-d vector containing the corrupt label (corrupt -> 1, not-corrupt -> 0)
#   vfac: vfac params
#   sampling_rate: sampling rate used
#   threshold: threshold for sacc detection
#   vel_smooth: indicator for using velocity smoothing
#   vel_sample_diff: indicating if we want to use sample differences for the velocities
#   input_velocity: flag, indicating whether the input is the velocity of differences
def issac(x,corrupt = None,vfac=6,min_dur=10,sampling_rate=1000, threshold=(10,10),
            vel_smooth = True,vel_sample_diff=False,
            input_velocity = False): # median/mean threshold von allen texten: (9,6)
    # Compute velocity
    if not input_velocity:
        v = vecvel(x,sampling_rate=sampling_rate,smooth=vel_smooth,sample_diff=vel_sample_diff)
    else:
        v = x
    if threshold: # global threshold provided
        msdx, msdy = threshold[0], threshold[1]
    else:  
        # Compute threshold
        if corrupt is not None:
            use_v = v[np.where(corrupt == 0)[0]]
        else:
            use_v = v
        msdx = np.sqrt( np.nanmedian(np.power(use_v[:,0] - np.nanmedian(use_v[:,0]),2))) # median-based std of x-velocity
        msdy = np.sqrt( np.nanmedian(np.power(use_v[:,1] - np.nanmedian(use_v[:,1]),2)))
        if msdx < 1e-10 or msdy < 1e-10:
            warnings.warn('Warning: no saccades found (e.g. velocities too small or all zero)!')
            return np.zeros(len(x))            
        # Removed the assert, because we could have input data with high percentage of corrupt samples (vel = 0)
        """
        assert  msdx>1e-10 # there must be enough variance in the data
        assert  msdy>1e-10
        """
    radiusx = vfac*msdx # x-radius of elliptic threshold
    radiusy = vfac*msdy # y-radius of elliptic threshold
    # test if sample is within elliptic threshold
    test = np.power((v[:,0]/radiusx),2) + np.power((v[:,1]/radiusy),2) # test is <1 iff sample within ellipse
    indx = np.where(np.greater(test,1))[0] # indices of candidate saccades; runtime warning because of nans in test => is ok, the nans come from nans in x
    # Determine saccades
    N = len(indx) # anzahl der candidate saccades
    dur = 1
    a = 0 # (möglicher) begin einer saccade
    k = 0 # (möglisches) ende einer saccade, hierueber wird geloopt
    issac = np.zeros(len(x)) # codes if x[issac] is a saccade
    # Loop over saccade candidates
    while k<N-1:  # loop over candidate saccades
        if indx[k+1]-indx[k]==1: # saccade candidates, die aufeinanderfolgen
            dur = dur + 1 # erhoehe sac dur
        else:  # wenn nicht (mehr) in saccade
            # Minimum duration criterion (exception: last saccade)
            if dur>=min_dur: # schreibe saccade, sofern MINDUR erreicht wurde
                issac[indx[a]:indx[k]+1] = 1 # code as saccade from onset to offset        
            a = k+1 # potential onset of next saccade
            dur = 1 # reset duration  
        k = k+1
    # Check minimum duration for last microsaccade
    if  dur>=min_dur and len(indx) > a and len(indx) > k:
        issac[indx[a]:indx[k]+1] = 1 # code as saccade from onset to offset       
    return issac
  
def detect_saccades(x_deg, y_deg, corrupt = None,
                    pp_params=(None, 6),  minDurFix=20, minDurSac=6,
                    sampling = 1000,
                    screenPX_x = 1280, screenPX_y = 1024, 
                    screenCM_x = 38, screenCM_y = 30.2, dist = 68,
                    input_velocity = False):
    expt = Experiment(screenPX_x = screenPX_x, screenPX_y = screenPX_y, screenCM_x = screenCM_x, screenCM_y = screenCM_y, dist = dist, sampling=sampling)
    d = {'x_deg':x_deg,
          'y_deg':y_deg}
    d = pd.DataFrame(d)
    
    d['sac'] = issac(np.array(d[['x_deg', 'y_deg']]), corrupt = corrupt, sampling_rate=sampling, 
                        threshold=pp_params[0], min_dur=pp_params[1],
                        input_velocity = input_velocity).astype(bool) 
    
    # identify corrupt samples    
    if input_velocity is False:
        corruptIdx = corruptSamplesIdx(d.x_deg, d.y_deg, x_max=expt.screen.x_max, y_max=expt.screen.y_max, x_min=expt.screen.x_min, y_min=expt.screen.y_min, theta=0.6, samplingRate=sampling)
    else:
        if corrupt is None:
            corruptIdx = []
        else:
            corruptIdx = list(np.where(corrupt == 1)[0])
    
    # corrupt samples
    d['corrupt'] = d.index.isin(corruptIdx)
    d['event'] = np.where(d.corrupt, 3, np.where(d.sac,2,1))
    return d


###################################################################
#
# dispersion algorithm
#
###################################################################

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

