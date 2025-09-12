# adapt the code from https://digital.library.txstate.edu/handle/10877/6874
# smooth the velocities and degrees of visual angle
# written by Paul Prasse


import numpy as np
from scipy.signal import savgol_coeffs
from scipy.special import factorial
from scipy.signal import lfilter

# 
# params:
#   x_dva: raw input of degrees of visual angle for x-axis
#   y_dva: raw input of degrees of visual angle for y-axis
#   n: order of polynomial fit
#   smoothing_window_length: smoothing window length in seconds
#   sampling_rate: sampling rate
def smooth_data(x_dva, y_dva,
                n=2, smoothing_window_length = 0.007,
                sampling_rate = 1000):
    # calculate f, to set up Savitzky-Golar filter setting
    f = np.ceil(smoothing_window_length*sampling_rate)
    if np.mod(f,2)!= 1:
        for i in np.arange(f,100,1):
            if np.mod(i,2) == 1:
                f = i
                break
    if f < 5:
        f = 5
    
    g = np.array([savgol_coeffs(f, n, deriv=d, use='dot') for d in range(n+1)]).T / factorial(np.arange(n+1))
    
    x_smo = lfilter(g[:,0],1, x_dva)
    y_smo = lfilter(g[:,0],1, y_dva)
    
    # calculate the velocities and accelerations
    vel_x = lfilter(g[:,1],1, x_dva) * sampling_rate
    vel_x = vel_x * -1.
    vel_y = lfilter(g[:,1],1, y_dva) * sampling_rate
    vel_y = vel_y * -1.
    vel   = np.sqrt(vel_x**2 + vel_y**2)
    
    acc_x = lfilter(g[:,2],1, x_dva) * sampling_rate**2
    acc_y = lfilter(g[:,2],1, y_dva) * sampling_rate**2
    acc   = np.sqrt(acc_x**2 + acc_y**2)
    
    return {'x_smo':x_smo,
            'y_smo':y_smo,
            'vel_x':vel_x,
            'vel_y':vel_y,
            'vel':vel,
            'acc_x':acc_x,
            'acc_y':acc_y,
            'acc':acc}