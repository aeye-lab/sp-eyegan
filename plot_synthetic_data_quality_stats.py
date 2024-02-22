import os
import numpy as np
import random
from tqdm import tqdm
import sys
import joblib
import seaborn as sns
import pandas as pd
from typing import Dict
from typing import List
from typing import Tuple



import tensorflow
import tensorflow as tf
from sp_eyegan.model import eventGAN as eventGAN
from sp_eyegan.model import vae_baseline as vae
from scipy.stats import ttest_ind, ttest_1samp


def vel_to_dva(vel_data, x_start = 0,
             y_start = 0):
    x_vel = vel_data[:,0]
    y_vel = vel_data[:,1]
    x_px  = []
    y_px  = []
    cur_x_pos = x_start
    cur_y_pos = y_start
    for i in range(len(x_vel)):
        x_px.append(cur_x_pos + x_vel[i])
        y_px.append(cur_y_pos + y_vel[i])
        cur_x_pos = x_px[-1]
        cur_y_pos = y_px[-1]
    return np.concatenate([np.expand_dims(np.array(x_px),axis=1),
                           np.expand_dims(np.array(y_px),axis=1)],axis=1)

def get_fixation_stats(input_data):
    real_x_vels      = []
    real_y_vels      = []
    real_vels        = []
    real_dispersions = []
    cur_vel_data     = input_data
    for i in tqdm(np.arange(cur_vel_data.shape[0])):
        cur_vels = cur_vel_data[i]
        cur_dva = vel_to_dva(cur_vels)
        try:
            end_id = np.where(np.logical_and(cur_vels[:,0] == 0,
                                             cur_vels[:,1] ==0 ))[0][0]
        except:
            end_id = len(cur_vels)
        if end_id == 0:
            continue
        cur_vels = cur_vels[0:end_id]
        cur_dva = cur_dva[0:end_id]
        x_dva = cur_dva[:,0]
        y_dva = cur_dva[:,1]
        real_x_vels.append(cur_vels[:,0])
        real_y_vels.append(cur_vels[:,1])
        
        vels =  np.power(np.array(real_x_vels[-1]),2) +\
                np.power(np.array(real_y_vels[-1]),2)
        vels = np.sqrt(vels)
        real_vels.append(vels)
        x_amp = np.abs(np.max(x_dva) - np.min(x_dva))
        y_amp = np.abs(np.max(y_dva) - np.min(y_dva))
        cur_dispersion = x_amp + y_amp
        real_dispersions.append(cur_dispersion)
    
    return real_vels, real_x_vels,real_y_vels, real_dispersions


def get_saccade_stats(input_data, max_velocity = 0.5):
    real_x_vels      = []
    real_y_vels      = []
    real_vels        = []
    real_amplitudes  = []
    real_x_accs      = []
    real_y_accs      = []
    real_accs        = []
    cur_vel_data     = input_data
    for i in tqdm(np.arange(cur_vel_data.shape[0])):
        cur_vels = cur_vel_data[i]
        cur_dva = vel_to_dva(cur_vels)
        try:
            end_id = np.where(np.logical_and(cur_vels[:,0] == 0,
                                             cur_vels[:,1] ==0 ))[0][0]
        except:
            end_id = len(cur_vels)
        if end_id == 0:
            continue
        cur_vels = cur_vels[0:end_id]
        cur_dva = cur_dva[0:end_id]
        x_dva = cur_dva[:,0]
        y_dva = cur_dva[:,1]
        real_x_vels.append(cur_vels[:,0])
        real_y_vels.append(cur_vels[:,1])
        
        vels =  np.power(np.array(real_x_vels[-1]),2) +\
                np.power(np.array(real_y_vels[-1]),2)
        vels = np.sqrt(vels)
        real_vels.append(vels)        
        
        x_accs = cur_vels[1:,0] - cur_vels[:-1,0]
        y_accs = cur_vels[1:,1] - cur_vels[:-1,0]
        
        accs = np.power(np.array(x_accs),2) +\
                np.power(np.array(y_accs),2)
        accs = np.sqrt(accs)
        
        real_x_accs.append(x_accs)
        real_y_accs.append(y_accs)
        real_accs.append(accs)
        cur_complete_vels = np.sqrt(np.power(cur_vels[:,0],2) + np.power(cur_vels[:,1],2))
        cur_complete_vels[cur_complete_vels > max_velocity] = max_velocity
        cur_amplitude = np.sqrt(np.power(x_dva[0] - x_dva[-1],2) + np.power(y_dva[0] - y_dva[-1],2))
        real_amplitudes.append(cur_amplitude)
    
    return real_vels, real_x_vels, real_y_vels, real_accs, real_x_accs, real_y_accs, np.array(real_amplitudes)


# calculate the kl divergence
def kl_divergence(p, q):
    from math import log2
    from math import sqrt
    rel_etropies = [p[i] * log2(p[i]/q[i]) for i in range(len(p))]
    return np.sum(np.array(rel_etropies,dtype=np.float32))
 
# calculate the js divergence
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)



def js_divergence_sampling( values_1, values_2,
                            bins,
                            epsilon =  0.00001,
                            iterations = 10,
                            number_per_iter = 10000,
                            random_state = 42):
    np.random.seed(random_state)
    values = []
    for iter in range(iterations):
        idx_1 = np.random.choice(len(values_1), number_per_iter)
        idx_2 = np.random.choice(len(values_2), number_per_iter)
        counts_1,_ = np.histogram(values_1[idx_1], bins = bins, density=True)
        counts_1 /= np.sum(counts_1)
        counts_2,_ = np.histogram(values_2[idx_2], bins = bins, density=True)
        counts_2 /= np.sum(counts_2)
        values.append(js_divergence(np.array(counts_1)  + epsilon,
                                      np.array(counts_2) + epsilon))
    return values

def draw_display(dispsize: Tuple[int, int], imagefile=None):
    # construct screen (black background)
    # dots per inch
    img = image.imread(imagefile)
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(img)

    return fig, ax



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
        pix = pix-(screenPX)/2 # pixel coordinates start with (0,0) 
    # eye-to-screen-distance in pixels of screen
    distancePX = distanceCM*(screenPX/screenCM)
    return np.arctan2(pix,distancePX) * 180/np.pi #  *180/pi wandelt bogenmass in grad


def deg2pix(deg, screenPX, screenCM, distanceCM, adjust_origin = True, offsetCM = 0):
    # Converts degrees of visual angle to pixel screen coordinates
    # screenPX is the number of pixels that the monitor has in the horizontal
    # screenCM is the width of the monitor in centimeters
    # distanceCM is the distance of the monitor to the retina 
    phi = np.arctan2(1,distanceCM)*180/np.pi
    pix = deg/(phi/(screenPX/(screenCM)))
    if adjust_origin:
        pix += (screenPX/2)
    if offsetCM != 0:
        offsetPX = offsetCM*(screenPX/screenCM)
        pix += offsetPX
    return pix
    
    


# params
GPU = 0
data_dir = 'data/'
plot_dir = 'plots/'
result_dir = 'results/'

model_name = 'GAN'
real_name = 'real'
gauss_name = 'Gauss'


# load data
column_dict = joblib.load(data_dir + 'column_dict.joblib')
fixation_matrix = np.load(data_dir + 'fixation_matrix_gazebase_vd_text.npy')
saccade_matrix = np.load(data_dir  + 'saccade_matrix_gazebase_vd_text.npy')

fix_window_size = fixation_matrix.shape[1]
sac_window_size = saccade_matrix.shape[1]
gen_kernel_sizes_fixation = [fix_window_size,8,4,2]
gen_kernel_sizes_saccade = [sac_window_size,8,4,2]
    
# params for NN
random_size = 32
gen_filter_sizes = [16,8,4,2]
channels = 2
relu_in_last = False
batch_size = 256

dis_kernel_sizes = [8,16,32]
dis_fiter_sizes = [32,64,128]
dis_dropout = 0.3

sample_size = 1000
max_velocity = 0.5
seed = 42
np.random.seed(seed)

model_config_fixation = {'gen_kernel_sizes':gen_kernel_sizes_fixation,
                'gen_filter_sizes':gen_filter_sizes,
                'dis_kernel_sizes':dis_kernel_sizes,
                'dis_fiter_sizes':dis_fiter_sizes,
                'dis_dropout':dis_dropout,
                'window_size':fix_window_size,
                'channels':channels,
                'batch_size':batch_size,
                'random_size':random_size,
                'relu_in_last':relu_in_last,
               }

model_config_saccade = {'gen_kernel_sizes':gen_kernel_sizes_saccade,
                'gen_filter_sizes':gen_filter_sizes,
                'dis_kernel_sizes':dis_kernel_sizes,
                'dis_fiter_sizes':dis_fiter_sizes,
                'dis_dropout':dis_dropout,
                'window_size':sac_window_size,
                'channels':channels,
                'batch_size':batch_size,
                'random_size':random_size,
                'relu_in_last':relu_in_last,
               }


fixation_path  = 'event_model/fixation_model_text'
saccade_path   = 'event_model/saccade_model_text'








flag_train_on_gpu = True
if flag_train_on_gpu:
    import tensorflow as tf
    # select graphic card
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1.
    config.gpu_options.allow_growth = True
    tf_session = tf.compat.v1.Session(config=config)
else:
    import tensorflow as tf
    # select graphic card
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
print('shape of fixation_matrix: ' + str(fixation_matrix.shape))
print('shape of saccade_matrix: ' + str(saccade_matrix.shape))


## Data for statitical baseline
### select model for baseline method (statistical method)
gen_data_statitical = pd.read_csv('data/statistical_baseline_fix_data.txt',sep=';',header = None)
gen_data_statitical.columns = ['vel','type','frame']
gen_data_statitical.head()


fixation_index = 0
saccade_index = 1
velocities = np.array(gen_data_statitical['vel'])
event_type = np.array(gen_data_statitical['type'])
print('number fix points: ' + str(np.sum(event_type == fixation_index)))
print('number sac points: ' + str(np.sum(event_type == saccade_index)))


fix_vels_statistical = []
sac_vels_statistical = []
sac_acc_statistical  = []

# transfrom °/s to °/ms
scaling_factor = 1000.
prev_event = -1
cur_fixation = []
cur_saccade  = []
for i in tqdm(np.arange(len(velocities))):
    cur_event = event_type[i]
    cur_vel   = velocities[i]
    if cur_event == prev_event:
        if cur_event == fixation_index:
            cur_fixation.append(cur_vel/scaling_factor)
        elif cur_event == saccade_index:
            cur_saccade.append(cur_vel/scaling_factor)
    else:
        if prev_event == fixation_index and len(cur_fixation) > 0:
            fix_vels_statistical.append(np.array(cur_fixation))
        elif prev_event == saccade_index and len(cur_saccade) > 0:
            sac_vels_statistical.append(np.array(cur_saccade))
            if len(cur_saccade) > 2:
                sac_acc_statistical.append(np.array(cur_saccade)[1:] - np.array(cur_saccade)[0:-1])
        
        if cur_event == fixation_index:
            cur_fixation = [cur_vel/scaling_factor]
            cur_saccade = []
        elif cur_event == saccade_index:
            cur_saccade = [cur_vel/scaling_factor]
            cur_fixation = []
    prev_event = cur_event

## Evaluate Fixations
    
# EyeSyn
from scipy.io import loadmat
eye_syn_fixation_data = loadmat('data/eyeSyn_velocities.mat')
eye_syn_fixation_data = eye_syn_fixation_data['out_data']


tf.keras.backend.clear_session()
fix_model = eventGAN.eventGAN(model_config_fixation)
fix_model.load_model(fixation_path)

noise = tf.random.normal([sample_size, random_size], seed = seed)
gen_fixations = np.array(fix_model.generator(noise, training=False),dtype=np.float32)

print('shape of gen_fixations: ' + str(gen_fixations.shape))
#pd.DataFrame(gen_fixations.flatten()).describe()



tf.keras.backend.clear_session()
vae_model = vae.VAE(vae.get_vae_encoder(64, 2, 2), vae.get_vae_decoder(2, 2))
vae_model.load_model('event_model/vae_fixation_10')
noise = tf.random.normal([sample_size * 2, 2], seed = seed)
vae_fixations = np.array(vae_model.decoder(noise, training=False),dtype=np.float32)
vae_fixations = np.concatenate([vae_fixations[0:sample_size],
                                vae_fixations[sample_size:]],axis=1)
vae_fixations = vae_fixations[:,0:100,:] -0.5

print('shape of gen_fixations: ' + str(vae_fixations.shape))
#pd.DataFrame(vae_fixations.flatten()).describe()


rand_ids        = np.random.permutation(np.arange(fixation_matrix.shape[0]))
rand_ids_1      = rand_ids[0:sample_size]
rand_ids_2      = rand_ids[sample_size:sample_size+sample_size]
orig_fixations  = fixation_matrix[rand_ids_1,:,4:6]
orig_fixations[orig_fixations > max_velocity] = max_velocity
orig_fixations[orig_fixations < -max_velocity] = -max_velocity

orig_fixations_2 = fixation_matrix[rand_ids_2,:,4:6]
orig_fixations_2[orig_fixations_2 > max_velocity] = max_velocity
orig_fixations_2[orig_fixations_2 < -max_velocity] = -max_velocity


real_vels, real_x_vels, real_y_vels, real_dispersions = get_fixation_stats(orig_fixations)
real_vels_2, real_x_vels_2, real_y_vels_2, real_dispersions_2 = get_fixation_stats(orig_fixations_2)
fake_vels, fake_x_vels, fake_y_vels, fake_dispersions = get_fixation_stats(gen_fixations)
eyesyn_vels, eyesyn_x_vels, eyesyn_y_vels, eyesyn_dispersions =  get_fixation_stats(eye_syn_fixation_data)
vae_vels, vae_x_vels, vae_y_vels, vae_dispersions = get_fixation_stats(vae_fixations)


x_vel_mean = np.nanmean([item for sublist in real_x_vels for item in sublist])
x_vel_std = np.nanstd([item for sublist in real_x_vels for item in sublist])

y_vel_mean = np.nanmean([item for sublist in real_y_vels for item in sublist])
y_vel_std = np.nanstd([item for sublist in real_y_vels for item in sublist])
gauss_fixations = np.concatenate([np.random.normal(x_vel_mean,x_vel_std,(sample_size,fix_window_size,1)),
                                np.random.normal(y_vel_mean,y_vel_std,(sample_size,fix_window_size,1))],axis=2)
gauss_vels, gauss_x_vels, gauss_y_vels, gauss_dispersions = get_fixation_stats(gauss_fixations)

x_vel_mean = np.nanmean([item for sublist in real_x_vels for item in sublist])
x_vel_std = np.nanstd([item for sublist in real_x_vels for item in sublist])
#print(x_vel_mean, x_vel_std)

y_vel_mean = np.nanmean([item for sublist in real_y_vels for item in sublist])
y_vel_std = np.nanstd([item for sublist in real_y_vels for item in sublist])
#print(y_vel_mean, y_vel_std)

## velocity
real   = np.array([item for sublist in real_vels for item in sublist])
real_2 = np.array([item for sublist in real_vels_2 for item in sublist])
fake   = np.array([item for sublist in fake_vels for item in sublist])
gauss  = np.array([item for sublist in gauss_vels for item in sublist])
eyesyn = np.array([item for sublist in eyesyn_vels for item in sublist])
stat   = np.array([item for sublist in fix_vels_statistical for item in sublist])
f_vae  = np.array([item for sublist in vae_vels for item in sublist])

min_val = np.min(list(real) + list(fake) + list(gauss) + list(real_2) + list(eyesyn) + list(stat) + list(f_vae))
max_val = np.max(list(real) + list(fake) + list(gauss) + list(real_2) + list(eyesyn) + list(stat) + list(f_vae))
num_bins = 100
epsilon = 0.00001
number_per_iter = 10000
iterations = 10
bins = np.linspace(min_val,max_val,num_bins)

real_counts,_   = np.histogram(real, bins = bins, density=True)
real_counts /= np.sum(real_counts)
real_counts_2,_   = np.histogram(real_2, bins = bins, density=True)
real_counts_2 /= np.sum(real_counts_2)
fake_counts,_   = np.histogram(fake, bins = bins, density=True)
fake_counts /= np.sum(fake_counts)
gauss_counts,_  = np.histogram(gauss, bins = bins, density=True)
gauss_counts /= np.sum(gauss_counts)
eyesyn_counts,_  = np.histogram(eyesyn, bins = bins, density=True)
eyesyn_counts /= np.sum(eyesyn_counts)
stat_counts,_  = np.histogram(stat, bins = bins, density=True)
stat_counts /= np.sum(stat_counts)
vae_counts,_  = np.histogram(f_vae, bins = bins, density=True)
vae_counts /= np.sum(vae_counts)


js_divergence_vel_fake  = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(fake_counts) + epsilon)

js_divergence_vel_gauss = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(gauss_counts) + epsilon)

js_divergence_vel_real = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(real_counts_2) + epsilon)

js_divergence_vel_syneye = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(eyesyn_counts) + epsilon)

js_divergence_vel_stat = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(stat_counts) + epsilon)

js_divergence_vel_vae = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(vae_counts) + epsilon)

print('Fixation velocity one fold')
print('JS(real || stat): ' + str(np.round(js_divergence_vel_stat,decimals = 3)))
print('JS(real || VAE): ' + str(np.round(js_divergence_vel_vae,decimals = 3)))
print('JS(real || EyeSyn): ' + str(np.round(js_divergence_vel_syneye,decimals = 3)))
print('JS(real || GAN): ' + str(np.round(js_divergence_vel_fake,decimals = 3)))
#print('JS(real || Gauss): ' + str(np.round(js_divergence_vel_gauss,decimals=3)))
print('JS(real || real): ' + str(np.round(js_divergence_vel_real,decimals = 3)))

stat_vals = js_divergence_sampling( real, stat, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
vae_vals = js_divergence_sampling( real, f_vae, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
eyesyn_vals = js_divergence_sampling( real, eyesyn, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
gan_vals = js_divergence_sampling( real, fake, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
real_vals = js_divergence_sampling( real, real_2, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)

model_vals = [stat_vals, vae_vals, eyesyn_vals, gan_vals]
model_means = [np.mean(a) for a in model_vals]
prefixes = ['' for a in range(len(model_means))]
suffixes = ['' for a in range(len(model_means))]
arg_sort = np.argsort(model_means)
prefixes[arg_sort[0]] = '\\textbf{'
suffixes[arg_sort[0]] = '}'

tt_test_pvalue = ttest_ind(model_vals[arg_sort[0]],model_vals[arg_sort[1]],alternative='two-sided').pvalue
if tt_test_pvalue <= 0.05:
    suffixes[arg_sort[0]] += '$^*$'

                                        
print('Fixation velocity ' + str(iterations) + ' folds')
print('JS(real || stat): ' + prefixes[0] + str(np.round(np.mean(stat_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(stat_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[0])
print('JS(real || VAE): ' + prefixes[1] +  str(np.round(np.mean(vae_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(vae_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[1])
print('JS(real || EyeSyn): ' + prefixes[2] +  str(np.round(np.mean(eyesyn_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(eyesyn_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[2])
print('JS(real || GAN): ' + prefixes[3] +  str(np.round(np.mean(gan_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(gan_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[3])
print('JS(real || Real): ' + str(np.round(np.mean(real_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(real_vals)/np.sqrt(iterations),decimals = 3)))

    
# mean velocity
real  = np.array([np.mean(a) for a in real_vels])
real_2 = np.array([np.mean(a) for a in real_vels_2])
fake  = np.array([np.mean(a) for a in fake_vels])
gauss = np.array([np.mean(a) for a in gauss_vels])
eyesyn = np.array([np.mean(a) for a in eyesyn_vels])
f_vae  = np.array([np.mean(a) for a in vae_vels])

min_val = np.min(list(real) + list(fake) + list(gauss) + list(real_2) + list(eyesyn) + list(stat) + list(f_vae))
max_val = np.max(list(real) + list(fake) + list(gauss) + list(real_2) + list(eyesyn) + list(stat) + list(f_vae))
num_bins = 100
epsilon = 0.00001
number_per_iter = 500
iterations = 10
bins = np.linspace(min_val,max_val,num_bins)

real_counts,_   = np.histogram(real, bins = bins, density=True)
real_counts /= np.sum(real_counts)
real_counts_2,_   = np.histogram(real_2, bins = bins, density=True)
real_counts_2 /= np.sum(real_counts_2)
fake_counts,_   = np.histogram(fake, bins = bins, density=True)
fake_counts /= np.sum(fake_counts)
gauss_counts,_  = np.histogram(gauss, bins = bins, density=True)
gauss_counts /= np.sum(gauss_counts)
eyesyn_counts,_  = np.histogram(eyesyn, bins = bins, density=True)
eyesyn_counts /= np.sum(eyesyn_counts)
stat_counts,_  = np.histogram(stat, bins = bins, density=True)
stat_counts /= np.sum(stat_counts)
vae_counts,_  = np.histogram(f_vae, bins = bins, density=True)
vae_counts /= np.sum(vae_counts)


js_divergence_vel_fake  = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(fake_counts) + epsilon)

js_divergence_vel_gauss = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(gauss_counts) + epsilon)

js_divergence_vel_real = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(real_counts_2) + epsilon)

js_divergence_vel_syneye = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(eyesyn_counts) + epsilon)

js_divergence_vel_stat = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(stat_counts) + epsilon)

js_divergence_vel_vae = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(vae_counts) + epsilon)

print('Fixation mean velocity one fold')
print('JS(real || stat): ' + str(np.round(js_divergence_vel_stat,decimals = 3)))
print('JS(real || VAE): ' + str(np.round(js_divergence_vel_vae,decimals = 3)))
print('JS(real || EyeSyn): ' + str(np.round(js_divergence_vel_syneye,decimals = 3)))
print('JS(real || GAN): ' + str(np.round(js_divergence_vel_fake,decimals = 3)))
print('JS(real || real): ' + str(np.round(js_divergence_vel_real,decimals = 3)))

stat_vals = js_divergence_sampling( real, stat, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
vae_vals = js_divergence_sampling( real, f_vae, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
eyesyn_vals = js_divergence_sampling( real, eyesyn, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
gan_vals = js_divergence_sampling( real, fake, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
real_vals = js_divergence_sampling( real, real_2, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
                                        
model_vals = [stat_vals, vae_vals, eyesyn_vals, gan_vals]
model_means = [np.mean(a) for a in model_vals]
prefixes = ['' for a in range(len(model_means))]
suffixes = ['' for a in range(len(model_means))]
arg_sort = np.argsort(model_means)
prefixes[arg_sort[0]] = '\\textbf{'
suffixes[arg_sort[0]] = '}'

tt_test_pvalue = ttest_ind(model_vals[arg_sort[0]],model_vals[arg_sort[1]],alternative='two-sided').pvalue
if tt_test_pvalue <= 0.05:
    suffixes[arg_sort[0]] += '$^*$'

                                        
print('Fixation mean velocity ' + str(iterations) + ' folds')
print('JS(real || stat): ' + prefixes[0] + str(np.round(np.mean(stat_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(stat_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[0])
print('JS(real || VAE): ' + prefixes[1] +  str(np.round(np.mean(vae_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(vae_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[1])
print('JS(real || EyeSyn): ' + prefixes[2] +  str(np.round(np.mean(eyesyn_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(eyesyn_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[2])
print('JS(real || GAN): ' + prefixes[3] +  str(np.round(np.mean(gan_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(gan_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[3])
print('JS(real || Real): ' + str(np.round(np.mean(real_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(real_vals)/np.sqrt(iterations),decimals = 3)))




# dispersion
real   = np.array(real_dispersions)
real_2 = np.array(real_dispersions_2)
fake   = np.array(fake_dispersions)
gauss  = np.array(gauss_dispersions)
eyesyn = np.array(eyesyn_dispersions)
f_vae  = np.array(vae_dispersions)

min_val = np.min(list(real) + list(fake) + list(gauss) + list(real_2) + list(eyesyn) + list(f_vae))
max_val = np.max(list(real) + list(fake) + list(gauss) + list(real_2) + list(eyesyn) + list(f_vae))
num_bins = 100
epsilon = 0.00001
number_per_iter = 500
iterations = 10
bins = np.linspace(min_val,max_val,num_bins)

real_counts,_   = np.histogram(real, bins = bins, density=True)
real_counts /= np.sum(real_counts)
real_counts_2,_   = np.histogram(real_2, bins = bins, density=True)
real_counts_2 /= np.sum(real_counts_2)
fake_counts,_   = np.histogram(fake, bins = bins, density=True)
fake_counts /= np.sum(fake_counts)
gauss_counts,_  = np.histogram(gauss, bins = bins, density=True)
gauss_counts /= np.sum(gauss_counts)
eyesyn_counts,_  = np.histogram(eyesyn, bins = bins, density=True)
eyesyn_counts /= np.sum(eyesyn_counts)
vae_counts,_  = np.histogram(f_vae, bins = bins, density=True)
vae_counts /= np.sum(vae_counts)


js_divergence_vel_fake  = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(fake_counts) + epsilon)

js_divergence_vel_gauss = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(gauss_counts) + epsilon)

js_divergence_vel_real = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(real_counts_2) + epsilon)

js_divergence_vel_syneye = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(eyesyn_counts) + epsilon)

js_divergence_vel_vae = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(vae_counts) + epsilon)
                                      
                                      
print('Fixation dispersion one fold')
print('JS(real || stat): -')
print('JS(real || VAE): ' + str(np.round(js_divergence_vel_vae,decimals = 3)))
print('JS(real || EyeSyn): ' + str(np.round(js_divergence_vel_syneye,decimals = 3)))
print('JS(real || GAN): ' + str(np.round(js_divergence_vel_fake,decimals = 3)))
print('JS(real || real): ' + str(np.round(js_divergence_vel_real,decimals = 3)))


vae_vals = js_divergence_sampling( real, f_vae, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
eyesyn_vals = js_divergence_sampling( real, eyesyn, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
gan_vals = js_divergence_sampling( real, fake, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
real_vals = js_divergence_sampling( real, real_2, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)

model_vals = [vae_vals, eyesyn_vals, gan_vals]
model_means = [np.mean(a) for a in model_vals]
prefixes = ['' for a in range(len(model_means))]
suffixes = ['' for a in range(len(model_means))]
arg_sort = np.argsort(model_means)
prefixes[arg_sort[0]] = '\\textbf{'
suffixes[arg_sort[0]] = '}'

tt_test_pvalue = ttest_ind(model_vals[arg_sort[0]],model_vals[arg_sort[1]],alternative='two-sided').pvalue
if tt_test_pvalue <= 0.05:
    suffixes[arg_sort[0]] += '$^*$'

                                        
print('Fixation velocity ' + str(iterations) + ' folds')
print('JS(real || stat): -')
print('JS(real || VAE): ' + prefixes[0] +  str(np.round(np.mean(vae_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(vae_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[0])
print('JS(real || EyeSyn): ' + prefixes[1] +  str(np.round(np.mean(eyesyn_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(eyesyn_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[1])
print('JS(real || GAN): ' + prefixes[2] +  str(np.round(np.mean(gan_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(gan_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[2])
print('JS(real || Real): ' + str(np.round(np.mean(real_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(real_vals)/np.sqrt(iterations),decimals = 3)))




# Saccade
tf.keras.backend.clear_session()
sac_model = eventGAN.eventGAN(model_config_saccade)
sac_model.load_model(saccade_path)

noise = tf.random.normal([sample_size, random_size], seed = seed)
gen_saccades = np.array(sac_model.generator(noise, training=False),dtype=np.float32)

tf.keras.backend.clear_session()
vae_model = vae.VAE(vae.get_vae_encoder(64, 2, 2), vae.get_vae_decoder(2, 2))
vae_model.load_model('event_model/vae_saccade_10')
noise = tf.random.normal([sample_size, 2], seed = seed)
vae_saccades = np.array(vae_model.decoder(noise, training=False),dtype=np.float32)
vae_saccades = vae_saccades[:,0:30,:] -0.5


rand_ids        = np.random.permutation(np.arange(saccade_matrix.shape[0]))
rand_ids_1      = rand_ids[0:sample_size]
rand_ids_2      = rand_ids[sample_size:sample_size+sample_size]

rand_ids        = np.arange(saccade_matrix.shape[0])
rand_ids        = np.random.permutation(rand_ids)[0:sample_size]
orig_saccades   = saccade_matrix[rand_ids_1,:,4:6]
orig_saccades[orig_saccades > max_velocity] = max_velocity
orig_saccades[orig_saccades < -max_velocity] = -max_velocity

orig_saccades_2 = saccade_matrix[rand_ids_2,:,4:6]
orig_saccades_2[orig_saccades_2 > max_velocity] = max_velocity
orig_saccades_2[orig_saccades_2 < -max_velocity] = -max_velocity


real_vels_sac, real_x_vels_sac, real_y_vels_sac, real_accs_sac, real_x_accs_sac, real_y_accs_sac, real_amplitudes_sac      = get_saccade_stats(orig_saccades)
real_vels_sac_2, real_x_vels_sac_2, real_y_vels_sac_2, real_accs_sac_2, real_x_accs_sac_2, real_y_accs_sac_2, real_amplitudes_sac_2      = get_saccade_stats(orig_saccades_2)
fake_vels_sac, fake_x_vels_sac, fake_y_vels_sac, fake_accs_sac, fake_real_x_accs_sac, fake_y_accs_sac, fake_amplitudes_sac = get_saccade_stats(gen_saccades)
vae_vels_sac, vae_x_vels_sac, vae_y_vels_sac, vae_accs_sac, vae_x_accs_sac, vae_y_accs_sac, vae_amplitudes_sac = get_saccade_stats(vae_saccades)


x_vel_mean_sac = np.nanmean([item for sublist in real_x_vels_sac for item in sublist])
x_vel_std_sac = np.nanstd([item for sublist in real_x_vels_sac for item in sublist])

y_vel_mean_sac = np.nanmean([item for sublist in real_y_vels_sac for item in sublist])
y_vel_std_sac = np.nanstd([item for sublist in real_y_vels_sac for item in sublist])
gauss_saccades = np.concatenate([np.random.normal(x_vel_mean_sac,x_vel_std_sac,(sample_size,sac_window_size,1)),
                                np.random.normal(y_vel_mean_sac,y_vel_std_sac,(sample_size,sac_window_size,1))],axis=2)
gauss_vels_sac, gauss_x_vels_sac, gauss_y_vels_sac, gauss_accs_sac, gauss_real_x_accs_sac, gauss_y_accs_sac, gauss_amplitudes_sac = get_saccade_stats(gauss_saccades)


# peak velocity
real = np.array([np.max(a) for a in real_vels_sac])
real_2 = np.array([np.max(a) for a in real_vels_sac_2])
fake = np.array([np.max(a) for a in fake_vels_sac])
gauss = np.array([np.max(a) for a in gauss_vels_sac])
stat   = np.array([np.max(a) for a in sac_vels_statistical])
f_vae  = np.array([np.max(a) for a in vae_vels_sac])

min_val = np.min(list(real) + list(fake)  + list(real_2)  + list(stat) + list(f_vae))
max_val = np.max(list(real) + list(fake)  + list(real_2)  + list(stat) + list(f_vae))
num_bins = 100
epsilon = 0.00001
number_per_iter = 500
iterations = 10
bins = np.linspace(min_val,max_val,num_bins)

real_counts,_   = np.histogram(real, bins = bins, density=True)
real_counts /= np.sum(real_counts)
real_counts_2,_   = np.histogram(real_2, bins = bins, density=True)
real_counts_2 /= np.sum(real_counts_2)
fake_counts,_   = np.histogram(fake, bins = bins, density=True)
fake_counts /= np.sum(fake_counts)
stat_counts,_  = np.histogram(stat, bins = bins, density=True)
stat_counts /= np.sum(stat_counts)
vae_counts,_  = np.histogram(f_vae, bins = bins, density=True)
vae_counts /= np.sum(vae_counts)


js_divergence_vel_fake  = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(fake_counts) + epsilon)

js_divergence_vel_real = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(real_counts_2) + epsilon)

js_divergence_vel_syneye = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(eyesyn_counts) + epsilon)

js_divergence_vel_stat = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(stat_counts) + epsilon)

js_divergence_vel_vae = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(vae_counts) + epsilon)

print('Saccade peak velocity one fold')
print('JS(real || stat): ' + str(np.round(js_divergence_vel_stat,decimals = 3)))
print('JS(real || VAE): ' + str(np.round(js_divergence_vel_vae,decimals = 3)))
print('JS(real || GAN): ' + str(np.round(js_divergence_vel_fake,decimals = 3)))
print('JS(real || Real): ' + str(np.round(js_divergence_vel_real,decimals = 3)))

print('Saccade peak velocity ' + str(iterations) + ' folds')

stat_vals = js_divergence_sampling( real, stat, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
vae_vals = js_divergence_sampling( real, f_vae, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
gan_vals = js_divergence_sampling( real, fake, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
real_vals = js_divergence_sampling( real, real_2, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)

model_vals = [stat_vals, vae_vals, gan_vals]
model_means = [np.mean(a) for a in model_vals]
prefixes = ['' for a in range(len(model_means))]
suffixes = ['' for a in range(len(model_means))]
arg_sort = np.argsort(model_means)
prefixes[arg_sort[0]] = '\\textbf{'
suffixes[arg_sort[0]] = '}'

tt_test_pvalue = ttest_ind(model_vals[arg_sort[0]],model_vals[arg_sort[1]],alternative='two-sided').pvalue
if tt_test_pvalue <= 0.05:
    suffixes[arg_sort[0]] += '$^*$'
                                       
print('JS(real || stat): ' + prefixes[0] + str(np.round(np.mean(stat_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(stat_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[0])
print('JS(real || VAE): ' + prefixes[1] +  str(np.round(np.mean(vae_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(vae_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[1])
print('JS(real || GAN): ' + prefixes[2] +  str(np.round(np.mean(gan_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(gan_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[2])
print('JS(real || Real): ' + str(np.round(np.mean(real_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(real_vals)/np.sqrt(iterations),decimals = 3)))

# mean velocity
real = np.array([np.mean(a) for a in real_vels_sac])
real_2 = np.array([np.mean(a) for a in real_vels_sac_2])
fake = np.array([np.mean(a) for a in fake_vels_sac])
gauss = np.array([np.mean(a) for a in gauss_vels_sac])
stat   = np.array([np.mean(a) for a in sac_vels_statistical])
f_vae  = np.array([np.mean(a) for a in vae_vels_sac])

min_val = np.min(list(real) + list(fake)  + list(real_2)  + list(stat) + list(f_vae))
max_val = np.max(list(real) + list(fake)  + list(real_2)  + list(stat) + list(f_vae))
num_bins = 100
epsilon = 0.00001
number_per_iter = 500
iterations = 10
bins = np.linspace(min_val,max_val,num_bins)

real_counts,_   = np.histogram(real, bins = bins, density=True)
real_counts /= np.sum(real_counts)
real_counts_2,_   = np.histogram(real_2, bins = bins, density=True)
real_counts_2 /= np.sum(real_counts_2)
fake_counts,_   = np.histogram(fake, bins = bins, density=True)
fake_counts /= np.sum(fake_counts)
stat_counts,_  = np.histogram(stat, bins = bins, density=True)
stat_counts /= np.sum(stat_counts)
vae_counts,_  = np.histogram(f_vae, bins = bins, density=True)
vae_counts /= np.sum(vae_counts)


js_divergence_vel_fake  = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(fake_counts) + epsilon)

js_divergence_vel_real = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(real_counts_2) + epsilon)

js_divergence_vel_syneye = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(eyesyn_counts) + epsilon)

js_divergence_vel_stat = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(stat_counts) + epsilon)

js_divergence_vel_vae = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(vae_counts) + epsilon)

print('Saccade mean velocity one fold')
print('JS(real || stat): ' + str(np.round(js_divergence_vel_stat,decimals = 3)))
print('JS(real || VAE): ' + str(np.round(js_divergence_vel_vae,decimals = 3)))
print('JS(real || GAN): ' + str(np.round(js_divergence_vel_fake,decimals = 3)))
print('JS(real || Real): ' + str(np.round(js_divergence_vel_real,decimals = 3)))

print('Saccade mean velocity ' + str(iterations) + ' folds')
stat_vals = js_divergence_sampling( real, stat, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
vae_vals = js_divergence_sampling( real, f_vae, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
gan_vals = js_divergence_sampling( real, fake, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
real_vals = js_divergence_sampling( real, real_2, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
                                        
model_vals = [stat_vals, vae_vals, gan_vals]
model_means = [np.mean(a) for a in model_vals]
prefixes = ['' for a in range(len(model_means))]
suffixes = ['' for a in range(len(model_means))]
arg_sort = np.argsort(model_means)
prefixes[arg_sort[0]] = '\\textbf{'
suffixes[arg_sort[0]] = '}'

tt_test_pvalue = ttest_ind(model_vals[arg_sort[0]],model_vals[arg_sort[1]],alternative='two-sided').pvalue
if tt_test_pvalue <= 0.05:
    suffixes[arg_sort[0]] += '$^*$'
                                       
print('JS(real || stat): ' + prefixes[0] + str(np.round(np.mean(stat_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(stat_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[0])
print('JS(real || VAE): ' + prefixes[1] +  str(np.round(np.mean(vae_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(vae_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[1])
print('JS(real || GAN): ' + prefixes[2] +  str(np.round(np.mean(gan_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(gan_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[2])
print('JS(real || Real): ' + str(np.round(np.mean(real_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(real_vals)/np.sqrt(iterations),decimals = 3)))





# peak acceleration
real = np.array([np.max(a) for a in real_accs_sac])
real_2 = np.array([np.mean(a) for a in real_accs_sac])
fake = np.array([np.max(a) for a in fake_accs_sac])
gauss = np.array([np.max(a) for a in gauss_accs_sac])
stat   = np.array([np.max(a) for a in sac_acc_statistical])
f_vae = np.array([np.max(a) for a in vae_accs_sac])

min_val = np.min(list(real) + list(fake) + list(gauss) + list(real_2) + list(eyesyn) + list(stat) + list(f_vae))
max_val = np.max(list(real) + list(fake) + list(gauss) + list(real_2) + list(eyesyn) + list(stat) + list(f_vae))
num_bins = 100
epsilon = 0.00001
bins = np.linspace(min_val,max_val,num_bins)

real_counts,_   = np.histogram(real, bins = bins, density=True)
real_counts /= np.sum(real_counts)
real_counts_2,_   = np.histogram(real_2, bins = bins, density=True)
real_counts_2 /= np.sum(real_counts_2)
fake_counts,_   = np.histogram(fake, bins = bins, density=True)
fake_counts /= np.sum(fake_counts)
gauss_counts,_  = np.histogram(gauss, bins = bins, density=True)
gauss_counts /= np.sum(gauss_counts)
eyesyn_counts,_  = np.histogram(eyesyn, bins = bins, density=True)
eyesyn_counts /= np.sum(eyesyn_counts)
stat_counts,_  = np.histogram(stat, bins = bins, density=True)
stat_counts /= np.sum(stat_counts)
vae_counts,_  = np.histogram(f_vae, bins = bins, density=True)
vae_counts /= np.sum(vae_counts)


js_divergence_vel_fake  = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(fake_counts) + epsilon)

js_divergence_vel_gauss = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(gauss_counts) + epsilon)

js_divergence_vel_real = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(real_counts_2) + epsilon)

js_divergence_vel_syneye = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(eyesyn_counts) + epsilon)

js_divergence_vel_stat = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(stat_counts) + epsilon)

js_divergence_vel_vae = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(vae_counts) + epsilon)


print('Saccade peak acceleration one fold')
print('JS(real || stat): ' + str(np.round(js_divergence_vel_stat,decimals = 3)))
print('JS(real || VAE): ' + str(np.round(js_divergence_vel_vae,decimals = 3)))
print('JS(real || GAN): ' + str(np.round(js_divergence_vel_fake,decimals = 3)))
print('JS(real || Real): ' + str(np.round(js_divergence_vel_real,decimals = 3)))

print('Saccade peak acceleration ' + str(iterations) + ' folds')

stat_vals = js_divergence_sampling( real, stat, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
vae_vals = js_divergence_sampling( real, f_vae, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
gan_vals = js_divergence_sampling( real, fake, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
real_vals = js_divergence_sampling( real, real_2, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
                                        
model_vals = [stat_vals, vae_vals, gan_vals]
model_means = [np.mean(a) for a in model_vals]
prefixes = ['' for a in range(len(model_means))]
suffixes = ['' for a in range(len(model_means))]
arg_sort = np.argsort(model_means)
prefixes[arg_sort[0]] = '\\textbf{'
suffixes[arg_sort[0]] = '}'

tt_test_pvalue = ttest_ind(model_vals[arg_sort[0]],model_vals[arg_sort[1]],alternative='two-sided').pvalue
if tt_test_pvalue <= 0.05:
    suffixes[arg_sort[0]] += '$^*$'
                                       
print('JS(real || stat): ' + prefixes[0] + str(np.round(np.mean(stat_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(stat_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[0])
print('JS(real || VAE): ' + prefixes[1] +  str(np.round(np.mean(vae_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(vae_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[1])
print('JS(real || GAN): ' + prefixes[2] +  str(np.round(np.mean(gan_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(gan_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[2])
print('JS(real || Real): ' + str(np.round(np.mean(real_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(real_vals)/np.sqrt(iterations),decimals = 3)))



# mean acceleration
real = np.array([np.mean(a) for a in real_accs_sac])
real_2 = np.array([np.mean(a) for a in real_accs_sac])
fake = np.array([np.mean(a) for a in fake_accs_sac])
gauss = np.array([np.mean(a) for a in gauss_accs_sac])
stat   = np.array([np.mean(a) for a in sac_acc_statistical])
f_vae = np.array([np.mean(a) for a in vae_accs_sac])

min_val = np.min(list(real) + list(fake) + list(gauss) + list(real_2) + list(eyesyn) + list(stat) + list(f_vae))
max_val = np.max(list(real) + list(fake) + list(gauss) + list(real_2) + list(eyesyn) + list(stat) + list(f_vae))
num_bins = 100
epsilon = 0.00001
bins = np.linspace(min_val,max_val,num_bins)

real_counts,_   = np.histogram(real, bins = bins, density=True)
real_counts /= np.sum(real_counts)
real_counts_2,_   = np.histogram(real_2, bins = bins, density=True)
real_counts_2 /= np.sum(real_counts_2)
fake_counts,_   = np.histogram(fake, bins = bins, density=True)
fake_counts /= np.sum(fake_counts)
gauss_counts,_  = np.histogram(gauss, bins = bins, density=True)
gauss_counts /= np.sum(gauss_counts)
eyesyn_counts,_  = np.histogram(eyesyn, bins = bins, density=True)
eyesyn_counts /= np.sum(eyesyn_counts)
stat_counts,_  = np.histogram(stat, bins = bins, density=True)
stat_counts /= np.sum(stat_counts)
vae_counts,_  = np.histogram(f_vae, bins = bins, density=True)
vae_counts /= np.sum(vae_counts)


js_divergence_vel_fake  = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(fake_counts) + epsilon)

js_divergence_vel_gauss = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(gauss_counts) + epsilon)

js_divergence_vel_real = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(real_counts_2) + epsilon)

js_divergence_vel_syneye = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(eyesyn_counts) + epsilon)

js_divergence_vel_stat = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(stat_counts) + epsilon)

js_divergence_vel_vae = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(vae_counts) + epsilon)


print('Saccade mean acceleration one fold')
print('JS(real || stat): ' + str(np.round(js_divergence_vel_stat,decimals = 3)))
print('JS(real || VAE): ' + str(np.round(js_divergence_vel_vae,decimals = 3)))
print('JS(real || GAN): ' + str(np.round(js_divergence_vel_fake,decimals = 3)))
print('JS(real || Real): ' + str(np.round(js_divergence_vel_real,decimals = 3)))

print('Saccade mean acceleration ' + str(iterations) + ' folds')

stat_vals = js_divergence_sampling( real, stat, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
vae_vals = js_divergence_sampling( real, f_vae, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
gan_vals = js_divergence_sampling( real, fake, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
real_vals = js_divergence_sampling( real, real_2, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)

model_vals = [stat_vals, vae_vals, gan_vals]
model_means = [np.mean(a) for a in model_vals]
prefixes = ['' for a in range(len(model_means))]
suffixes = ['' for a in range(len(model_means))]
arg_sort = np.argsort(model_means)
prefixes[arg_sort[0]] = '\\textbf{'
suffixes[arg_sort[0]] = '}'

tt_test_pvalue = ttest_ind(model_vals[arg_sort[0]],model_vals[arg_sort[1]],alternative='two-sided').pvalue
if tt_test_pvalue <= 0.05:
    suffixes[arg_sort[0]] += '$^*$'
                                       
print('JS(real || stat): ' + prefixes[0] + str(np.round(np.mean(stat_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(stat_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[0])
print('JS(real || VAE): ' + prefixes[1] +  str(np.round(np.mean(vae_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(vae_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[1])
print('JS(real || GAN): ' + prefixes[2] +  str(np.round(np.mean(gan_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(gan_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[2])
print('JS(real || Real): ' + str(np.round(np.mean(real_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(real_vals)/np.sqrt(iterations),decimals = 3)))


# amplitude
real = real_amplitudes_sac
real_2 = np.array([np.mean(a) for a in real_amplitudes_sac_2])
fake = fake_amplitudes_sac
gauss = gauss_amplitudes_sac
f_vae = vae_amplitudes_sac

min_val = np.min(list(real) + list(fake) + list(gauss) + list(real_2) + list(f_vae))
max_val = np.max(list(real) + list(fake) + list(gauss) + list(real_2) + list(f_vae))
num_bins = 100
epsilon = 0.00001
bins = np.linspace(min_val,max_val,num_bins)

real_counts,_   = np.histogram(real, bins = bins, density=True)
real_counts /= np.sum(real_counts)
real_counts_2,_   = np.histogram(real_2, bins = bins, density=True)
real_counts_2 /= np.sum(real_counts_2)
fake_counts,_   = np.histogram(fake, bins = bins, density=True)
fake_counts /= np.sum(fake_counts)
gauss_counts,_  = np.histogram(gauss, bins = bins, density=True)
gauss_counts /= np.sum(gauss_counts)
eyesyn_counts,_  = np.histogram(eyesyn, bins = bins, density=True)
eyesyn_counts /= np.sum(eyesyn_counts)
vae_counts,_  = np.histogram(f_vae, bins = bins, density=True)
vae_counts /= np.sum(vae_counts)


js_divergence_vel_fake  = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(fake_counts) + epsilon)

js_divergence_vel_gauss = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(gauss_counts) + epsilon)

js_divergence_vel_real = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(real_counts_2) + epsilon)

js_divergence_vel_syneye = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(eyesyn_counts) + epsilon)

js_divergence_vel_vae = js_divergence(np.array(real_counts)  + epsilon,
                                      np.array(vae_counts) + epsilon)
                                      
                                      
print('Saccade amplitude one fold')
print('JS(real || stat): -')
print('JS(real || VAE): ' + str(np.round(js_divergence_vel_vae,decimals = 3)))
print('JS(real || GAN): ' + str(np.round(js_divergence_vel_fake,decimals = 3)))
print('JS(real || Real): ' + str(np.round(js_divergence_vel_real,decimals = 3)))

print('Saccade amplitude ' + str(iterations) + ' folds')

vae_vals = js_divergence_sampling( real, f_vae, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
gan_vals = js_divergence_sampling( real, fake, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)
real_vals = js_divergence_sampling( real, real_2, bins, epsilon =  epsilon,
                                        iterations = iterations, number_per_iter = number_per_iter)

model_vals = [vae_vals, gan_vals]
model_means = [np.mean(a) for a in model_vals]
prefixes = ['' for a in range(len(model_means))]
suffixes = ['' for a in range(len(model_means))]
arg_sort = np.argsort(model_means)
prefixes[arg_sort[0]] = '\\textbf{'
suffixes[arg_sort[0]] = '}'

tt_test_pvalue = ttest_ind(model_vals[arg_sort[0]],model_vals[arg_sort[1]],alternative='two-sided').pvalue
if tt_test_pvalue <= 0.05:
    suffixes[arg_sort[0]] += '$^*$'
                                       
print('JS(real || stat): -')
print('JS(real || VAE): ' + prefixes[0] +  str(np.round(np.mean(vae_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(vae_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[0])
print('JS(real || GAN): ' + prefixes[1] +  str(np.round(np.mean(gan_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(gan_vals)/np.sqrt(iterations),decimals = 3)) + suffixes[1])
print('JS(real || Real): ' + str(np.round(np.mean(real_vals),decimals = 3)) + ' $\\pm$ ' + str(np.round(np.std(real_vals)/np.sqrt(iterations),decimals = 3)))