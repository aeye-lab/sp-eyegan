import sys
import os
import joblib
import numpy as np
import random
import sys
import seaborn as sns
import pandas as pd
from tqdm.notebook import tqdm
from sklearn import metrics
import socket
import argparse


import tensorflow
import tensorflow as tf
from Model import eventGAN as eventGAN
from Model import stat_scanpath_model as stat_scanpath_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-GPU', '--GPU', type=int, default=0)
    parser.add_argument('-num_samples', '--num_samples', type=int, default=100000)
    parser.add_argument('-output_size', '--output_size', type=int, default=5000)
    parser.add_argument('-flag_train_on_gpu', '--flag_train_on_gpu', type=int, default=1)
    parser.add_argument('-data_dir', '--data_dir', type=str, default='data/')
    parser.add_argument('-stimulus', '--stimulus', type=str, default='text')
    parser.add_argument('-sac_window_size', '--sac_window_size', type=int, default=30)
    parser.add_argument('-fix_window_size', '--fix_window_size', type=int, default=100)
    parser.add_argument('-scanpath_model', '--scanpath_model', type=str, default='random') # random|stat_model

    args = parser.parse_args()
    GPU = args.GPU
    data_dir = args.data_dir
    num_samples = args.num_samples
    output_size = args.output_size
    stimulus = args.stimulus
    sac_window_size = args.sac_window_size
    fix_window_size = args.fix_window_size
    scanpath_model = args.scanpath_model
    
    # params for stimulus
    expt_txt = {  'px_x':1680,
                  'px_y':1050,
                  'max_dva_x': 30,
                  'max_dva_y': 25
                 }
    
    
    # params for NN
    random_size = 32
    window_size = 100
    gen_kernel_sizes_fixation = [fix_window_size,8,4,2]
    gen_kernel_sizes_saccade = [sac_window_size,8,4,2]
    gen_filter_sizes = [16,8,4,2]
    channels = 2
    relu_in_last = False
    batch_size = 256

    dis_kernel_sizes = [8,16,32]
    dis_fiter_sizes = [32,64,128]
    dis_dropout = 0.3

    sample_size = 1000

    
    
    # params for generator
    window_size = 100
    random_size = 32
    channels = 2
    mean_sacc_len = 20
    std_sacc_len  = 10

    mean_fix_len  = 250
    std_fix_len   = 225

    
    
    
    
    fixation_path  = 'event_model/fixation_model_' + stimulus
    saccade_path   = 'event_model/saccade_model_' + stimulus 
    
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
    
    gan_config = {'window_size':window_size,
                  'random_size':random_size,
                  'channels':channels,
                  'mean_sacc_len':mean_sacc_len,
                  'std_sacc_len':std_sacc_len,
                  'mean_fix_len':mean_fix_len,
                  'std_fix_len':std_fix_len,
                  'fixation_path':fixation_path,
                  'saccade_path':saccade_path,
                 }
    
    flag_train_on_gpu = args.flag_train_on_gpu
    if flag_train_on_gpu == 1:
        flag_train_on_gpu = True
    else:
        flag_train_on_gpu = False
        
    # set up GPU
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
    
    
    data_generator = eventGAN.dataGenerator(gan_config,
                                               model_config_fixation,
                                               model_config_saccade,
                                               )
    
    if scanpath_model == 'random':
        syt_data = data_generator.sample_random_data(num_samples = num_samples,
                                                 output_size = output_size)
    elif scanpath_model == 'stat_model':
        stat_model = stat_scanpath_model.satisticalScanPath()
        # collect texts
        data_dir = '/home/prasse/work/Projekte/AEye/aeye_synthetic_data/ocr_detection/images/'
        text_data_csvs = []
        file_list = os.listdir(data_dir)
        text_lists = []
        for ii in range(len(file_list)):
            if file_list[ii].endswith('.csv'):
                ocr_data = pd.read_csv(data_dir + file_list[ii])
                confs = np.array(ocr_data['conf'])
                lefts = np.array(ocr_data['left'])
                tops = np.array(ocr_data['top'])
                widths = np.array(ocr_data['width'])
                heights = np.array(ocr_data['height'])
                words = np.array(ocr_data['text'])

                text_list = []

                for i in range(len(confs)):
                    cur_conf = confs[i]
                    if cur_conf != -1:        
                        text_list.append((words[i],
                                          lefts[i] + (widths[i]/2),
                                        tops[i] + (heights[i]/2)))
                text_lists.append(text_list)
        expt_txts = [expt_txt for _ in range(len(text_lists))]
        data_dict = data_generator.sample_scanpath_dataset_stat_model(
                                           stat_model,
                                           text_lists,
                                           expt_txts,
                                           num_sample_saccs = 250,
                                           dva_threshold = 0.2,
                                           max_iter = 10,
                                           num_scanpaths_per_text = 10,
                                           num_samples = num_samples,
                                           output_size = output_size,
                                           store_dva_data = False,
                                          )
        syt_data = data_dict['vel_data']
        
        
    np.save(data_dir + 'synthetic_data_' + str(stimulus) + '_' + str(scanpath_model),syt_data)
    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()