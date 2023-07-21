import os
import numpy as np
import random
from tqdm import tqdm
import sys
import joblib
import argparse
import pandas as pd

import tensorflow
import tensorflow as tf
from Model import eventGAN as eventGAN



def main():
    # global params
    data_dir = 'data/'
    verbose = 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-GPU','--GPU',type=int,default=0)
    parser.add_argument('-model_dir','--model_dir',type=str,default='event_model/')
    parser.add_argument('-event_type','--event_type',type=str,default='fixation')
    parser.add_argument('-stimulus','--stimulus',type=str,default='text')
    parser.add_argument('-hp_result_path','--hp_result_path',type=str,default=None)
    
    args = parser.parse_args()
    
    GPU            = args.GPU
    model_dir      = args.model_dir
    event_type     = args.event_type
    stimulus       = args.stimulus
    hp_result_path = args.hp_result_path
    
    if GPU != -1:
        flag_train_on_gpu = True
    else:
        flag_train_on_gpu = False
    
    if event_type == 'fixation':
        data = np.load(data_dir + 'fixation_matrix_gazebase_vd_' + stimulus + '.npy')        
        save_path = model_dir + 'fixation_model_' + stimulus
    elif event_type == 'saccade':
        data = np.load(data_dir + 'saccade_matrix_gazebase_vd_' + stimulus + '.npy')
        save_path = model_dir + 'saccade_model_' + stimulus
    
    if hp_result_path is not None:
        save_path += '_optimized'
    
    window_size = data.shape[1]
    
    # params for NN
    random_size = 32
    channels = 2
    relu_in_last = False
    batch_size = 256
    
    if hp_result_path is None:
        gen_kernel_sizes = [window_size,8,4,2]
        gen_filter_sizes = [16,8,4,2]

        dis_kernel_sizes = [8,16,32]
        dis_fiter_sizes = [32,64,128]
        dis_dropout = 0.3
    else:
        # select best found hyperparameters
        hp_result_data = pd.read_csv(hp_result_path)
        event_accs = list(hp_result_data['event_acc'])
        model_names = list(hp_result_data['model_name'])
        best_id = np.argmax(event_accs)
        best_model_name = model_names[best_id]
        
        gen_kernel_sizes = [int(a) for a in np.array(best_model_name.split('_')[0].replace('[','').replace(']','').split(','))]
        gen_filter_sizes = [int(a) for a in np.array(best_model_name.split('_')[1].replace('[','').replace(']','').split(','))]

        dis_kernel_sizes = [int(a) for a in np.array(best_model_name.split('_')[2].replace('[','').replace(']','').split(','))]
        dis_fiter_sizes = [int(a) for a in np.array(best_model_name.split('_')[3].replace('[','').replace(']','').split(','))]

        dis_dropout = float(best_model_name.split('_')[4])
        

    model_config = {'gen_kernel_sizes':gen_kernel_sizes,
                    'gen_filter_sizes':gen_filter_sizes,
                    'dis_kernel_sizes':dis_kernel_sizes,
                    'dis_fiter_sizes':dis_fiter_sizes,
                    'dis_dropout':dis_dropout,
                    'window_size':window_size,
                    'channels':channels,
                    'batch_size':batch_size,
                    'random_size':random_size,
                    'relu_in_last':relu_in_last,
                   }


    EPOCHS = 100
    noise_dim = random_size
    num_examples_to_generate = 16
    BUFFER_SIZE = 60000
    BATCH_SIZE = model_config['batch_size']

    
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
        
        
        
    # load data
    column_dict = joblib.load(data_dir + 'column_dict.joblib')
    
        
    data[data == -1] = 0.
    
    # create train data
    train_dataset = tf.data.Dataset.from_tensor_slices(data[:,:,4:6]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    # intialize model
    model = eventGAN.eventGAN(model_config)
    
    # train model
    gen_loss_list, disc_loss_list = model.train(train_dataset, EPOCHS, verbose = verbose)
        
    # save model
    model.save_model(save_path)
    
    
if __name__ == "__main__":
    # execute only if run as a script
    main() 