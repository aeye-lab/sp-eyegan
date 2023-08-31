import os
import numpy as np
import random
from tqdm import tqdm
import sys
import joblib
import seaborn as sns
import pandas as pd
import argparse
import tensorflow
import tensorflow as tf
from tensorflow import keras
import config as config

from Preprocessing import data_loader as data_loader
from Model import contrastive_learner as contrastive_learner



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

def main():
    # global
    parser = argparse.ArgumentParser()
    parser.add_argument('-GPU','--GPU',type=int,default=3)
    parser.add_argument('-temperature','--temperature',type=float,default=0.1)
    parser.add_argument('-sd','--sd',type=float,default=0.1)
    parser.add_argument('-sd_factor','--sd_factor',type=float,default=1.25)
    parser.add_argument('-window_size','--window_size',type=int,default=5000)
    parser.add_argument('-orig_sampling_rate','--orig_sampling_rate',type=int,default=1000)
    parser.add_argument('-target_sampling_rate','--target_sampling_rate',type=int,default=1000)
    parser.add_argument('-overall_size','--overall_size',type=int,default=5000)
    parser.add_argument('-channels','--channels',type=int,default=2)
    parser.add_argument('-batch_size','--batch_size',type=int,default=32)
    parser.add_argument('-num_epochs','--num_epochs',type=int,default=1000)
    parser.add_argument('-model_dir','--model_dir',type=str,default='pretrain_model/')
    parser.add_argument('-data_dir','--data_dir',type=str,default='data/')
    parser.add_argument('-augmentation_mode','--augmentation_mode',type=str,default='random')
    parser.add_argument('-check_point_saver','--check_point_saver',type=int,default=100)
    parser.add_argument('-max_rotation','--max_rotation',type=float,default=6.)
    parser.add_argument('-stimulus','--stimulus',type=str,default='video') # video text original
    parser.add_argument('-encoder_name','--encoder_name',type=str,default='ekyt')
    parser.add_argument('-scanpath_model','--scanpath_model',type=str,default='random') # random|stat_model
    parser.add_argument('-num_pretrain_instances','--num_pretrain_instances',type=int,default=-1)
    parser.add_argument('-flag_redo','--flag_redo',type=int,default=0)
    parser.add_argument('-data_suffix','--data_suffix',type=str,default='baseline_1000')
    parser.add_argument('-data_path','--data_path',type=str,default=None)
    
    
    
    args = parser.parse_args()
    GPU = args.GPU
    temperature = args.temperature
    window_size = args.window_size
    overall_size = args.overall_size
    channels = args.channels
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    model_dir = args.model_dir
    data_dir = args.data_dir
    augmentation_mode = args.augmentation_mode
    check_point_saver = args.check_point_saver
    sd = args.sd
    sd_factor = args.sd_factor
    stimulus = args.stimulus
    max_rotation = args.max_rotation
    encoder_name = args.encoder_name
    scanpath_model = args.scanpath_model
    num_pretrain_instances = args.num_pretrain_instances
    flag_redo = args.flag_redo
    if flag_redo == 1:
        flag_redo = True
    else:
        flag_redo = False
    data_suffix = args.data_suffix
    orig_sampling_rate = args.orig_sampling_rate
    target_sampling_rate = args.target_sampling_rate
    
    model_window_size = int(window_size / (orig_sampling_rate / target_sampling_rate))
    skip_rate = int(orig_sampling_rate / target_sampling_rate)
    
    
    if encoder_name == 'clrgaze':
        embedding_size = 512
    elif encoder_name == 'ekyt':
        embedding_size = 128
    
    if augmentation_mode == 'crop':
        contrastive_augmentation = {'window_size': model_window_size, 'overall_size': overall_size,'channels':channels, 'name':'crop'}
        model_save_path = model_dir + encoder_name + '_' + augmentation_mode + '_window_size_' + str(model_window_size) +\
                            '_overall_size_' + str(overall_size) +\
                            '_embedding_size_' + str(embedding_size) + '_stimulus_' + str(stimulus) +\
                            '_model_' + str(scanpath_model) + '_' + str(num_pretrain_instances)
        per_process_gpu_memory_fraction = 1.
    elif augmentation_mode == 'random':
        contrastive_augmentation = {'window_size': model_window_size, 'channels':channels, 'name':'random','sd':sd}
        model_save_path = model_dir + encoder_name + '_' + augmentation_mode + '_window_size_' + str(model_window_size) +\
                            '_sd_' + str(sd) + '_sd_factor_' + str(sd_factor) +\
                            '_embedding_size_' + str(embedding_size) + '_stimulus_' + str(stimulus) +\
                            '_model_' + str(scanpath_model) + '_' + str(num_pretrain_instances)
        per_process_gpu_memory_fraction = 1.
    elif augmentation_mode == 'rotation':
        contrastive_augmentation = {'window_size': model_window_size, 'channels':channels, 'name':'rotation','max_rotation':max_rotation}
        model_save_path = model_dir + encoder_name + '_' + augmentation_mode + '_window_size_' + str(model_window_size) +\
                            '_max_rotation_' + str(max_rotation) +\
                            '_embedding_size_' + str(embedding_size) + '_stimulus_' + str(stimulus) +\
                            '_model_' + str(scanpath_model) + '_' + str(num_pretrain_instances)
        per_process_gpu_memory_fraction = 1.
    elif augmentation_mode == 'mixed':
        contrastive_augmentation = contrastive_augmentation = {'window_size': model_window_size, 'channels':channels, 'name':'random','sd':sd}
        model_save_path = model_dir + encoder_name + '_' + augmentation_mode + '_window_size_' + str(model_window_size) +\
                            '_max_rotation_' + str(max_rotation) +\
                            '_sd_' + str(sd) + '_sd_factor_' + str(sd_factor) +\
                            '_embedding_size_' + str(embedding_size) + '_stimulus_' + str(stimulus) +\
                            '_model_' + str(scanpath_model) + '_' + str(num_pretrain_instances)
        per_process_gpu_memory_fraction = 1.
    
    model_save_path += data_suffix
    
    if not flag_redo and (os.path.exists(model_save_path) or os.path.exists(model_save_path + '.index')):
        print('already exists')
        return 0
    
    print('pretrain config: ' + str(contrastive_augmentation))
        
    flag_train_on_gpu = True
    if flag_train_on_gpu:
        import tensorflow as tf
        # select graphic card
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        config = tf.compat.v1.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        config.gpu_options.allow_growth = True
        tf_session = tf.compat.v1.Session(config=config)
    else:
        import tensorflow as tf
        # select graphic card
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        
    # load data
    if stimulus != 'original':
        if args.data_path is not None:
            syn_data_path = args.data_path
        else:
            if window_size != 5000:
                syn_data_path = data_dir + 'synthetic_data_' + str(stimulus) + '_' + str(scanpath_model) + '_' + str(output_size) + data_suffix + '.npy'
            else:
                syn_data_path = data_dir + 'synthetic_data_' + str(stimulus) + '_' + str(scanpath_model) + data_suffix + '.npy'            
        
        print('load ' + str(syn_data_path))
        syn_data = np.load(syn_data_path)
        if num_pretrain_instances != -1:
            random_ids = np.random.permutation(np.arange(syn_data.shape[0]))
            syn_data = syn_data[random_ids[0:num_pretrain_instances]]
        if skip_rate != 1:
            skips = np.arange(0,window_size,skip_rate)[0:model_window_size]
            syn_data = syn_data[:,skips,:]
    else:
        output_length = window_size
        max_round = 9
        use_trial_types = ['TEX']
        X_dict,Y,Y_columns = data_loader.load_gazebase_data(gaze_base_dir = config.gaze_base_dir,
                                use_trial_types = use_trial_types,
                                number_train = -1,
                                max_round = max_round,
                                output_length = output_length,
                                only_all_rounds = False,
                                )
        X_vel = X_dict.copy()['X_vel'] / 1000. # bring to range 0-1
        X_px = X_dict.copy()['X_px']
        syn_data = X_vel
        
        if skip_rate != 1:
            skips = np.arange(0,window_size,skip_rate)[0:model_window_size]
            syn_data = syn_data[:,skips,:]
    
    if augmentation_mode == 'mixed':
        train_dataset_random =  contrastive_learner.prepare_prtrain_dataset_from_array(unlabeled_train_data = syn_data, 
                                                                                               batch_size = batch_size)
                                                                                               
        syn_data_dva = np.zeros(syn_data.shape)
        for i in tqdm(np.arange(syn_data_dva.shape[0])):
            syn_data_dva[i] = vel_to_dva(np.array(syn_data[i]))
        train_dataset_rotation =  contrastive_learner.prepare_prtrain_dataset_from_array(unlabeled_train_data = syn_data_dva, 
                                                                                       batch_size = batch_size)                                                                                    
    else:
    
    
        # create train data and train model
        if augmentation_mode != 'rotation':
            train_dataset =  contrastive_learner.prepare_prtrain_dataset_from_array(unlabeled_train_data = syn_data, 
                                                                                               batch_size = batch_size)
        else:
            syn_data_dva = np.zeros(syn_data.shape)
            for i in tqdm(np.arange(syn_data_dva.shape[0])):
                syn_data_dva[i] = vel_to_dva(np.array(syn_data[i]))
            train_dataset =  contrastive_learner.prepare_prtrain_dataset_from_array(unlabeled_train_data = syn_data_dva, 
                                                                                           batch_size = batch_size)
    
    # model training
    # Contrastive pretraining
    pretraining_model = contrastive_learner.ContrastiveModel(temperature=temperature,
                                                            embedding_size = embedding_size,
                                                            contrastive_augmentation = contrastive_augmentation,
                                                            channels = channels,
                                                            window_size = model_window_size,
                                                            encoder_name = encoder_name)
    pretraining_model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
    )
    
    if augmentation_mode == 'mixed':
        for epoch_num in range(num_epochs):
            modulo = epoch_num % 2
            if modulo == 0:
                cur_mode = 'random'
            else:
                cur_mode = 'rotation'
            if epoch_num > 0 and epoch_num % check_point_saver == 0:
                flag_save = True
            else:
                flag_save = False
            if cur_mode == 'random':
                contrastive_augmentation = {'window_size': model_window_size, 'channels':channels, 'name':'random','sd':sd}
                pretraining_model.set_augmenter(contrastive_augmentation)
                
                pretraining_history = pretraining_model.fit(
                    train_dataset_random, epochs=1,
                )
            elif cur_mode == 'rotation':
                contrastive_augmentation = {'window_size': model_window_size, 'channels':channels, 'name':'rotation','max_rotation':max_rotation}
                pretraining_model.set_augmenter(contrastive_augmentation)
                pretraining_history = pretraining_model.fit(
                    train_dataset_rotation, epochs=1,
                )
            
            if flag_save:
                pretraining_model.save_encoder_weights(model_save_path + '_checkpoint_' + str(epoch_num))
    else:
        # check if we want to save checkpoint every check_point_saver epochs
        if check_point_saver != -1:
            epochs_per_checkpoint = check_point_saver
            iterations = int(np.ceil(num_epochs / epochs_per_checkpoint))
            used_iterations = 0
            for train_iter in range(iterations):
                if augmentation_mode == 'random':
                    contrastive_augmentation = {'window_size': model_window_size, 'channels':channels, 'name':'random','sd':sd}
                    sd = sd * sd_factor
                    pretraining_model.set_augmenter(contrastive_augmentation)
                cur_epochs = epochs_per_checkpoint
                if cur_epochs + used_iterations > num_epochs:
                    cur_epochs = num_epochs - used_iterations
                pretraining_history = pretraining_model.fit(
                    train_dataset, epochs=cur_epochs,
                )
                used_iterations += cur_epochs
                pretraining_model.save_encoder_weights(model_save_path + '_checkpoint_' + str(used_iterations))
        else:
            pretraining_history = pretraining_model.fit(
                train_dataset, epochs=num_epochs,
            )

    pretraining_model.save_encoder_weights(model_save_path)
    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()