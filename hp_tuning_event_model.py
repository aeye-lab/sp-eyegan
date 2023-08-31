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
import pymovements as pm



def main():
    # global params
    data_dir = 'data/'
    verbose = 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-GPU','--GPU',type=int,default=0)
    parser.add_argument('-model_dir','--model_dir',type=str,default='event_model/')
    parser.add_argument('-event_type','--event_type',type=str,default='fixation')
    parser.add_argument('-stimulus','--stimulus',type=str,default='text')
    parser.add_argument('-result_dir','--result_dir',type=str,default='results/')
    
    args = parser.parse_args()
    
    GPU = args.GPU
    model_dir = args.model_dir
    event_type = args.event_type
    stimulus = args.stimulus
    result_dir = args.result_dir
    
    if GPU != -1:
        flag_train_on_gpu = True
    else:
        flag_train_on_gpu = False
    
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    
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
    
    if event_type == 'fixation':
        data = np.load(data_dir + 'fixation_matrix_gazebase_vd_' + stimulus + '.npy')
        save_path = model_dir + 'fixation_model_' + stimulus
    elif event_type == 'saccade':
        data = np.load(data_dir + 'saccade_matrix_gazebase_vd_' + stimulus + '.npy')
        save_path = model_dir + 'saccade_model_' + stimulus
    
    
    hp_path = result_dir + event_type + '_hp.csv'
    if os.path.exists(hp_path):
        hp_data = pd.read_csv(hp_path)
    else:
        hp_data = {'model_name':[],
                   'event_acc':[],
                  }
    
    model_names = set(list(hp_data['model_name']))
    
    # load data
    column_dict = joblib.load(data_dir + 'column_dict.joblib')
    
        
    data[data == -1] = 0.
    
    # create train data
    train_dataset = tf.data.Dataset.from_tensor_slices(data[:,:,4:6]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    
    window_size = data.shape[1]
    
    while True:
        while True:
            # create config
            # params for NN
            random_size = 32
            channels = 2


            gen_depth_list = [1,2,3,4,5]
            num_gen_depth = int(np.random.choice(gen_depth_list))
            gen_kernel_sizes_list = [2,4,8,16,32,64]
            gen_filter_sizes_list = [2,4,8,16,32,64]

            gen_kernel_sizes = []
            gen_filter_sizes = []
            relu_in_last = False
            for i in range(num_gen_depth):
                gen_kernel_sizes.append(int(np.random.choice(gen_kernel_sizes_list,1)))
                gen_filter_sizes.append(int(np.random.choice(gen_filter_sizes_list,1)))
                if i == 0:
                    gen_kernel_sizes[0] = window_size
                if i == num_gen_depth-1:
                    gen_filter_sizes[-1] = channels


            dis_depth_list = [1,2,3,4,5]
            num_dis_depth = int(np.random.choice(dis_depth_list))
            dis_kernel_sizes_list = [2,4,8,16,32,64]
            dis_filter_sizes_list = [2,4,8,16,32,64]
            dis_dropout_list = [0.1,0.2,0.3,0.4,0.5]
            dis_dropout = float(np.random.choice(dis_dropout_list))

            dis_kernel_sizes = []
            dis_fiter_sizes = []

            for i in range(num_dis_depth):
                dis_kernel_sizes.append(int(np.random.choice(dis_kernel_sizes_list,1)))
                dis_fiter_sizes.append(int(np.random.choice(dis_filter_sizes_list,1)))


            model_config = {'gen_kernel_sizes':gen_kernel_sizes,
                            'gen_filter_sizes':gen_filter_sizes,
                            'dis_kernel_sizes':dis_kernel_sizes,
                            'dis_fiter_sizes':dis_fiter_sizes,
                            'dis_dropout':dis_dropout,
                            'window_size':window_size,
                            'channels':channels,
                            'batch_size':BATCH_SIZE,
                            'random_size':random_size,
                            'relu_in_last':relu_in_last,
                           }

            cur_model_name = str(gen_kernel_sizes) + '_' +\
                            str(gen_filter_sizes) + '_' +\
                            str(dis_kernel_sizes) + '_' +\
                            str(dis_fiter_sizes) + '_' +\
                            str(dis_dropout)

            if cur_model_name not in model_names:
                save_path += '_' + cur_model_name
                break
        
        model = eventGAN.eventGAN(model_config)
    
    


        EPOCHS = 100
        noise_dim = random_size
        num_examples_to_generate = 16
    
        # train model
        gen_loss_list, disc_loss_list = model.train(train_dataset, EPOCHS, verbose = verbose)
            
        # save model
        model.save_model(save_path)
        
        if os.path.exists(hp_path):
            hp_data_pd = pd.read_csv(hp_path)
            hp_data = {'model_name':list(hp_data_pd['model_name']),
                       'event_acc':list(hp_data_pd['event_acc']),
                      }
        else:
            hp_data = {'model_name':[],
                       'event_acc':[],
                      }
        
        # create synthetic data
        sample_instances = 1000
        num_iter = int(np.ceil(sample_instances / model_config['batch_size']))
        out_data = np.zeros([num_iter * model_config['batch_size'],model_config['window_size'],2])
        for i in tqdm(np.arange(num_iter)):
            noise = tf.random.normal([model_config['batch_size'], model_config['random_size']])
            syn_data = model.generator(noise)
            out_data[i*model_config['batch_size']:(i+1)*model_config['batch_size'],:,:] = syn_data
        
        # load data for other event
        if event_type == 'fixation':
            other_event_data = np.load(data_dir + 'saccade_matrix_gazebase_vd_' + stimulus + '.npy')
            other_event_data = other_event_data[:,:,4:6]
        elif event_type == 'saccade':
            other_event_data = np.load(data_dir + 'fixation_matrix_gazebase_vd_' + stimulus + '.npy')
            other_event_data = other_event_data[:,:,4:6]
        
        num_add = out_data.shape[0]
        scaling = 1000.

        for i in range(num_add):
            use_id = i
            cur_seq = np.concatenate([out_data[use_id],
                                      other_event_data[use_id]], axis=0)
            flag_seq = np.concatenate([np.ones([out_data[use_id].shape[0],]),
                                       np.zeros([other_event_data[use_id].shape[0],])], axis=0)
            if i == 0:
                syn_data_sample = cur_seq
                syn_flag = flag_seq
            else:
                syn_data_sample = np.concatenate([syn_data_sample, cur_seq], axis=0)
                syn_flag = np.concatenate([syn_flag, flag_seq], axis=0)
        print(syn_data_sample.shape)
        print(syn_flag.shape)

        out = pm.events.microsaccades(positions = syn_data_sample*scaling,
                                velocities = syn_data_sample*scaling,
                                timesteps= None,
                                minimum_duration = 6,
                                threshold ='engbert2015',
                                threshold_factor = 6,
                                minimum_threshold = 1e-10,
                                include_nan = False)

        if event_type == 'fixation':
            detect_seq = np.ones([syn_data_sample.shape[0],])
        elif event_type == 'saccade':
            detect_seq = np.zeros([syn_data_sample.shape[0],])
        onsets = list(out.frame['onset'])
        offsets = list(out.frame['offset'])
        for i in range(len(onsets)):
            if event_type == 'fixation':
                detect_seq[onsets[i]:offsets[i]] = 0
            elif event_type == 'saccade':
                detect_seq[onsets[i]:offsets[i]] = 1

        event_acc = np.sum(detect_seq == syn_flag) / syn_flag.shape[0]
        print('accuracy: ' + str(event_acc))

        zero_ids = np.where(syn_flag == 0)[0]
        zero_syn = syn_flag[zero_ids]

        event_acc_detect = np.sum(zero_syn == detect_seq[zero_ids]) / zero_syn.shape[0]
        print('accuracy event: ' + str(event_acc_detect))
        
        if os.path.exists(hp_path):
            hp_data_pd = pd.read_csv(hp_path)
            hp_data = {'model_name':list(hp_data_pd['model_name']),
                       'event_acc':list(hp_data_pd['event_acc']),
                      }
        else:
            hp_data = {'model_name':[],
                       'event_acc':[],
                      }
        
        hp_data['model_name'].append(cur_model_name)
        hp_data['event_acc'].append(event_acc_detect)
        pd.DataFrame(hp_data).to_csv(hp_path, index=False)
        
if __name__ == "__main__":
    # execute only if run as a script
    main() 