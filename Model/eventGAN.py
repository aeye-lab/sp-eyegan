import time
import numpy as np
from tqdm import tqdm

import tensorflow
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation, LeakyReLU, Reshape
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import Model


class eventGAN():
    def __init__(
        self, model_config,
    ):
        self.config           = model_config
        self.generator, self.discriminator = eventGAN.build_model(model_config)
        self.gen_kernel_sizes = model_config['gen_kernel_sizes']
        self.gen_filter_sizes = model_config['gen_filter_sizes']
        self.dis_kernel_sizes = model_config['dis_kernel_sizes']
        self.dis_fiter_sizes  = model_config['dis_fiter_sizes']
        self.dis_dropout      = model_config['dis_dropout']
        self.window_size      = model_config['window_size']
        self.channels         = model_config['channels']
        self.random_size      = model_config['random_size']
        self.batch_size       = model_config['batch_size']
        self.relu_in_last     = model_config['relu_in_last']
        
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        
    def save_model(self,model_path = None):        
        self.generator.save_weights(
                model_path,
            )
    
    def load_model(self,model_path):
        self.generator.load_weights(
            model_path
        )
    
    
    @staticmethod
    def build_model(model_config,
                    ):
        tf.keras.backend.clear_session()
        input_generator = Input(
                                shape=(model_config['random_size']), name='random_input',
                            )

        dense_1 = Dense(model_config['gen_kernel_sizes'][0] * model_config['gen_filter_sizes'][0], use_bias = False) (input_generator)
        bn_1 = BatchNormalization()(dense_1)
        leaky_relu_1 = LeakyReLU()(bn_1)
        reshape_1 = Reshape((model_config['gen_kernel_sizes'][0],model_config['gen_filter_sizes'][0]))(leaky_relu_1)

        layer_list = [dense_1,bn_1,leaky_relu_1,reshape_1]
        for i in range(len(model_config['gen_filter_sizes'])-1):
            layer_list.append(Conv1DTranspose(model_config['gen_filter_sizes'][i+1], (model_config['gen_kernel_sizes'][i+1]), padding='same', use_bias = False)(layer_list[-1]))            
            if i == len(model_config['gen_filter_sizes'])-2 and not model_config['relu_in_last']:
                continue
            else:
                layer_list.append(BatchNormalization()(layer_list[-1]))
                layer_list.append(LeakyReLU()(layer_list[-1]))
                
        generator = Model(
                inputs=input_generator,
                outputs=[layer_list[-1]], name = 'generator'
            )
        
        
        input_discriminator = Input(
                                shape=(model_config['window_size'],model_config['channels']), name='random_input',
                            )

        layer_list = []
        for i in range(len(model_config['dis_kernel_sizes'])):
            if i == 0:
                layer_list.append(Conv1D(model_config['dis_fiter_sizes'][i], (model_config['dis_kernel_sizes'][i]), padding='same')(input_discriminator))
            else:
                layer_list.append(Conv1D(model_config['dis_fiter_sizes'][i], (model_config['dis_kernel_sizes'][i]), padding='same')(layer_list[-1]))
            layer_list.append(LeakyReLU()(layer_list[-1]))
            layer_list.append(Dropout(model_config['dis_dropout'])(layer_list[-1]))
        flatten = Flatten()(layer_list[-1])
        dense_out = Dense(1, activation='sigmoid')(flatten)

        discriminator = Model(
                inputs=input_discriminator,
                outputs=[dense_out], name = 'discriminator'
            )

        return generator, discriminator
    
    @staticmethod
    def discriminator_loss(real_output, fake_output):
        # This method returns a helper function to compute cross entropy loss
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(fake_output):
        # This method returns a helper function to compute cross entropy loss
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    
    @tf.function
    def train_step(self,images, verbose = 0):
        noise = tf.random.normal([self.batch_size, self.random_size])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = eventGAN.generator_loss(fake_output)
            disc_loss = eventGAN.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        if verbose > 0:
            print('generator loss: ' + str(np.round(gen_loss,decimals = 3)) +\
                  'discrimator loss: ' + str(np.round(disc_loss,decimals = 3)))
        return gen_loss, disc_loss
    
    
    def train(self, dataset, epochs, verbose = 0):
        gen_loss_list  = []
        disc_loss_list = []
        for epoch in range(epochs):
            start = time.time()
            g_losses = []
            d_losses = []
            for image_batch in dataset:
                gen_loss, disc_loss = self.train_step(image_batch, verbose = 0)
                g_losses.append(gen_loss)
                d_losses.append(disc_loss)
            gen_loss_list.append(np.mean(g_losses))
            disc_loss_list.append(np.mean(d_losses))
            if verbose > 0:
                print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start) +\
                          '    generator loss: ' + str(np.round(gen_loss_list[-1],decimals = 3)) +\
                          '    discrimator loss: ' + str(np.round(disc_loss_list[-1],decimals = 3)))
        return gen_loss_list, disc_loss_list
        
        
        
        
        
class dataGenerator():
    def __init__(
        self, 
        gan_config,
        model_config_fixation,
        model_config_saccade,
    ):
        self.gan_config    = gan_config
        self.model_config_fixation  = model_config_fixation
        self.model_config_saccade = model_config_saccade
        self.fix_window_size   = model_config_fixation['window_size']
        self.sac_window_size   = model_config_saccade['window_size']
        self.random_size   = gan_config['random_size']
        self.channels      = gan_config['channels']
        self.mean_sacc_len = gan_config['mean_sacc_len']
        self.std_sacc_len  = gan_config['std_sacc_len']
        self.mean_fix_len  = gan_config['mean_fix_len']
        self.std_fix_len   = gan_config['std_fix_len']
        self.fixation_path = gan_config['fixation_path']
        self.saccade_path  = gan_config['saccade_path']
        
        # load fixation GAN
        self.fix_model = eventGAN(self.model_config_fixation)
        self.fix_model.load_model(self.fixation_path)
        
        # load saccade GAN
        self.sac_model = eventGAN(self.model_config_saccade)
        self.sac_model.load_model(self.saccade_path)        
    
    
    def sample_scanpath_dataset_stat_model(self,
                                           stat_model,
                                           text_lists,
                                           expt_txts,
                                           num_sample_saccs = 1000,
                                           dva_threshold = 0.1,
                                           max_iter = 10,
                                           num_scanpaths_per_text = 10,
                                           num_samples = 100,
                                           output_size = 5000,
                                           store_dva_data = False,
                                           ):
        def dva_to_vel(vector):
            vel = np.array(vector[1:]) - np.array(vector[0:-1])
            vel = np.array([0] + list(vel))
            return vel
        
        output_data     = np.zeros([num_samples,output_size,self.channels])
        if store_dva_data:
            output_data_dva = np.zeros([num_samples,output_size,self.channels])
        else:
            output_data_dva = None
        max_number = int(np.ceil(num_samples / (len(text_lists) * num_scanpaths_per_text)))
        num_added = 0
        for ii in tqdm(np.arange(max_number)):
            if num_added >= num_samples:
                break
            for d_i in range(len(text_lists)):
                text_list = text_lists[d_i]
                expt_txt = expt_txts[d_i]
                x_locations, y_locations, x_dva, y_dva, fix_durations = stat_model.sample_postions_for_text(text_list,
                                                                                                    expt_txt)
                saccade_durations = None
                x_locs, y_locs = self.sample_scanpath(
                                        x_fix_locations = x_dva,
                                        y_fix_locations = y_dva,
                                        num_sample_saccs = num_sample_saccs,
                                        dva_threshold = dva_threshold,
                                        fixation_durations = fix_durations,
                                        saccade_durations = saccade_durations,
                                       )
                
                x_vels, y_vels = dva_to_vel(x_locs), dva_to_vel(y_locs)
                min_id = 0
                max_id = len(x_vels) - (output_size +1)
                sample_start_ids = np.array(np.random.choice(np.arange(min_id,max_id,1),num_scanpaths_per_text), dtype=np.int32)
                for start_id in sample_start_ids:
                    output_data[num_added,:,0] = x_vels[start_id:start_id + output_size]
                    output_data[num_added,:,1] = y_vels[start_id:start_id + output_size]
                    if store_dva_data:
                        output_data_dva[num_added,:,0] = x_locs[start_id:start_id + output_size]
                        output_data_dva[num_added,:,1] = y_locs[start_id:start_id + output_size]
                    num_added += 1
                    if num_added >= num_samples:
                        break
                if num_added >= num_samples:
                    break
        return {'vel_data': output_data,
                'dva_data': output_data_dva,
               }
    
    
    def sample_scanpath(self,
                        x_fix_locations,
                        y_fix_locations,
                        num_sample_saccs = 1000,
                        dva_threshold = 0.1,
                        max_iter = 10,
                        fixation_durations = None,
                        saccade_durations = None,
                        ):
        # helper functions
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
        
        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        def vector_length(vector):
            return np.sqrt(np.sum(np.power(vector,2)))
        
        def sample_saccades(num_sample,saccade_model,
                            mean_sacc_len = 20,
                            std_sacc_len  = 10,
                            random_size = 32,
                            sac_window_size = 30,
                            fixed_duration = None,
                            ):
            noise = tf.random.normal([num_sample, random_size])
            gen_saccades = np.array(saccade_model.generator(noise, training=False),dtype=np.float32)

            saccades = []
            for j in range(num_sample):
                # sample fixation
                if fixed_duration is None:
                    sac_duration = -1
                    while sac_duration <= 0:
                        sac_duration = int(np.random.normal(mean_sacc_len,std_sacc_len,(1)))
                else:
                    sac_duration = fixed_duration
                num_sac_samples = int(np.ceil(sac_duration / sac_window_size))

                rand_ids = np.random.permutation(np.arange(gen_saccades.shape[0]))[0:num_sac_samples]
                use_velocities = gen_saccades[rand_ids]

                use_velocities = np.reshape(use_velocities,[use_velocities.shape[0]*use_velocities.shape[1],use_velocities.shape[2]])
                saccades.append(use_velocities[0:sac_duration,:])
            return saccades

        def sample_saccade(saccade_model,
                           num_samplesize = 1000,
                           start_location = [0,0],
                           end_location = [1,1],
                           mean_sacc_len = 20,
                           std_sacc_len  = 10,
                           random_size = 32,
                           sac_window_size = 30,
                           max_deviation = 0.1,
                           max_iter = 10,
                           fixed_duration = None,
                          ):
            cur_x_distance = end_location[0] - start_location[0]
            cur_y_distance = end_location[1] - start_location[1]

            cur_sacc_length = vector_length(np.array([[cur_x_distance,cur_y_distance]]))
            # calculate amplitudes for sampled
            for i in range(max_iter):
                sampled_saccades = sample_saccades(num_samplesize,saccade_model,
                            mean_sacc_len = mean_sacc_len,
                            std_sacc_len  = std_sacc_len,
                            random_size = random_size,
                            sac_window_size = sac_window_size,
                            fixed_duration = fixed_duration,
                            )

                distances = []
                for jj in tqdm(np.arange(len(sampled_saccades)),disable = True):
                    cur_vels = sampled_saccades[jj]
                    cur_dva = vel_to_dva(cur_vels)  
                    x_dva = cur_dva[:,0]
                    y_dva = cur_dva[:,1]

                    sac_length = vector_length(np.array([x_dva[-1],y_dva[-1]]))
                    distances.append(np.abs(cur_sacc_length - sac_length))

                min_id = np.argmin(distances)
                if distances[min_id] < max_deviation:
                    break
            use_saccade = sampled_saccades[min_id]

            cur_dva = vel_to_dva(use_saccade)  
            x_dva = cur_dva[:,0]
            y_dva = cur_dva[:,1]

            angle = angle_between([cur_x_distance,cur_y_distance],[x_dva[-1],y_dva[-1]])
            try_angles = [angle,-angle]
            diffs = []
            for try_angle in try_angles:
                rotation_matrix = np.array([[np.cos(try_angle),-np.sin(try_angle)],
                                        [np.sin(try_angle),np.cos(try_angle)]])            
                rotated_points = np.transpose(np.dot(rotation_matrix,np.transpose(cur_dva)))

                x_locations = list(rotated_points[:,0] + start_location[0])
                y_locations = list(rotated_points[:,1] + start_location[1])
                diffs.append(np.sum(np.sqrt([np.power(x_locations[-1] - end_location[0],2),
                             np.power(y_locations[-1] - end_location[1],2)])))

            min_diff_id = np.argmin(diffs)
            angle = try_angles[min_diff_id]

            rotation_matrix = np.array([[np.cos(angle),-np.sin(angle)],
                                    [np.sin(angle),np.cos(angle)]])            
            rotated_points = np.transpose(np.dot(rotation_matrix,np.transpose(cur_dva)))

            x_locations = list(rotated_points[:,0] + start_location[0])
            y_locations = list(rotated_points[:,1] + start_location[1])
            return x_locations, y_locations                    
        
        # set start to first fix location
        x_location = [x_fix_locations[0]]
        y_location = [y_fix_locations[0]]
        for i in tqdm(np.arange(len(x_fix_locations)-1),disable = True):
            x_target_location = x_fix_locations[i+1]
            y_target_location = y_fix_locations[i+1]
            
            if fixation_durations is None:
                # sample fixation
                fix_duration = -1
                while fix_duration <= 0:
                    fix_duration = int(np.random.normal(self.mean_fix_len,self.std_fix_len,(1)))
            else:
                fix_duration = fixation_durations[i]
                
            num_fix_samples = int(np.ceil(fix_duration / self.fix_window_size))
            

            noise = tf.random.normal([num_fix_samples, self.random_size])
            gen_fixations = np.array(self.fix_model.generator(noise, training=False),dtype=np.float32)
            fix_data = np.reshape(gen_fixations,[gen_fixations.shape[0]*gen_fixations.shape[1],gen_fixations.shape[2]])
            fix_data = fix_data[0:fix_duration]

            # convert to dva
            fix_dva = vel_to_dva(fix_data)    
            x_location += list(fix_dva[:,0] + x_location[-1])
            y_location += list(fix_dva[:,1] + y_location[-1])

            # add saccade
            if saccade_durations is not None:
                fixed_duration = saccade_durations[i]
            else:
                fixed_duration = saccade_durations
            x_locs_sac, y_locs_sac = sample_saccade(self.sac_model,
                                           num_samplesize = num_sample_saccs,
                                           start_location = [x_location[-1],y_location[-1]],
                                           end_location = [x_target_location,y_target_location],
                                           mean_sacc_len = self.mean_sacc_len,
                                           std_sacc_len  = self.std_sacc_len,
                                           random_size = self.random_size,
                                           sac_window_size = self.sac_window_size,
                                           max_deviation = dva_threshold,
                                           max_iter = 10,
                                           fixed_duration = fixed_duration,
                                          )
            x_location += list(x_locs_sac)
            y_location += list(y_locs_sac)
            continue

            cur_x_distance = x_target_location - x_location[-1]
            cur_y_distance = y_target_location - y_location[-1]
            cur_sacc_length = vector_length(np.array([cur_x_distance,cur_y_distance]))
            counter = 0
            while np.abs(cur_x_distance) >= dva_threshold or np.abs(cur_y_distance) >= dva_threshold:
                noise = tf.random.normal([num_sample_saccs, random_size])
                gen_saccades = np.array(sac_model.generator(noise, training=False),dtype=np.float32)

                # calculate amplitudes for sampled 
                distances = []
                for jj in tqdm(np.arange(gen_saccades.shape[0]),disable = True):
                    cur_vels = gen_saccades[jj]
                    cur_dva = vel_to_dva(cur_vels)  
                    x_dva = cur_dva[:,0]
                    y_dva = cur_dva[:,1]

                    sac_length = vector_length(np.array([x_dva[-1],y_dva[-1]]))
                    distances.append(np.abs(cur_sacc_length - sac_length))

                min_id = np.argmin(distances)
                use_saccade = gen_saccades[min_id]

                cur_dva = vel_to_dva(use_saccade)
                x_dva = cur_dva[:,0]
                y_dva = cur_dva[:,1]

                cur_x_distance = x_target_location - x_location[-1]
                cur_y_distance = y_target_location - y_location[-1]

                angle = angle_between([cur_x_distance,cur_y_distance],[x_dva[-1],y_dva[-1]])            
                rotation_matrix = np.array([[np.cos(angle),-np.sin(angle)],
                                        [np.sin(angle),np.cos(angle)]])            
                rotated_points = np.transpose(np.dot(rotation_matrix,np.transpose(cur_dva)))

                x_location += list(rotated_points[:,0] + x_location[-1])
                y_location += list(rotated_points[:,1] + y_location[-1])

                cur_x_distance = x_target_location - x_location[-1]
                cur_y_distance = y_target_location - y_location[-1]
                cur_sacc_length = vector_length(np.array([cur_x_distance,cur_y_distance]))
                counter += 1
                if counter > max_iter:
                    break
        return x_location, y_location
        
        
        
    def sample_random_data(self,
                        num_samples,
                        output_size,
                        ):
        output_data = np.zeros([num_samples,output_size,self.channels])
        
        # sample fixation and saccade data
        fix_noise = tf.random.normal([self.model_config_fixation['batch_size'], self.random_size])
        cur_fix_data = self.fix_model.generator(fix_noise)
        cur_fix_counter = 0
        
        sac_noise = tf.random.normal([self.model_config_saccade['batch_size'], self.random_size])
        cur_sac_data = self.sac_model.generator(sac_noise)
        cur_sac_counter = 0
        
        for i in tqdm(np.arange(num_samples)):
            counter = 0
            while True:
                # sample fixation duration
                fix_duration = -1
                while fix_duration <= 0:
                    fix_duration = int(np.random.normal(self.mean_fix_len,self.std_fix_len,(1)))
                
                # sample saccade duration
                sac_duration = -1
                while sac_duration <= 0:
                    sac_duration = int(np.random.normal(self.mean_sacc_len,self.std_sacc_len,(1)))
                
                num_fix_samples = int(np.ceil(fix_duration / self.fix_window_size))
                num_sac_samples = int(np.ceil(sac_duration / self.sac_window_size))
                
                # select random fixations and saccades
                if cur_fix_counter + num_fix_samples >= cur_fix_data.shape[0]:
                    fix_noise = tf.random.normal([self.model_config_fixation['batch_size'], self.random_size])
                    cur_fix_data = self.fix_model.generator(fix_noise)
                    cur_fix_counter = 0
                fix_data = cur_fix_data[cur_fix_counter:cur_fix_counter + num_fix_samples]
                fix_data = np.reshape(fix_data,[fix_data.shape[0]*fix_data.shape[1],fix_data.shape[2]])
                cur_fix_counter += num_fix_samples
                
                
                if cur_sac_counter + num_sac_samples >= cur_sac_data.shape[0]:
                    sac_noise = tf.random.normal([self.model_config_saccade['batch_size'], self.random_size])
                    cur_sac_data = self.sac_model.generator(sac_noise)
                    cur_sac_counter = 0
                sac_data = cur_sac_data[cur_sac_counter:cur_sac_counter + num_sac_samples]
                sac_data = np.reshape(sac_data,[sac_data.shape[0]*sac_data.shape[1],sac_data.shape[2]])
                cur_sac_counter += num_sac_samples
                
                '''
                fix_noise = tf.random.normal([num_fix_samples, self.random_size])
                fix_data = self.fix_model.generator(fix_noise)
                fix_data = np.reshape(fix_data,[fix_data.shape[0]*fix_data.shape[1],fix_data.shape[2]])
                
                sac_noise = tf.random.normal([num_sac_samples, self.random_size])
                sac_data = self.sac_model.generator(sac_noise)
                sac_data = np.reshape(sac_data,[sac_data.shape[0]*sac_data.shape[1],sac_data.shape[2]])
                '''
                
                if counter == 0:
                    sample = np.concatenate([fix_data[0:fix_duration],
                                             sac_data[0:sac_duration]],axis=0)
                else:
                    sample = np.concatenate([sample,
                                            np.concatenate([fix_data[0:fix_duration],
                                             sac_data[0:sac_duration]],axis=0)],
                                            axis = 0)
                counter += 1
                if sample.shape[0] > output_size:
                    output_data[i] = sample[0:output_size]
                    break
                continue
        return output_data