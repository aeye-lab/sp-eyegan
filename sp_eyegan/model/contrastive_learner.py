from __future__ import annotations

import math
import os
import random
import socket
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Reshape
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import Model
from tqdm import tqdm


# CLRGaze
def _res_block(
    inp,
    filters: int,
    dilation_rate_1: int,
    dilation_rate_2: int,
    res_block_num: int,
    kernel_size: int = 3,
    strides: int = 1,
):
    conv_1 = Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate_1,
        name=f'conv_{2*res_block_num-1}',
    )(inp)
    a_1 = Activation('relu', name=f'a_{2*res_block_num-1}')(conv_1)
    bn_1 = BatchNormalization(axis=-1, name=f'bn_{2*res_block_num-1}')(a_1)
    conv_2 = Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate_2,
        name=f'conv_{2*res_block_num}',
    )(bn_1)
    ga_1 = GlobalAveragePooling1D(name=f'ga_res_{res_block_num}')(conv_2)
    dense_1 = Dense(filters // 4, activation='relu', name=f'dense_{2*res_block_num-1}')(
        Reshape((1, filters))(ga_1),
    )
    dense_2 = Dense(filters, activation='sigmoid', name=f'dense_{2*res_block_num}')(dense_1)
    mult_1 = Multiply(name=f'mult_{res_block_num}')([dense_2, conv_2])
    skip_conv_1 = Conv1D(
        filters=filters,
        kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
        name=f'skip_conv_{res_block_num}',
    )(inp)
    skip_1 = Add(name=f'skip_{res_block_num}')([mult_1, skip_conv_1])
    a_2 = Activation('relu', name=f'a_{2*res_block_num}')(skip_1)
    bn_2 = BatchNormalization(axis=-1, name=f'bn_{2*res_block_num}')(a_2)
    mp_1 = MaxPooling1D(pool_size=2, name=f'mp_{res_block_num}')(bn_2)
    return mp_1


def get_encoder_clrgaze(
        embedding_size: int = 512,
        channels: int = 2,
        window_size: int = 5000,
):
    input_velocity = Input(
        shape=[window_size, channels], name='velocity_input',
    )

    r_1 = _res_block(
        inp=input_velocity,
        filters=128,
        dilation_rate_1=1,
        dilation_rate_2=1,
        res_block_num=1,
    )
    r_2 = _res_block(
        inp=r_1,
        filters=256,
        dilation_rate_1=1,
        dilation_rate_2=2,
        res_block_num=2,
    )
    r_3 = _res_block(
        inp=r_2,
        filters=512,
        dilation_rate_1=4,
        dilation_rate_2=8,
        res_block_num=3,
    )
    ga = GlobalAveragePooling1D(name='g_a')(r_3)

    dense = Dense(
        embedding_size, activation=None, name='dense',
    )(ga)

    clrgaze_model = Model(
        inputs=input_velocity,
        outputs=[dense], name='encoder',
    )
    return clrgaze_model


#EKYT
def get_encoder_ekyt(embedding_size = 128,
                channels = 2,
                window_size = 5000):
    input_velocity = Input(
                shape=(window_size, channels), name='velocity_input',
            )

    conv_1 = Conv1D(
                filters=32, kernel_size=3, strides=1,
                padding='same',
                kernel_initializer='he_normal',
                dilation_rate=1,
                name='conv_1',
            )(input_velocity)

    res_1 = Concatenate(axis=-1, name='res_1')(
                [input_velocity, conv_1],
            )
    bn_1 = BatchNormalization(axis=-1, name='bn_1')(res_1)
    a_1 = Activation('relu', name='a_1')(bn_1)

    conv_2 = Conv1D(
                filters=32, kernel_size=3, strides=1,
                padding='same',
                kernel_initializer='he_normal',
                dilation_rate=2,
                name='conv_2',
            )(a_1)

    res_2 = Concatenate(axis=-1, name='res_2')(
                [res_1, conv_2],
            )
    bn_2 = BatchNormalization(axis=-1, name='bn_2')(res_2)
    a_2 = Activation('relu', name='a_2')(bn_2)

    conv_3 = Conv1D(
                filters=32, kernel_size=3, strides=1,
                padding='same',
                kernel_initializer='he_normal',
                dilation_rate=4,
                name='conv_3',
            )(a_2)

    res_3 = Concatenate(axis=-1, name='res_3')(
                [res_2, conv_3],
            )
    bn_3 = BatchNormalization(axis=-1, name='bn_3')(res_3)
    a_3 = Activation('relu', name='a_3')(bn_3)

    conv_4 = Conv1D(
                filters=32, kernel_size=3, strides=1,
                padding='same',
                kernel_initializer='he_normal',
                dilation_rate=8,
                name='conv_4',
            )(a_3)

    res_4 = Concatenate(axis=-1, name='res_4')(
                [res_3, conv_4],
            )
    bn_4 = BatchNormalization(axis=-1, name='bn_4')(res_4)
    a_4 = Activation('relu', name='a_4')(bn_4)

    conv_5 = Conv1D(
                filters=32, kernel_size=3, strides=1,
                padding='same',
                kernel_initializer='he_normal',
                dilation_rate=16,
                name='conv_5',
            )(a_4)

    res_5 = Concatenate(axis=-1, name='res_5')(
                [res_4, conv_5],
            )
    bn_5 = BatchNormalization(axis=-1, name='bn_5')(res_5)
    a_5 = Activation('relu', name='a_5')(bn_5)

    conv_6 = Conv1D(
                filters=32, kernel_size=3, strides=1,
                padding='same',
                kernel_initializer='he_normal',
                dilation_rate=32,
                name='conv_6',
            )(a_5)

    res_6 = Concatenate(axis=-1, name='res_6')(
                [res_5, conv_6],
            )
    bn_6 = BatchNormalization(axis=-1, name='bn_6')(res_6)
    a_6 = Activation('relu', name='a_6')(bn_6)

    conv_7 = Conv1D(
                filters=32, kernel_size=3, strides=1,
                padding='same',
                kernel_initializer='he_normal',
                dilation_rate=64,
                name='conv_7',
            )(a_6)

    res_7 = Concatenate(axis=-1, name='res_7')(
                [res_6, conv_7],
            )
    bn_7 = BatchNormalization(axis=-1, name='bn_7')(res_7)
    a_7 = Activation('relu', name='a_7')(bn_7)

    conv_8 = Conv1D(
                filters=32, kernel_size=3, strides=1,
                padding='same',
                kernel_initializer='he_normal',
                dilation_rate=1,
                name='conv_8',
            )(a_7)

    res_8 = Concatenate(axis=-1, name='res_8')(
                [res_7, conv_8],
            )
    bn_8 = BatchNormalization(axis=-1, name='bn_8')(res_8)
    a_8 = Activation('relu', name='a_8')(bn_8)

    g_a = GlobalAveragePooling1D(name='g_a')(a_8)

    dense = Dense(
                embedding_size, activation=None, name='dense',
            )(g_a)


    ekyt_model = Model(
        inputs=input_velocity,
        outputs=[dense], name = 'encoder'
    )
    return ekyt_model

def get_encoder(embedding_size = 128,
            channels = 2,
            window_size = 5000,
            model_name = 'ekyt'):
    if model_name == 'ekyt':
        return get_encoder_ekyt(embedding_size = embedding_size,
            channels = channels,
            window_size = window_size,
            )
    elif model_name == 'clrgaze':
        return get_encoder_clrgaze(embedding_size = embedding_size,
            channels = channels,
            window_size = window_size,
            )

def get_ekyt_projection_head(embedding_size):
    projection_head = keras.Sequential(
            [
                keras.Input(shape=(embedding_size,)),
                layers.Dense(embedding_size, activation='relu'),
                layers.Dense(embedding_size),
            ],
            name='projection_head',
        )
    return projection_head

def get_clrgaze_projection_head(embedding_size):
    projection_head = keras.Sequential(
            [
                keras.Input(shape=(embedding_size,)),
                layers.Dense(embedding_size, activation='relu'),
                layers.Dense(32),
            ],
            name='projection_head',
        )
    return projection_head


def get_projection_head(embedding_size,model_name):
    if model_name == 'ekyt':
        return get_ekyt_projection_head(embedding_size = embedding_size,
            )
    elif model_name == 'clrgaze':
        return get_clrgaze_projection_head(embedding_size = embedding_size,
            )

# layer to rotate data
# uses dva as input and outputs velocities: dva/s
class RotationLayer(tf.keras.layers.Layer):
    def __init__(self, max_rotation, sampling_rate = 1000., seed = 1):
        super(RotationLayer, self).__init__()
        self.max_rotation = tf.convert_to_tensor(value=max_rotation)
        self.random_generator = tf.random.Generator.from_seed(seed)
        self.sampling_rate = sampling_rate

    def dva_vel(self,output):
        out_vals = np.zeros(output.shape)
        N = out_vals.shape[1]
        # convert to velocities
        out_vals[:,1:N,:] = output[:,1:N,:] - output[:,0:N-1,:]
        return out_vals

    def call(self, inputs):
        point = tf.expand_dims(inputs, axis=-1)

        rand_value = self.random_generator.uniform([])
        cur_rot = tf.multiply(rand_value,self.max_rotation)
        sin_val = tf.convert_to_tensor(tf.math.sin(cur_rot))
        cos_val = tf.convert_to_tensor(tf.math.cos(cur_rot))
        #print(sin_val)
        #print(cos_val)
        #print(inputs)

        '''
        rotation_matrix = tf.constant([[np.cos(cur_rot),-np.sin(cur_rot)],
                                    [np.sin(cur_rot),np.cos(cur_rot)]])
        '''


        rotation_matrix = tf.convert_to_tensor([[cos_val,-sin_val],
                                    [sin_val,cos_val]])

        #print(rotation_matrix)
        rotated_point = tf.matmul(rotation_matrix, point)
        #print(rotated_point)


        out_points = tf.squeeze(rotated_point, axis=-1)
        out_vals = tf.py_function(self.dva_vel,[out_points],tf.float32)

        return tf.convert_to_tensor(value = out_vals)

def get_augmenter(name,**params):
    if name == 'crop':
        return get_crop_augmenter(params)
    elif name == 'random':
        return get_random_augmenter(params)
    elif name == 'rotation':
        return get_rotation_augmenter(params)

# augmentation module for rotation
def get_rotation_augmenter(params):
    max_rotation = params['max_rotation']
    window_size = params['window_size']
    channels = params['channels']

    inputs = layers.Input(shape=(window_size, channels))
    output = RotationLayer(max_rotation = max_rotation)(inputs)

    return Model(inputs=inputs,
        outputs=[output],
        name = 'augmenter')



# augmentation module for cropping
def get_crop_augmenter(params):
    window_size = params['window_size']
    overall_size = params['overall_size']
    channels = params['channels']

    # expand dimensions to use image random crops
    # reduce dims to end up with sequence data
    inputs = layers.Input(shape=(overall_size, channels))
    output = layers.Reshape((overall_size,1, channels))(inputs)
    output = layers.RandomCrop(window_size,1)(output)
    output = layers.Reshape((window_size,2))(output)

    return Model(inputs=inputs,
        outputs=[output],
        name = 'augmenter')

# augmentation module for cropping
def get_random_augmenter(params):
    sd = params['sd']
    window_size = params['window_size']
    channels = params['channels']

    # expand dimensions to use image random crops
    # reduce dims to end up with sequence data
    inputs = layers.Input(shape=(window_size, channels))
    output = layers.GaussianNoise(sd, seed=None)(inputs)

    return Model(inputs=inputs,
        outputs=[output],
        name = 'augmenter')


# Define the contrastive model with model-subclassing
class ContrastiveModel(keras.Model):
    def __init__(self,
                temperature,
                embedding_size,
                contrastive_augmentation,
                channels,
                window_size,
                encoder_name,
                ):
        super().__init__()

        self.temperature = temperature
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.encoder = get_encoder(embedding_size = embedding_size,
                                channels = channels,
                                window_size = window_size,
                                model_name = encoder_name)
        self.encoder_name = encoder_name

        # Non-linear MLP as projection head
        self.projection_head = get_projection_head(embedding_size = embedding_size,
                                                    model_name = encoder_name)


    def set_augmenter(self,contrastive_augmentation):
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)


    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer


        self.contrastive_loss_tracker = keras.metrics.Mean(name='c_loss')
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name='c_acc'
        )

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        (unlabeled_sequences, _) = data

        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(unlabeled_sequences, training=True)
        augmented_images_2 = self.contrastive_augmenter(unlabeled_sequences, training=True)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    def save_encoder_weights(self, path):
        self.encoder.save_weights(path)

    def load_encoder_weights(self, path):
        self.encoder.load_weights(path)


def prepare_prtrain_dataset_from_array(unlabeled_train_data,
                              batch_size = 128):
    # Labeled and unlabeled samples are loaded synchronously
    # with batch sizes selected accordingly
    unlabeled_dataset_size = unlabeled_train_data.shape[0]
    steps_per_epoch = (unlabeled_dataset_size) // batch_size
    unlabeled_batch_size = unlabeled_dataset_size // steps_per_epoch

    print(
        f'batch size is {unlabeled_batch_size} (unlabeled)'
    )
    
    with tf.device('CPU'):
        data_tensor = tf.convert_to_tensor(unlabeled_train_data)
    
    unlabeled_train_dataset =(tf.data.Dataset.from_tensor_slices((data_tensor,
                             tf.ones((unlabeled_dataset_size,1))))
                                .shuffle(buffer_size=10 * unlabeled_batch_size)
                                .batch(unlabeled_batch_size)
                                .prefetch(buffer_size=tf.data.AUTOTUNE)
                             )

    return unlabeled_train_dataset


def prepare_dataset_from_array(unlabeled_train_data,
                              labeled_train_data, labelded_train_label,
                              labeled_test_data, labeled_test_label,
                              batch_size = 128):
    # Labeled and unlabeled samples are loaded synchronously
    # with batch sizes selected accordingly
    unlabeled_dataset_size = unlabeled_train_data.shape[0]
    labeled_dataset_size   = labeled_train_data.shape[0]
    steps_per_epoch = (unlabeled_dataset_size + labeled_dataset_size) // batch_size
    unlabeled_batch_size = unlabeled_dataset_size // steps_per_epoch
    labeled_batch_size = labeled_dataset_size // steps_per_epoch

    print(
        f'batch size is {unlabeled_batch_size} (unlabeled) + {labeled_batch_size} (labeled)'
    )

    unlabeled_train_dataset =(tf.data.Dataset.from_tensor_slices((unlabeled_train_data,
                             tf.ones((unlabeled_dataset_size,1))))
                                .shuffle(buffer_size=10 * unlabeled_batch_size)
                                .batch(unlabeled_batch_size)#
                             )

    labeled_train_dataset = (tf.data.Dataset.from_tensor_slices((labeled_train_data,
                             labelded_train_label))
                                .shuffle(buffer_size=10 * unlabeled_batch_size)
                                .batch(unlabeled_batch_size)
                            )

    test_dataset = (tf.data.Dataset.from_tensor_slices((labeled_test_data,
                             labeled_test_label))
                            .batch(batch_size)
                            .prefetch(buffer_size=tf.data.AUTOTUNE)
                   )



    # Labeled and unlabeled datasets are zipped together
    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, labeled_train_dataset, test_dataset
