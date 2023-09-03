from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Reshape
from tensorflow.keras.losses import mse
from tensorflow.keras.metrics import Mean

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_var) * epsilon


def get_vae_encoder(seq_len: int, latent_dim: int, n_channel: int) -> Model:
    encoder_input = Input((seq_len, n_channel))
    e_conv_1 = Conv1D(
        filters=64, kernel_size=7, activation='relu', strides=1,
        padding='same',
    )(encoder_input)
    e_avg_1 = AveragePooling1D()(e_conv_1)
    e_conv_2 = Conv1D(
        filters=32, kernel_size=7, activation='relu', strides=1,
        padding='same',
    )(e_avg_1)
    e_avg_2 = AveragePooling1D()(e_conv_2)
    e_conv_3 = Conv1D(
        filters=2, kernel_size=16, activation='relu', strides=1,
        padding='same',
    )(e_avg_2)
    flatten = Conv1D(filters=1, kernel_size=1, padding='same')(e_conv_3)
    flatten = Flatten()(e_conv_3)#flatten)
    z_mean = Dense(latent_dim)(flatten)
    z_var = Dense(latent_dim)(flatten)
    z = Sampling()([z_mean, z_var])

    encoder = Model(encoder_input, [z_mean, z_var, z], name='encoder')
    #encoder.summary()
    return encoder


def get_vae_decoder(latent_dim: int, n_channel: int) -> Model:
    decoder_input = Input((latent_dim,))
    reshape = Reshape((1, latent_dim))(decoder_input)
    d_conv_1 = Conv1DTranspose(
        filters=32, kernel_size=32, activation='relu', strides=1,
    )(reshape)
    d_conv_2 = Conv1DTranspose(
        filters=64, kernel_size=33,  activation='relu', strides=1,
    )(d_conv_1)
    d_conv_3 = Conv1DTranspose(
        filters=n_channel, kernel_size=1, activation='relu', strides=1,
    )(d_conv_2)
    decoder = Model(decoder_input, d_conv_3, name='decoder')
    #decoder.summary()
    return decoder


class VAE(Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss = Mean(name='total_loss')
        self.rec_loss = Mean(name='rec_loss')
        self.kl_loss = Mean(name='kl_loss')

    def call(self, inputs):
        _, _, x = self.encoder(inputs)
        return self.decoder(x)

    def metrics(self):
        return [
            self.total_loss,
            self.rec_loss,
            self.kl_loss,
        ]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            rec_loss = tf.reduce_mean(
                mse(data, reconstruction),
                axis=-1,
            )
            kl_loss = -0.5 * (1 + z_var - tf.square(z_mean) - tf.exp(z_var))
            kl_loss = tf.reduce_mean(kl_loss, axis=-1)
            total_loss = rec_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        #print(grads[0])
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights),
        )
        self.total_loss.update_state(total_loss)
        self.rec_loss.update_state(rec_loss)
        self.kl_loss.update_state(kl_loss)
        return {
            'total_loss': self.total_loss.result(),
            'rec_loss': self.rec_loss.result(),
            'kl_loss': self.kl_loss.result(),
        }

    def train(self, data, epochs, verbose = 0):
        rec_loss_list  = []
        kl_loss_list = []
        for epoch in range(epochs):
            start = time.time()
            r_losses = []
            kl_losses = []
            counter = 0
            for data_batch in data:
                #print(counter)
                locc_dict = self.train_step(data_batch)
                r_losses.append(locc_dict['rec_loss'])
                kl_losses.append(locc_dict['kl_loss'])
                counter += 1
            rec_loss_list.append(np.mean(r_losses))
            kl_loss_list.append(np.mean(kl_losses))
            if verbose > 0:
                print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start) +\
                          '    reconstruction loss: ' + str(np.round(rec_loss_list[-1],decimals = 3)) +\
                          '    kl loss: ' + str(np.round(kl_loss_list[-1],decimals = 3)))
        return rec_loss_list, kl_loss_list

    def save_model(self,model_path = None):
        self.decoder.save_weights(
                model_path,
            )

    def load_model(self,model_path):
        self.decoder.load_weights(
            model_path
        )
