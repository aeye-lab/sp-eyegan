from __future__ import annotations

import argparse
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
import pymovements as pm
import seaborn as sns
import tensorflow
import tensorflow as tf
from scipy import interpolate
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

import config as config
from sp_eyegan.gpu_selection import select_gpu
from sp_eyegan.model import contrastive_learner as contrastive_learner
from sp_eyegan.preprocessing import data_loader as data_loader



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

def cut_data(data,
                    max_vel = 0.5,
                    window = 5000,
                    verbose = 0,
                    ):
    x_left = np.array(data['x_left_pos'], dtype=np.float32)
    y_left = np.array(data['y_left_pos'], dtype=np.float32)
    x_diff = x_left[1:-1] - x_left[:-2]
    x_diff[x_diff > max_vel] = max_vel
    x_diff[x_diff < -max_vel] = -max_vel
    y_diff = y_left[1:-1] - y_left[:-2]
    y_diff[y_diff > max_vel] = max_vel
    y_diff[y_diff < -max_vel] = -max_vel

    num_windows = int(np.floor(len(x_diff) / window))
    if verbose != 0:
        disable = False
    else:
        disable = True
    out_matrix = np.zeros([num_windows, window, 2])
    for i in tqdm(np.arange(num_windows), disable=disable):
        out_matrix[i] = np.array([x_diff[i*window:(i+1)*window],
                                 y_diff[i*window:(i+1)*window]]).T
    out_matrix[np.isnan(out_matrix)] = 0.
    return out_matrix

def get_indicies(
    enrolled_users,
    impostors,
    enrollment_sessions,
    test_sessions,
    data_user,
    data_sessions,
    data_seqIds,
    seconds_per_session=None,
    random_state=42,
    num_enrollment = 12,
):

    random.seed(random_state)
    idx_enrollment = []
    for enrolled_user in enrolled_users:
        cur_ids = np.logical_and(
                np.isin(data_user, enrolled_user),
                np.isin(data_sessions, enrollment_sessions),
        )
        pos_ids = np.where(cur_ids)[0]
        random.shuffle(pos_ids)
        idx_enrollment += list(pos_ids[0:num_enrollment])

    test_idx = np.logical_and(
        np.logical_or(
            np.isin(data_user, enrolled_users),
            np.isin(data_user, impostors),
        ),
        np.isin(data_sessions, test_sessions),
    )

    return (idx_enrollment, test_idx)

def get_user_similarity_scores_and_labels(
    cosine_distances, y_enrollment, y_test, enrollment_users, impostors, window_size=1,
    sim_to_enroll='min',
    verbose=0,
):
    """
    :param cosine_distances: cosine distances of all pairs of enrollment and test instances, n_test x n_enrollment
    :param y_enrollment: n_enrollment labels for enrollment instances
    :param y_test: n_test labels for test instances
    :param enrollment_users: all ids of enrolled users
    :param impostors: all ids of impostors
    :param window_size: number of instances the similarity score should be based upon
    :param sim_to_enroll: how to compute simalarity to enrollment users; should be in {'min','mean'}
    :return: similarity scores of two persons; true labels: test person is impostor (0), same person (1) or another enrolled person (2)
    """
    if verbose == 0:
        disable = True
    else:
        disable = False

    scores = []      # similarity score between two users, based on number of test instances specified by window size
    # true labels: test person is 0 (impostor),1 (correct), 2 (confused)
    labels = []
    person_one = []
    person_two = []
    for test_user in tqdm(np.unique(y_test), disable=disable):
        idx_test_user = y_test == test_user

        # iterate over each possible window start position for test user
        dists_test_user = cosine_distances[idx_test_user, :]
        if str(window_size) != 'all':
            for i in range(dists_test_user.shape[0] - window_size):
                dists_test_user_window = dists_test_user[i:i+window_size, :]

                # calculate score and prediction and create true label for each window
                distances_to_enrolled = []
                enrolled_u = []
                enrolled_persons = np.unique(y_enrollment)

                for enrolled_user in enrolled_persons:
                    idx_enrolled_user = y_enrollment == enrolled_user

                    # calculate aggregated distance of instances in window with each enrolled user seperately
                    dists_test_user_window_enrolled_user = dists_test_user_window[
                        :,
                        idx_enrolled_user
                    ]

                    # aggregate distances for each test sequence to all enrolled sequences by taking the minimum distance
                    if sim_to_enroll == 'min':
                        dists_test_sequences_of_window = np.min(
                            dists_test_user_window_enrolled_user, axis=1,
                        )  # n_test_sequences x 1 array
                    elif sim_to_enroll == 'mean':
                        dists_test_sequences_of_window = np.mean(
                            dists_test_user_window_enrolled_user, axis=1,
                        )  # n_test_sequences x 1 array

                    # aggregate min distances of all test sequences in this window by taking the mean
                    window_mean_dist = np.mean(dists_test_sequences_of_window)

                    distances_to_enrolled.append(window_mean_dist)
                    enrolled_u.append(enrolled_user)

                    # create corresponding true label for this window
                    if test_user in list(impostors):
                        label = 0  # test user of this window is an impostor
                    elif test_user in list(enrollment_users):
                        if test_user == enrolled_user:
                            label = 1  # test user of this window is this enrolled user
                        else:
                            label = 2  # test user of this window is another enrolled user
                    else:
                        print(
                            f'user {test_user} is neither enrolled user nor impostor',
                        )
                        label = -1  # should never happen

                    scores.append(1-window_mean_dist)
                    labels.append(label)
                    person_one.append(enrolled_user)
                    person_two.append(test_user)
        else:
            dists_test_user_window = dists_test_user

            # calculate score and prediction and create true label for each window
            distances_to_enrolled = []
            enrolled_u = []
            enrolled_persons = np.unique(y_enrollment)

            for enrolled_user in enrolled_persons:
                idx_enrolled_user = y_enrollment == enrolled_user

                # calculate aggregated distance of instances in window with each enrolled user seperately
                dists_test_user_window_enrolled_user = dists_test_user_window[
                    :,
                    idx_enrolled_user
                ]

                # aggregate distances for each test sequence to all enrolled sequences by taking the minimum distance
                if sim_to_enroll == 'min':
                    dists_test_sequences_of_window = np.min(
                        dists_test_user_window_enrolled_user, axis=1,
                    )  # n_test_sequences x 1 array
                elif sim_to_enroll == 'mean':
                    dists_test_sequences_of_window = np.mean(
                        dists_test_user_window_enrolled_user, axis=1,
                    )  # n_test_sequences x 1 array

                # aggregate min distances of all test sequences in this window by taking the mean
                window_mean_dist = np.mean(dists_test_sequences_of_window)

                distances_to_enrolled.append(window_mean_dist)
                enrolled_u.append(enrolled_user)

                # create corresponding true label for this window
                if test_user in list(impostors):
                    label = 0  # test user of this window is an impostor
                elif test_user in list(enrollment_users):
                    if test_user == enrolled_user:
                        label = 1  # test user of this window is this enrolled user
                    else:
                        label = 2  # test user of this window is another enrolled user
                else:
                    print(
                        f'user {test_user} is neither enrolled user nor impostor',
                    )
                    label = -1  # should never happen

                scores.append(1-window_mean_dist)
                labels.append(label)
                person_one.append(enrolled_user)
                person_two.append(test_user)

    not_nan_ids = np.where(np.isnan(scores) == 0)[0]
    return np.array(scores)[not_nan_ids], np.array(labels)[not_nan_ids], np.array(person_one)[not_nan_ids], np.array(person_two)[not_nan_ids]

def get_scores_and_labels(
    test_embeddings,
    test_user, #
    test_sessions, # session
    test_seqIds, # round
    window_sizes,
    n_train_users=0,
    n_enrolled_users=20,
    n_impostors=5,
    n_enrollment_sessions=3,
    n_test_sessions=1,
    user_test_sessions=None,
    enrollment_sessions=None,
    verbose=1,
    random_state=None,
    seconds_per_session=None,
    num_enrollment = 12,
    ):

    if random_state is not None:
        random.seed(random_state)

    score_dicts = dict()
    label_dicts = dict()
    person_one_dicts = dict()
    person_two_dicts = dict()

    if verbose > 0:
        print('number of different users: ' + str(len(np.unique(test_user))))

    users = list(np.unique(test_user))

    # shuffle users
    random.shuffle(users)

    enrolled_users = users[n_train_users:n_train_users+n_enrolled_users]
    impostors = users[
        n_train_users +
        n_enrolled_users: n_train_users + n_enrolled_users + n_impostors
    ]
    if verbose > 0:
        print('number of different enrollment users: ' + str(len(np.unique(enrolled_users))))
        print('number of different impostors users: ' + str(len(np.unique(impostors))))

    sessions = np.unique(test_sessions)
    random.shuffle(sessions)
    cur_enrollment_sessions = sessions[0: n_enrollment_sessions]
    cur_test_sessions = sessions[n_enrollment_sessions:n_enrollment_sessions + n_test_sessions]


    if verbose > 0:
        print(f'enrolled_users: {enrolled_users} enroll-sessions: {cur_enrollment_sessions} test-sessions: {cur_test_sessions}')

    (idx_enrollment, test_idx) = get_indicies(
        enrolled_users,
        impostors,
        cur_enrollment_sessions,
        cur_test_sessions,
        test_user,
        test_sessions,
        test_seqIds,
        seconds_per_session=seconds_per_session,
        random_state=random_state,
        num_enrollment = num_enrollment,
    )
    if verbose > 0:
        print('len idx_enrollment: ' + str(len(idx_enrollment)))
        print('len test_idx: ' + str(np.sum(test_idx)))

    test_feature_vectors = test_embeddings[test_idx, :]
    enrollment_feature_vectors = test_embeddings[idx_enrollment,:]

    # labels for embedding feature vectors:
    y_enrollment_user = test_user[idx_enrollment]
    y_test_user = test_user[test_idx]

    #print('number of different enrollment users: ' + str(len(np.unique(y_enrollment_user))))
    #print('number of different test users: ' + str(len(np.unique(y_test_user))))

    dists = distance.cdist(
        test_feature_vectors,
        enrollment_feature_vectors, metric='cosine',
    )

    for window_size in window_sizes:
        scores, labels , person_one, person_two = get_user_similarity_scores_and_labels(
            dists,
            y_enrollment_user,
            y_test_user,
            enrolled_users,
            impostors,
            window_size=window_size,
            verbose=verbose,
        )
        cur_key = str(window_size)
        score_dicts[cur_key] = scores.tolist()
        label_dicts[cur_key] = labels.tolist()
        person_one_dicts[cur_key] = person_one.tolist()
        person_two_dicts[cur_key] = person_two.tolist()
    return (score_dicts, label_dicts, person_one_dicts, person_two_dicts)



def save_model(model,
               model_path,
              ):
    model.save_weights(
            model_path,
        )

def load_weights(model,
                 model_path):
    model.load_weights(
        model_path,
    )
    return model

def get_biometric_model(encoder_name,
                        model_path = None,
                        num_classes = None,
                        window_size = 5000,
                        channels = 2,
                        sd = 0.1,
                        temperature = 0.1,
                        loss_weight_ce = 0.1,
                        loss_ms = 1.,
                        learning_rate = 1e-2,
                        ):
    if encoder_name == 'clrgaze':
        embedding_size = 512
    elif encoder_name == 'ekyt':
        embedding_size = 128

    contrastive_augmentation = {'window_size': window_size, 'channels':channels, 'name':'random','sd':sd}
    pretraining_model = contrastive_learner.ContrastiveModel(temperature=temperature,
                                                    embedding_size = embedding_size,
                                                    contrastive_augmentation = contrastive_augmentation,
                                                    channels = channels,
                                                    window_size = window_size,
                                                    encoder_name = encoder_name)

    embedding = Model(
            inputs=pretraining_model.get_layer('encoder').input,
            outputs=pretraining_model.get_layer('encoder').output,
        )

    input_nn = embedding.input
    dense = embedding.layers[-1].output

    # add biometric classification head
    bn_final = BatchNormalization(axis=-1, name='bn_final')(dense)
    a_final = Activation('relu', name='a_final')(bn_final)

    output_ekyt = Dense(
        num_classes, activation='softmax',
        name='final_sm',
    )(a_final)

    ekyt_model = Model(
        inputs=input_nn,
        outputs=[output_ekyt,dense],
    )

    LossFunc    =     {'final_sm':cat_crossentropy, 'dense':ms_loss}
    lossWeights =     {'final_sm':loss_weight_ce, 'dense':loss_ms}


    opt_ekyt = tensorflow.keras.optimizers.Adam(
                learning_rate=learning_rate,
            )

    ekyt_model.compile(
                optimizer=opt_ekyt,
                loss = LossFunc, loss_weights=lossWeights, metrics=['accuracy'],
                run_eagerly = True,
            )

    if model_path is not None:
        ekyt_model = load_weights(ekyt_model, model_path)
    return ekyt_model



###############################################################################################################################

###########################################################
#
#   BATCH GENERATOR
#
###########################################################

# 256 (16 subjects * 16 events each)
def create_batch(X,y,batchsize = 256):
    num_subjects = int(np.ceil(np.sqrt(batchsize)))
    num_events = num_subjects

    #print(y.shape)
    y_idx = np.argmax(y,axis=1)
    #print(y_idx)
    #print(y_idx.shape)
    unique_ids = np.array(list(np.unique(y_idx)))
    #print('unique_ids.shape: ' + str(unique_ids.shape))
    random.shuffle(unique_ids)
    use_subs = unique_ids[0:num_subjects]
    use_ids = []
    for i in range(len(use_subs)):
        cur_ids = np.where(y_idx == use_subs[i])[0]
        random.shuffle(cur_ids)
        use_ids += list(cur_ids[0:num_events])
    if len(use_ids) < batchsize:
        num_add = batchsize - len(use_ids)
        rand_ids = np.arange(len(y_idx))
        random.shuffle(rand_ids)
        use_ids += list(rand_ids[0:num_add])
    X_out = X[use_ids]
    y_out = y[use_ids]
    return X_out,y_out


def generate_data(X,y, batchsize = 256):
    while True:
        X_out,y_out = create_batch(X,y, batchsize = batchsize)
        yield (X_out,y_out)

###########################################################
#
#   LOSS FUNCTIONS
#
###########################################################

# code from: https://github.com/geonm/tf_ms_loss/blob/master/tf_ms_loss.py
# slightly adjusted to fit our data structure
def ms_loss(labels, embeddings, alpha=2.0, beta=50.0, lamb=0.5, eps=0.1, ms_mining=True):
    '''
    ref: http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    official codes: https://github.com/MalongTech/research-ms-loss
    '''

    # convert labels to index vector containing the label
    labels = tf.argmax(labels, axis=1)

    #tf.print(labels.shape)
    #tf.print(embeddings.shape)

    # make sure emebedding should be l2-normalized
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    labels = tf.reshape(labels, [-1, 1])

    #tf.print(labels.shape)
    #tf.print(embeddings.shape)

    batch_size = embeddings.get_shape().as_list()[0]

    adjacency = tf.equal(labels, tf.transpose(labels))
    adjacency_not = tf.logical_not(adjacency)

    #tf.print(adjacency)

    mask_pos = tf.cast(adjacency, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32)
    mask_neg = tf.cast(adjacency_not, dtype=tf.float32)

    sim_mat = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)
    sim_mat = tf.maximum(sim_mat, 0.0)

    pos_mat = tf.multiply(sim_mat, mask_pos)
    neg_mat = tf.multiply(sim_mat, mask_neg)

    if ms_mining:
        max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
        tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
        min_val = tf.reduce_min(tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True) + tmp_max_val

        max_val = tf.tile(max_val, [1, batch_size])
        min_val = tf.tile(min_val, [1, batch_size])

        mask_pos = tf.where(pos_mat < max_val + eps, mask_pos, tf.zeros_like(mask_pos))
        mask_neg = tf.where(neg_mat > min_val - eps, mask_neg, tf.zeros_like(mask_neg))

    pos_exp = tf.exp(-alpha * (pos_mat - lamb))
    pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

    neg_exp = tf.exp(beta * (neg_mat - lamb))
    neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

    pos_term = tf.math.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / alpha
    neg_term = tf.math.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / beta

    loss = tf.reduce_mean(pos_term + neg_term)

    return loss


def cat_crossentropy(y_true, y_pred):
    #print('cat_crossentropy y_true: ' + str(y_true.shape))
    #print('cat_crossentropy y_pred: ' + str(y_pred.shape))
    #tf.print(y_true.shape)
    #tf.print(y_pred.shape)
    loss_function = CategoricalCrossentropy()
    crossentropy = loss_function(y_true,y_pred)
    #crossentropy = tf.reduce_mean(tf.keras.metrics.categorical_crossentropy(y_true, y_pred))
    #tf.print(crossentropy)
    #tf.print(crossentropy.shape)
    return crossentropy

###########################################################
#
#   LR SCHEDULER
#
###########################################################

# cosine annealing learning rate
def decayed_learning_rate(step, decay_steps = 70, initial_learning_rate = 0.01, alpha=0.0):
    step = np.min([step, decay_steps])
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_learning_rate * decayed


# we set lr to zero to use own schedule depending on max_lr
def ekyt_learning_rate_scheduler(max_lr = 1e-2,first_steps = 30, max_epochs = 100):
    def one_cycle_cosine_annealing_ekyt(step,lr=0.0,
                                        ):
        # epochs starts @ 1
        step += 1
        #print(step)
        if step < first_steps:
            val = decayed_learning_rate(step=first_steps-step,
                                         decay_steps=first_steps,initial_learning_rate=max_lr)
            print('lr: ' + str(val) + ' [max_lr: ' + str(max_lr) + ']')
            return val
        else:
            val = decayed_learning_rate(step=step - first_steps, decay_steps=max_epochs-first_steps,
                                  initial_learning_rate=max_lr)
            print('lr: ' + str(val) + ' [max_lr: ' + str(max_lr) + ']')
            return val
    return one_cycle_cosine_annealing_ekyt


def configure_gpu(args: argparse.Namespace) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1.
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

def get_argument_parser() -> argparse.Namespace:
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--stimulus', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='gazebase')
    parser.add_argument('--encoder_name', type=str, default='ekyt')

    parser.add_argument('--pretrain_augmentation_mode', type=str, default=None)#'random')
    parser.add_argument('--pretrain_stimulus', type=str, default='text')
    parser.add_argument('--pretrain_encoder_name', type=str, default='ekyt')
    parser.add_argument('--pretrain_scanpath_model', type=str, default='random')
    parser.add_argument('--pretrain_sd', type=float, default=0.1)
    parser.add_argument('--pretrain_sd_factor', type=float, default=1.25)
    parser.add_argument('--pretrain_max_rotation', type=float, default=6.)
    parser.add_argument('--pretrain_num_pretrain_instances', type=int, default=5000)
    parser.add_argument('--pretrain_window_size', type=int, default=5000)
    parser.add_argument('--pretrain_model_dir', type=str, default='pretrain_model/')
    parser.add_argument('--pretrain_channels', type=int, default=2)
    parser.add_argument('--pretrain_checkpoint', type=int, default=-1)
    parser.add_argument('--pretrain_data_suffix', type=str, default='')
    parser.add_argument('--pretrain_model_path', type=str, default=None)

    parser.add_argument('--result_dir', type=str, default='results/')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--num_train_user', type=int, default=100)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--flag_redo', type=int, default=0)
    parser.add_argument('--max_rounds', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--fine_tune', type=int, default=1)
    args = parser.parse_args()
    return args




def main() -> int:
    args = get_argument_parser()

    print(' === Configuring GPU ===')

    if args.gpu == -1:
        args.gpu = select_gpu(7700)
    print('~' * 79)
    print(f'{args.gpu=}')
    print('~' * 79)


    # get params
    stimulus = args.stimulus
    dataset_name = args.dataset_name
    encoder_name = args.encoder_name
    result_dir = args.result_dir
    gpu = args.gpu
    num_train_user = args.num_train_user
    fold = args.fold
    flag_redo = args.flag_redo
    if flag_redo == 1:
        flag_redo = True
    else:
        flag_redo = False
    pretrain_augmentation_mode = args.pretrain_augmentation_mode
    pretrain_stimulus = args.pretrain_stimulus
    pretrain_encoder_name = args.pretrain_encoder_name
    pretrain_scanpath_model = args.pretrain_scanpath_model
    pretrain_sd = args.pretrain_sd
    pretrain_sd_factor = args.pretrain_sd_factor
    pretrain_max_rotation = args.pretrain_max_rotation
    pretrain_num_pretrain_instances = args.pretrain_num_pretrain_instances
    pretrain_model_dir = args.pretrain_model_dir
    pretrain_window_size = args.pretrain_window_size
    pretrain_channels = args.pretrain_channels
    pretrain_checkpoint = args.pretrain_checkpoint
    pretrain_data_suffix = args.pretrain_data_suffix
    max_rounds = args.max_rounds


    if fold == -1:
        folds = list(np.arange(args.n_folds, dtype=np.int32))
    else:
        folds = [fold]

    for cur_fold in folds:
        args.fold = cur_fold

        if args.pretrain_model_path is None:
            if args.fine_tune != 0:
                pretrain_model_dir = config.CONTRASTIVE_PRETRAINED_MODELS_DIR
                if args.encoder_name == 'clrgaze':
                    args.embedding_size = 512
                    args.pretrain_model_path = config.CONTRASTIVE_PRETRAINED_MODELS_DIR + 'clrgaze_random_window_size_5000_sd_0.1_sd_factor_1.25_embedding_size_512_stimulus_video_model_random_-1baseline_1000'
                elif args.encoder_name == 'ekyt':
                    args.embedding_size = 128
                    args.pretrain_model_path = config.CONTRASTIVE_PRETRAINED_MODELS_DIR + 'ekyt_random_window_size_5000_sd_0.1_sd_factor_1.25_embedding_size_128_stimulus_video_model_random_-1baseline_1000'
            else:
                if args.encoder_name == 'clrgaze':
                    args.embedding_size = 512
                elif args.encoder_name == 'ekyt':
                    args.embedding_size = 128
        else:
            if args.encoder_name == 'clrgaze':
                args.embedding_size = 512
            elif args.encoder_name == 'ekyt':
                args.embedding_size = 128

        model_path = args.pretrain_model_path

        # create dummy augmentation
        args.contrastive_augmentation = {'window_size': args.pretrain_window_size, 'channels':args.pretrain_channels, 'name':'random','sd':args.pretrain_sd}
        if args.pretrain_model_path is not None:
            args.pretrained_model_name = args.pretrain_model_path.split('/')[-1]
        else:
            if args.encoder_name == 'clrgaze':
                args.pretrained_model_name = 'CLRGAZE'
            elif args.encoder_name == 'ekyt':
                args.pretrained_model_name = 'EKYT'
        result_save_path = result_dir + str(args.pretrained_model_name) + '_fold' + str(args.fold) + '_biometric_' + '_dataset_' + str(dataset_name) +\
                            '.joblib'

        if max_rounds != -1:
            result_save_path = result_save_path.replace('.joblib','_max_rounds' + str(max_rounds) + '.joblib')

        result_save_path = result_save_path.replace('.joblib','_num_folds' + str(args.n_folds) + '.joblib')


        if not flag_redo and os.path.exists(result_save_path):
            print('skip evaluation (already exists)')
            return 0


        # params
        loss_weight_ce = 0.1
        loss_ms = 1.
        learning_rate = 1e-2
        batch_size = args.batch_size
        epochs = 100
        temperature = 0.1

        # biometric eval params
        n_train_users = 0
        n_enrolled_users = -1
        n_impostors = 0
        window_sizes = [1,2,12]

        configure_gpu(args)

        if dataset_name == 'gazebase':
            # load gazebase data
            dataset = pm.Dataset('GazeBase', path=config.GAZE_BASE_DIR)
            subset = dict()
            if subset is not None:
                subset['task_name'] = stimulus

            if max_rounds != -1:
                subset['round_id'] = list(np.arange(1,max_rounds+1,1))

            try:
                dataset.load(subset=subset)
            except:
                dataset.download()
                dataset.load(subset=subset)


            num_pairs = len(dataset.gaze)
            counter = 0
            num_add = 100000
            round_ids = np.zeros([num_add,])
            subject_ids = np.zeros([num_add,])
            session_ids = np.zeros([num_add,])
            gaze_seq_data = np.zeros([num_add,5000,2])
            for i in tqdm(np.arange(num_pairs)):
                cur_data = dataset.gaze[i]
                try:
                    cur_data.unnest('position', output_columns=['x_left_pos', 'y_left_pos'])
                except:
                    pass
                cur_data = cur_data.frame
                cur_gazebase_data = cut_data(cur_data,
                                                    window=5000,
                                                    max_vel=.5,
                                                    )
                cur_len = cur_gazebase_data.shape[0]
                while counter + cur_len > gaze_seq_data.shape[0]:
                    gaze_seq_data = np.concatenate([gaze_seq_data, np.zeros([num_add,5000,2])], axis=0)
                    round_ids = np.concatenate([round_ids, np.zeros([num_add,])], axis=0)
                    subject_ids = np.concatenate([subject_ids, np.zeros([num_add,])], axis=0)
                    session_ids = np.concatenate([session_ids, np.zeros([num_add,])], axis=0)
                gaze_seq_data[counter:counter+cur_len] = cur_gazebase_data
                round_ids[counter:counter+cur_len]   = int(np.unique(cur_data['round_id']))
                subject_ids[counter:counter+cur_len] = int(np.unique(cur_data['subject_id']))
                session_ids[counter:counter+cur_len] = int(np.unique(cur_data['session_id']))
                counter += cur_len
            gaze_seq_data = gaze_seq_data[0:counter]
            round_ids = round_ids[0:counter]
            subject_ids = subject_ids[0:counter]
            session_ids = session_ids[0:counter]

            unique_user = list(np.unique(subject_ids))
            print('number of unique_users: ' + str(len(unique_user)))
            np.random.seed(args.fold)
            shuffled_user = np.random.permutation(unique_user)
            train_user = shuffled_user[0:num_train_user]
            test_user  = shuffled_user[num_train_user:]
            train_ids  = np.where(np.isin(np.array(subject_ids), train_user))[0]
            test_ids   = np.where(np.isin(np.array(subject_ids), test_user))[0]
            print('number train instances: ' + str(len(train_ids)))
            print('number test instances: ' + str(len(test_ids)))


            train_rounds   = np.array(round_ids)[train_ids]
            test_rounds    = np.array(round_ids)[test_ids]
            train_subjects = np.array(subject_ids)[train_ids]
            test_subjects  = np.array(subject_ids)[test_ids]
            train_sessions = np.array(session_ids)[train_ids]
            test_sessions  = np.array(session_ids)[test_ids]
            train_data     = gaze_seq_data[train_ids]
            test_data      = gaze_seq_data[test_ids]

            # create training data
            orig_data = train_data
            orig_sub  = np.array(train_subjects, dtype=np.int32)

            le = LabelEncoder()
            sub_transformed = le.fit_transform(orig_sub)

            n_train_users_f = len(np.unique(sub_transformed))
            y_train = to_categorical(
                sub_transformed, num_classes=n_train_users_f,
            )

            seq_len = orig_data.shape[1]
            n_channels = orig_data.shape[2]
            num_classes = y_train.shape[1]

            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            for train_idx, validation_idx in skf.split(orig_data, sub_transformed):
                break

        elif dataset_name == 'judo':
            dataset = pm.Dataset('JuDo1000', path=config.JUDO_BASE_DIR)
            try:
                dataset.load()
            except:
                dataset.download()
                dataset.load()

            # convert pixel data to degrees of visual angle
            dataset.pix2deg()

            num_pairs = len(dataset.gaze)
            counter = 0
            num_add = 100000
            subject_ids = np.zeros([num_add,])
            session_ids = np.zeros([num_add,])
            gaze_seq_data = np.zeros([num_add,5000,2])
            for i in tqdm(np.arange(num_pairs)):                
                cur_data = dataset.gaze[i]
                try:
                    cur_data.unnest('position', output_columns=['x_left_pos', 'y_left_pos',
                                                                'x_right_pos', 'y_right_pos'])
                except:
                    pass
                cur_data = cur_data.frame
                
                cur_cut_data = cut_data(cur_data,
                                                    window=5000,
                                                    max_vel=.5,
                                                    )

                cur_len = cur_cut_data.shape[0]
                while counter + cur_len > gaze_seq_data.shape[0]:
                    gaze_seq_data = np.concatenate([gaze_seq_data, np.zeros([num_add,5000,2])], axis=0)
                    subject_ids = np.concatenate([subject_ids, np.zeros([num_add,])], axis=0)
                    session_ids = np.concatenate([session_ids, np.zeros([num_add,])], axis=0)
                gaze_seq_data[counter:counter+cur_len] = cur_cut_data
                subject_ids[counter:counter+cur_len] = int(np.unique(cur_data['subject_id']))
                session_ids[counter:counter+cur_len] = int(np.unique(cur_data['session_id']))
                counter += cur_len
            gaze_seq_data = gaze_seq_data[0:counter]
            subject_ids = subject_ids[0:counter]
            session_ids = session_ids[0:counter]


            unique_user = list(np.unique(subject_ids))
            print('number of unique_users: ' + str(len(unique_user)))
            np.random.seed(args.fold)
            shuffled_user = np.random.permutation(unique_user)
            train_user = shuffled_user[0:num_train_user]
            test_user  = shuffled_user[num_train_user:]
            train_ids  = np.where(np.isin(np.array(subject_ids), train_user))[0]
            test_ids   = np.where(np.isin(np.array(subject_ids), test_user))[0]
            print('number train instances: ' + str(len(train_ids)))
            print('number test instances: ' + str(len(test_ids)))


            train_subjects = np.array(subject_ids)[train_ids]
            test_subjects  = np.array(subject_ids)[test_ids]
            train_sessions = np.array(session_ids)[train_ids]
            test_sessions  = np.array(session_ids)[test_ids]
            train_data     = gaze_seq_data[train_ids]
            test_data      = gaze_seq_data[test_ids]

            # create training data
            orig_data = train_data
            orig_sub  = np.array(train_subjects, dtype=np.int32)

            le = LabelEncoder()
            sub_transformed = le.fit_transform(orig_sub)

            n_train_users_f = len(np.unique(sub_transformed))
            y_train = to_categorical(
                sub_transformed, num_classes=n_train_users_f,
            )

            seq_len = orig_data.shape[1]
            n_channels = orig_data.shape[2]
            num_classes = y_train.shape[1]

            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            for train_idx, validation_idx in skf.split(orig_data, sub_transformed):
                break

        # get model
        biometric_model = get_biometric_model(encoder_name = encoder_name,
                                    model_path = model_path,
                                    num_classes = num_classes,
                                    temperature = temperature,
                                    learning_rate = learning_rate,
                                    )

        # get zero-shot embedding
        embedding_model = Model(
            inputs=biometric_model.input,
            #outputs=biometric_model.get_layer('a_final').output,
            outputs=biometric_model.get_layer('dense').output,
        )
        
        embedding_zero_shot = embedding_model.predict(
            test_data,
            batch_size=batch_size,
        )


        callbacks = [LearningRateScheduler(ekyt_learning_rate_scheduler(max_lr = learning_rate,
                                                        first_steps = 30, max_epochs = 100))]

        steps_per_epoch=int(len(train_idx) / batch_size)
        steps_per_val = int(len(validation_idx) / batch_size)

        history = biometric_model.fit(
            generate_data(orig_data[train_idx, :],
                    y_train[train_idx, :],
                    batchsize = batch_size,
            ),
            validation_data=(
                generate_data(orig_data[validation_idx, :],
                y_train[validation_idx, :],
                batchsize = batch_size,
                )
            ),
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=steps_per_val,
        )



        # get embedding for test set
        embedding_model = Model(
            inputs=biometric_model.input,
            #outputs=biometric_model.get_layer('a_final').output,
            outputs=biometric_model.get_layer('dense').output,
        )

        embedding = embedding_model.predict(
            test_data,
            batch_size=batch_size,
        )

        # evaluate embeddings
        if n_enrolled_users == -1:
            n_enrolled_users = len(np.unique(test_subjects))

        if dataset_name == 'gazebase':
            score_dicts, label_dicts, person_one_dicts, person_two_dicts = get_scores_and_labels(
                test_embeddings = embedding,
                test_user = np.array(test_subjects), #
                test_sessions = np.array(test_sessions), # session
                test_seqIds = np.array(test_rounds), # round
                window_sizes = window_sizes,
                n_train_users=n_train_users,
                n_enrolled_users=n_enrolled_users,
                n_impostors=n_impostors,
                n_enrollment_sessions=1,
                n_test_sessions=1,
                user_test_sessions=None,
                enrollment_sessions=None,
                verbose=1,
                random_state=args.fold,
                seconds_per_session=None,
                num_enrollment = 1,
                )

            score_dicts_zero_shot, label_dicts_zero_shot, person_one_dicts_zero_shot, person_two_dicts_zero_shot = get_scores_and_labels(
                test_embeddings = embedding_zero_shot,
                test_user = np.array(test_subjects), #
                test_sessions = np.array(test_sessions), # session
                test_seqIds = np.array(test_rounds), # round
                window_sizes = window_sizes,
                n_train_users=n_train_users,
                n_enrolled_users=n_enrolled_users,
                n_impostors=n_impostors,
                n_enrollment_sessions=1,
                n_test_sessions=1,
                user_test_sessions=None,
                enrollment_sessions=None,
                verbose=1,
                random_state=args.fold,
                seconds_per_session=None,
                num_enrollment = 1,
                )

        elif dataset_name == 'judo':
            score_dicts, label_dicts, person_one_dicts, person_two_dicts = get_scores_and_labels(
                test_embeddings = embedding,
                test_user = np.array(test_subjects), #
                test_sessions = np.array(test_sessions), # session
                test_seqIds = None, # round
                window_sizes = window_sizes,
                n_train_users=n_train_users,
                n_enrolled_users=n_enrolled_users,
                n_impostors=n_impostors,
                n_enrollment_sessions=3,
                n_test_sessions=1,
                user_test_sessions=None,
                enrollment_sessions=None,
                verbose=1,
                random_state=args.fold,
                seconds_per_session=None,
                num_enrollment = 1,
                )

            score_dicts_zero_shot, label_dicts_zero_shot, person_one_dicts_zero_shot, person_two_dicts_zero_shot = get_scores_and_labels(
                test_embeddings = embedding_zero_shot,
                test_user = np.array(test_subjects), #
                test_sessions = np.array(test_sessions), # session
                test_seqIds = None, # round
                window_sizes = window_sizes,
                n_train_users=n_train_users,
                n_enrolled_users=n_enrolled_users,
                n_impostors=n_impostors,
                n_enrollment_sessions=3,
                n_test_sessions=1,
                user_test_sessions=None,
                enrollment_sessions=None,
                verbose=1,
                random_state=args.fold,
                seconds_per_session=None,
                num_enrollment = 1,
                )

        # save to files
        joblib.dump({'score_dicts':score_dicts,
                     'label_dicts':label_dicts,
                     'person_one_dicts':person_one_dicts,
                     'person_two_dicts':person_two_dicts,
                     'embeddings':embedding,
                     'score_dicts_zero_shot':score_dicts_zero_shot,
                     'label_dicts_zero_shot':label_dicts_zero_shot,
                     'person_one_dicts_zero_shot':person_one_dicts_zero_shot,
                     'person_two_dicts_zero_shot':person_two_dicts_zero_shot,
                     'embedding_zero_shot':embedding_zero_shot,
                     'test_subjects':test_subjects,
                     }, result_save_path, compress=3, protocol=2)

        for window_size in window_sizes:
            np.savez(result_save_path.replace('.joblib','_window_size' + str(window_size)),
                     scores = score_dicts[str(window_size)],
                     labels = label_dicts[str(window_size)],
                     person_one = person_one_dicts[str(window_size)],
                     person_two = person_two_dicts[str(window_size)],
                     scores_zero_shot = score_dicts_zero_shot[str(window_size)],
                     labels_zero_shot = label_dicts_zero_shot[str(window_size)],
                     person_one_zero_shot = person_one_dicts_zero_shot[str(window_size)],
                     person_two_zero_shot = person_two_dicts_zero_shot[str(window_size)],
                    )

            '''
            joblib.dump({'scores':score_dicts[str(window_size)],
                     'labels':label_dicts[str(window_size)],
                     'person_one':person_one_dicts[str(window_size)],
                     'person_two':person_two_dicts[str(window_size)],
                     }, result_save_path.replace('.joblib','_window_size' + str(window_size) + '.joblib'), compress=3, protocol=2)
            '''
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
