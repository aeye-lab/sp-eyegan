from __future__ import annotations

import argparse
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pymovements.gaze.transforms import pix2deg
from pymovements.gaze.transforms import pos2vel
from scipy import interpolate
from scipy.spatial import distance
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

import config as config
from sp_eyegan.gpu_selection import select_gpu
from sp_eyegan.model import contrastive_learner as contrastive_learner
from sp_eyegan.preprocessing import data_loader as data_loader


def cut_data(data,
                    max_vel = 0.5,
                    window = 5000,
                    verbose = 0,
                    ):
    x_diff = data['xvel']
    x_diff[x_diff > max_vel] = max_vel
    x_diff[x_diff < -max_vel] = -max_vel
    y_diff = data['yvel']
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


def help_roc_auc(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return .5
    else:
        return roc_auc_score(y_true, y_pred)


def auroc(y_true, y_pred):
    return tf.py_function(help_roc_auc, (y_true, y_pred), tf.double)


def configure_gpu(args: argparse.Namespace) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1.
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)


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

def get_model(args: argparse.Namespace) -> Model:
    print(' === Loading model ===')

    if args.pretrain_model_path is None:
        contrastive_augmentation = {
            'window_size': args.window_size,
            'channels': args.channels,
            'name': 'random',
            'sd': args.sd,
        }

        # load contrastive pretrained model
        pretraining_model = contrastive_learner.ContrastiveModel(
            temperature=args.temperature,
            embedding_size=args.embedding_size,
            contrastive_augmentation=contrastive_augmentation,
            channels=args.channels,
            window_size=args.window_size,
            encoder_name=args.encoder_name,
        )
    else:
        contrastive_augmentation = {
            'window_size': args.window_size,
            'channels': args.channels,
            'name': 'random',
            'sd': args.sd,
        }
        pretraining_model = contrastive_learner.ContrastiveModel(args.temperature,
                                embedding_size=args.embedding_size,
                                contrastive_augmentation=contrastive_augmentation,
                                channels=args.channels,
                                window_size=args.window_size,
                                encoder_name=args.encoder_name,
                                )

    dense_out = Dense(
        1, activation='sigmoid', name='dense_out',
    )(pretraining_model.encoder.get_layer('dense').output)
    classification_model = Model(
                inputs=pretraining_model.encoder.get_layer('velocity_input').input,
                outputs=[dense_out], name='classifier',
    )

    if args.pretrain_model_path is not None:
        print('load weights from: ' + str(args.pretrain_model_path))
        classification_model = load_weights(classification_model, args.pretrain_model_path)
    if args.print_model_summary:
        classification_model.summary()
    return classification_model



def get_argument_parser() -> argparse.Namespace:
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--stimulus', type=str, default='TEX')
    parser.add_argument('--encoder_name', type=str, default='ekyt')

    parser.add_argument('--pretrain_augmentation_mode', type=str, default=None)#'random')
    parser.add_argument('--pretrain_stimulus', type=str, default='text')
    parser.add_argument('--pretrain_encoder_name', type=str, default='ekyt')
    parser.add_argument('--pretrain_scanpath_model', type=str, default='random')
    parser.add_argument('--pretrain_sd', type=float, default=0.1)
    parser.add_argument('--pretrain_sd_factor', type=float, default=1.25)
    parser.add_argument('--pretrain_max_rotation', type=float, default=6.)
    parser.add_argument('--pretrain_num_pretrain_instances', type=int, default=300)
    parser.add_argument('--pretrain_window_size', type=int, default=300)
    parser.add_argument('--pretrain_model_dir', type=str, default='pretrain_model/')
    parser.add_argument('--pretrain_channels', type=int, default=2)
    parser.add_argument('--pretrain_checkpoint', type=int, default=-1)
    parser.add_argument('--pretrain_data_suffix', type=str, default='')
    parser.add_argument('--pretrain_model_path', type=str, default=None)


    parser.add_argument('--result_dir', type=str, default='results/')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--flag_redo', action='store_true')
    parser.add_argument('--num_train', type=int, default=-1)
    parser.add_argument('--inner_cv_loops', type=int, default=2)
    parser.add_argument('--save_suffix', type=str, default='')
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
    encoder_name = args.encoder_name
    result_dir = args.result_dir
    gpu = args.gpu
    fold = args.fold
    n_folds = args.n_folds
    flag_redo = args.flag_redo

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

    # params
    gender_dict = {
               1:'male',
               2:'female',
              }
    sampling_rate = 60
    window_sec = 5
    window_size = sampling_rate * window_sec
    args.window_size = window_size
    args.channels = 2
    args.sd = pretrain_sd
    args.sd_factor = pretrain_sd_factor
    gof_data_dir = config.GOF_DATA_DIR
    gof_info_path = config.GOF_INFO_PATH


    param_grid = {
        'n_estimators': [500, 1000],
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [2,4,8,16,32, None],
        'criterion' :['entropy'],
        'n_jobs': [-1]
    }

    if fold == -1:
        folds = list(np.arange(n_folds, dtype=np.int32))
    else:
        folds = [fold]

    for cur_fold in folds:
        args.fold = cur_fold

        if args.pretrain_model_path is None:
            if args.fine_tune != 0:
                pretrain_model_dir = config.CONTRASTIVE_PRETRAINED_MODELS_DIR
                if args.encoder_name == 'clrgaze':
                    args.embedding_size = 512
                    args.pretrain_model_path = config.CONTRASTIVE_PRETRAINED_MODELS_DIR + 'clrgaze_random_window_size_300_sd_0.05_sd_factor_1.25_embedding_size_512_stimulus_video_model_random_-1baseline_60'
                elif args.encoder_name == 'ekyt':
                    args.embedding_size = 128
                    args.pretrain_model_path = config.CONTRASTIVE_PRETRAINED_MODELS_DIR + 'ekyt_random_window_size_300_sd_0.05_sd_factor_1.25_embedding_size_128_stimulus_video_model_random_-1baseline_60'
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

        if args.pretrain_model_path is not None:
            args.pretrained_model_name = args.pretrain_model_path.split('/')[-1]
        else:
            if args.encoder_name == 'clrgaze':
                args.pretrained_model_name = 'CLRGAZE'
            elif args.encoder_name == 'ekyt':
                args.pretrained_model_name = 'EKYT'

        result_save_path = result_dir + str(args.pretrained_model_name) + '_fold' + str(args.fold) + '_gender.joblib'

        if args.num_train != -1:
            result_save_path = result_save_path.replace('.joblib', '_num_train_' + str(args.num_train) + '.joblib')

        if args.save_suffix != '':
            result_save_path = result_save_path.replace('.joblib', '_' + str(args.save_suffix) + '.joblib')
        embedding_save_path = result_save_path.replace('.joblib','_embeddings.joblib')


        if not flag_redo and os.path.exists(result_save_path):
            print('skip evaluation (already exists) [' + str(result_save_path) + ']')
            return 0


        # params
        args.temperature = 0.1
        args.print_model_summary = True
        args.learning_rate = 0.0001
        args.num_epochs = 100
        args.batch_size = 32


        # set up gpu
        configure_gpu(args)

        # load data
        gof_info = pd.read_csv(gof_info_path)
        subject_gender_dict = dict()
        for d_i in range(len(gof_info)):
            cur_line = gof_info.loc[d_i]
            subject_gender_dict[int(cur_line['id'])] = int(cur_line['gender'])

        counter = 0
        num_add = 100000
        subject_label = np.zeros([num_add,])
        gender_label = np.zeros([num_add,])
        trial_label = np.zeros([num_add,])
        gaze_seq_data = np.zeros([num_add,window_size,2])

        csv_files = os.listdir(gof_data_dir)
        #print('number of files: ' +  str(len(csv_files)))

        for f_i in tqdm(np.arange(len(csv_files))):
            if not csv_files[f_i].endswith('.csv'):
                continue
            cur_csv_data = pd.read_csv(gof_data_dir + csv_files[f_i], header=None)

            cur_subject = int(csv_files[f_i].split('_')[1].replace('sub',''))
            cur_trial   = int(csv_files[f_i].split('_')[2].split('.')[0].replace('trial',''))

            if cur_subject not in subject_gender_dict:
                continue

            tmp_fix_coord = cur_csv_data[[0, 1]]
            _coord_fix_arr = tmp_fix_coord.to_numpy()

            # http://antoinecoutrot.magix.net/public/databases.html
            deg_fix_arr = pix2deg(
                _coord_fix_arr,
                screen_px=(1280, 1024),
                screen_cm=(38, 30),
                distance_cm=57,
                origin='center',
            )

            vel_fix_arr = pos2vel(deg_fix_arr, sampling_rate=sampling_rate)
            # convert deg/s to deg/ms
            vel_fix_arr = vel_fix_arr / 1000.

            tmp_data = cut_data(pd.DataFrame({'xvel':vel_fix_arr[:,0],
                               'yvel':vel_fix_arr[:,1],
                              }),
                            max_vel = 0.5,
                            window = window_size,
                            verbose = 0,
                            )

            cur_len = tmp_data.shape[0]
            while counter + cur_len > gaze_seq_data.shape[0]:
                gaze_seq_data = np.concatenate([gaze_seq_data, np.zeros([num_add,5000,2])], axis=0)
                subject_label = np.concatenate([subject_label, np.zeros([num_add,])], axis=0)
                gender_label = np.concatenate([gender_label, np.zeros([num_add,])], axis=0)
                trial_label = np.concatenate([trial_label, np.zeros([num_add,])], axis=0)


            gaze_seq_data[counter:counter+cur_len] = tmp_data
            gender_label[counter:counter+cur_len]  = subject_gender_dict[cur_subject]
            trial_label[counter:counter+cur_len]   = cur_trial
            subject_label[counter:counter+cur_len] = cur_subject
            counter += cur_len

        gaze_seq_data = gaze_seq_data[0:counter]
        gender_label = gender_label[0:counter]
        trial_label = trial_label[0:counter]
        subject_label = subject_label[0:counter]

        gender_label_zero_one = np.zeros([len(gender_label),1])
        gender_label_zero_one[np.array(gender_label) == 2] = 1

        unique_subjects = list(np.unique(subject_label))
        print('number of subjects: ' + str(len(unique_subjects)))

        genders = []
        for unique_sub in unique_subjects:
            genders.append(gender_dict[subject_gender_dict[unique_sub]])
        print(np.unique(genders, return_counts = True))

        np.random.seed(args.fold)
        shuffled_user = np.random.permutation(unique_subjects)

        kfold = KFold(n_splits=n_folds)
        for _fold, (train_idx, test_idx) in enumerate(kfold.split(shuffled_user)):
            if _fold == args.fold:
                break

        train_user = shuffled_user[train_idx]
        test_user  = shuffled_user[test_idx]
        train_ids  = np.where(np.isin(np.array(subject_label), train_user))[0]
        test_ids   = np.where(np.isin(np.array(subject_label), test_user))[0]
        print('number train instances: ' + str(len(train_ids)))
        print('number test instances: ' + str(len(test_ids)))

        train_subjects = np.array(subject_label)[train_ids]
        test_subjects  = np.array(subject_label)[test_ids]
        train_trials   = np.array(trial_label)[train_ids]
        test_trials    = np.array(trial_label)[test_ids]
        train_label    = np.array(gender_label_zero_one)[train_ids]
        test_label     = np.array(gender_label_zero_one)[test_ids]
        train_data     = gaze_seq_data[train_ids]
        test_data      = gaze_seq_data[test_ids]


        # use specified amount ouf training instances
        if args.num_train != -1:
            rand_ids = np.random.permutation(np.arange(len(train_label)))
            use_ids  = rand_ids[0:args.num_train]
            train_subjects = train_subjects[use_ids]
            train_trials = train_trials[use_ids]
            train_label = train_label[use_ids]
            train_data = train_data[use_ids]

        # create training data
        orig_data = train_data
        orig_sub  = np.array(train_subjects, dtype=np.int32)

        # get model and train model and predict
        classification_model = get_model(args)

        # get feature extractions
        embedding_model = Model(
            inputs=classification_model.input,
            outputs=classification_model.get_layer('dense').output,
        )

        test_embedding = embedding_model.predict(
            test_data,
            batch_size=args.batch_size,
        )

        train_embedding = embedding_model.predict(
            train_data,
            batch_size=args.batch_size,
        )

        # distance (zero-shot)
        print('evaluate zero-shot (distance based)')
        pos_train_ids = np.where(train_label == 1)[0]
        pos_mean = np.mean(train_embedding[pos_train_ids], axis=0)

        predictions_distances = distance.cdist(
            test_embedding,
            np.reshape(pos_mean, [1,len(pos_mean)]), metric='cosine',
        )

        # rf with features
        print('evaluate RF on features')
        grid_search_verbosity = 1

        # rf
        rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, verbose = grid_search_verbosity, cv = args.inner_cv_loops)
        rf.fit(train_embedding, train_label.ravel())

        best_parameters = rf.best_params_
        predictions_rf = rf.predict_proba(test_embedding)


        # NN
        print('fine-tuning on features')
        optimizer = Adam(learning_rate=args.learning_rate)
        classification_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', auroc],
        )
        callbacks = [
            EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True,
            ),
        ]


        history = classification_model.fit(
            train_data,
            train_label,
            validation_split=.2,
            epochs=args.num_epochs,
            callbacks=callbacks,
            verbose=1,
            batch_size=args.batch_size,
        )

        predictions_nn = classification_model.predict(test_data)

        # save to file
        joblib.dump({'test_subjects':test_subjects,
                     'test_label':test_label,
                     'test_trials':test_trials,
                     'predictions_distances':predictions_distances,
                     'predictions_rf':predictions_rf,
                     'predictions_nn':predictions_nn,
                     }, result_save_path, compress=3, protocol=2)

        # save embeddings
        joblib.dump({'test_subjects':test_subjects,
                     'train_subjects':train_subjects,
                     'test_label':test_label,
                     'train_label':train_label,
                     'test_trials':test_trials,
                     'train_embedding':'train_embedding',
                     'test_embedding':'test_embedding',
                     }, embedding_save_path, compress=3, protocol=2)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
