from __future__ import annotations

import argparse
import os
import random

import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import config
from sp_eyegan.gpu_selection import select_gpu
from sp_eyegan.load_sb_sat_data import get_sb_sat_data
from sp_eyegan.model import contrastive_learner


from sp_eyegan.model.helpers import helpers
from sp_eyegan.model.preprocessing import feature_extraction as feature_extraction


def help_roc_auc(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return .5
    else:
        return roc_auc_score(y_true, y_pred)


def auroc(y_true, y_pred):
    return tf.py_function(help_roc_auc, (y_true, y_pred), tf.double)


def get_auc_for_full_page(
        predictions: np.array,
        x_arr: np.array | tf.Tensor,
        x_arr_screen_id: np.array | tf.Tensor,
        y_test: np.array | tf.Tensor,
        y_all: np.array | tf.Tensor,
        test_arr: np.array,
) -> float:
    mean_x_arr = []
    new_y_test = []
    books = ['dickens', 'flytrap', 'genome', 'northpole']
    for screen_id in range(1, 6):
        for test_reader in test_arr:
            for book in books:
                # instance has to be in:
                # - screen_id
                # - test_reader
                # - book
                and_cond = np.logical_and(
                    np.logical_and(
                        np.isin(x_arr_screen_id, screen_id)[:, 0],
                        np.isin(y_all[:, -1], test_reader),
                    ),
                    np.isin(y_all[:, -2], book),
                )
                if sum(and_cond) == 0:
                    continue
                else:
                    mean_x_arr.append(np.mean(predictions[and_cond]))
                    # all labels are conditioned on subj-book-page-combination,
                    # hence can take first
                    new_y_test.append(y_test[and_cond][0])
    rocauc = roc_auc_score(new_y_test, mean_x_arr)
    return rocauc


def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--base-model', action='store_true')
    parser.add_argument('--encoder_name', type=str)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--window-size', type=int, default=5000)
    parser.add_argument('--overall-size', type=int, default=5000)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--sd', type=float, default=0.1)
    parser.add_argument('--sd-factor', type=float, default=1.25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--check-point-saver', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--per-process-gpu-memory-fraction', type=float, default=.5)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--pretrained-model-name', '-m', type=str)
    parser.add_argument('--split-criterion', '-s', type=str, default='subj')
    parser.add_argument('--print-model-summary', action='store_true')
    parser.add_argument('--normalization', action='store_true')
    parser.add_argument('--models-batch', type=int)

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
    parser.add_argument('--inner_cv_loops', type=int, default=2)
    parser.add_argument('--fine_tune', type=int, default=1)


    parser.add_argument(
        '--problem-setting', type=str, default='native',
        choices=['acc', 'difficulty', 'subj_acc', 'native'],
    )
    args = parser.parse_args()
    return args


def create_rf_data(pixel_data):    
    verbose = 0
    for i in tqdm(np.arange(pixel_data.shape[0])):
        cur_data = pd.DataFrame({'x_pixel':pixel_data[i,:,0],
                                 'y_pixel':pixel_data[i,:,1],
                                 #'eye_closure':np.zeros([len(train_data[i,:,0]),]),
                                 #'blink':np.zeros([len(train_data[i,:,0]),]),
                                 'corrupt':np.zeros([len(pixel_data[i,:,0]),]),
                                 #'pupil':np.zeros([len(train_data[i,:,0]),]),
                                 })
        columns = ['x_pixel','y_pixel','corrupt']#,'eye_closure','blink','corrupt','pupil']
        channels = {columns[i]:i for i in range(len(columns))}
        data = np.array(cur_data, dtype=np.float32)
        data = np.expand_dims(data, axis=0)
        rf_features, feature_names = feature_extraction.compute_combined_features(
                            data=data, data_format=channels,
                            verbose=verbose,
                            screenPX_x=1280,
                            screenPX_y=1024,
                            screenCM_x=38,
                            screenCM_y=30,
                            distanceCM=57,
        )
        if i == 0:
            rf_data = rf_features
        else:
            rf_data = np.concatenate([rf_data, rf_features], axis=0)
    rf_data[np.isnan(rf_data)] = 0.0
    return rf_data

def configure_gpu(args: argparse.Namespace) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = args.per_process_gpu_memory_fraction
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
        pretraining_model = contrastive_learner.ContrastiveModel(args.temperature,
                                embedding_size=args.embedding_size,
                                contrastive_augmentation=args.contrastive_augmentation,
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


def four_problem_setting_eval_on_sb(args: argparse.Namespace) -> int:
    print(' === Loading data ===')
    if args.encoder_name == 'rf':
        x_data, y_label, label_dict = get_sb_sat_data(args.window_size,
                                      verbose = 1,
                                      rf_features = True,
                                      )
        x_data_rf = create_rf_data(x_data[:, :, [0, 1]])
    else:
        x_data, y_label, label_dict = get_sb_sat_data(args.window_size,
                                      verbose = 1,
                                      )
    if args.split_criterion == 'subj':
        split_criterion = np.array(sorted(list(set(y_label[:, label_dict[args.split_criterion]]))))
        n_splits = 5

    problem_settings = ['native', 'difficulty', 'acc', 'subj_acc']
    results_dict = {}
    results_distance = {}
    results_rf = {}
    for problem_setting in problem_settings:
        print(f' Evaluation  on {problem_setting=} '.center(79, '='))
        problem_setting_mean = np.mean(y_label[:, label_dict[problem_setting]])
        y_label[:, label_dict[problem_setting]] = np.array(
            y_label[:, label_dict[problem_setting]] > problem_setting_mean, dtype=int,
        )
        kfold = KFold(n_splits=n_splits)
        five_sec_eval_aucs = {}
        trained_aucs = {}
        five_sec_eval_aucs_distance = {}
        trained_aucs_distance = {}
        five_sec_eval_aucs_rf = {}
        trained_aucs_rf = {}
        np.random.seed(12)
        random.seed(12)
        tf.random.set_seed(12)
        if args.pretrained_model_name:
            model_name = args.pretrained_model_name.split('/')[-1]
        else:
            raise NotImplementedError
        for fold, (train_idx, test_idx) in enumerate(kfold.split(split_criterion)):
            np.random.seed(fold)
            random.seed(fold)
            tf.random.set_seed(fold)
            print(f' {fold=}')
            N_train_split = split_criterion[train_idx]
            N_test_split = split_criterion[test_idx]
            if args.split_criterion == 'subj':
                if args.encoder_name != 'rf':
                    X_train = x_data[
                        np.isin
                        (
                            y_label[:, label_dict[args.split_criterion]],
                            N_train_split,
                        ),
                    ][:, :, [0, 1]]
                else:
                    X_train = x_data_rf[
                        np.isin
                        (
                            y_label[:, label_dict[args.split_criterion]],
                            N_train_split,
                        ),
                    ]
                y_train = y_label[
                    np.isin
                    (
                        y_label[:, label_dict[args.split_criterion]],
                        N_train_split,
                    ),
                ]
                if args.encoder_name != 'rf':
                    X_test = x_data[
                        np.isin
                        (
                            y_label[:, label_dict[args.split_criterion]],
                            N_test_split,
                        ),
                    ][:, :, [0, 1]]
                else:
                    X_test = x_data_rf[
                        np.isin
                        (
                            y_label[:, label_dict[args.split_criterion]],
                            N_test_split,
                        ),
                    ]
                X_test_screen_id = x_data[
                    np.isin
                    (
                        y_label[:, label_dict[args.split_criterion]],
                        N_test_split,
                    ),
                ][:, :, 2]
                y_test = y_label[
                    np.isin
                    (
                        y_label[:, label_dict[args.split_criterion]],
                        N_test_split,
                    ),
                ]
            
            if args.encoder_name != 'rf':
                # formatting
                if args.normalization:
                    dx_scaler = StandardScaler()
                    X_train[:, :, 0] = dx_scaler.fit_transform(X_train[:, :, 0])
                    X_test[:, :, 0] = dx_scaler.transform(X_test[:, :, 0])
                    dy_scaler = StandardScaler()
                    X_train[:, :, 1] = dy_scaler.fit_transform(X_train[:, :, 1])
                    X_test[:, :, 1] = dy_scaler.transform(X_test[:, :, 1])

                X_train = tf.cast(X_train, dtype=tf.float32)
                value_not_nan = tf.dtypes.cast(
                    tf.math.logical_not(tf.math.is_nan(X_train)), dtype=tf.float32,
                )
                X_train = tf.math.multiply_no_nan(X_train, value_not_nan)
                y_train = np.array(y_train[:, label_dict[problem_setting]], dtype=int)
                X_test = tf.cast(X_test, dtype=tf.float32)
                value_not_nan = tf.dtypes.cast(
                    tf.math.logical_not(tf.math.is_nan(X_test)), dtype=tf.float32,
                )
                X_test = tf.math.multiply_no_nan(X_test, value_not_nan)
                y_test_ps = np.array(y_test[:, label_dict[problem_setting]], dtype=int)

                # train model
                model = get_model(args)

                # get feature extractions
                embedding_model = Model(
                    inputs=model.input,
                    outputs=model.get_layer('dense').output,
                )

                test_embedding = embedding_model.predict(
                    X_test,
                    batch_size=args.batch_size,
                )

                train_embedding = embedding_model.predict(
                    X_train,
                    batch_size=args.batch_size,
                )


                print('evaluate zero-shot (distance based)')
                pos_train_ids = np.where(y_train == 1)[0]
                pos_mean = np.mean(train_embedding[pos_train_ids], axis=0)

                predictions_distances = distance.cdist(
                    test_embedding,
                    np.reshape(pos_mean, [1,len(pos_mean)]), metric='cosine',
                )

                # rf with features
                print('evaluate RF on features')
                grid_search_verbosity = 1
                param_grid = {
                    'n_estimators': [500, 1000],
                    'max_features': ['sqrt', 'log2'],
                    'max_depth' : [2,4,8,16,32, None],
                    'criterion' :['entropy'],
                    'n_jobs': [-1]
                }

                # rf
                rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, verbose = grid_search_verbosity, cv = args.inner_cv_loops)
                rf.fit(train_embedding, y_train.ravel())

                best_parameters = rf.best_params_
                predictions_rf = rf.predict_proba(test_embedding)


                optimizer = Adam(learning_rate=args.learning_rate)
                model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', auroc],
                )
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss', patience=50, restore_best_weights=True,
                    ),
                ]
                if args.verbose:
                    verbose = 1
                else:
                    verbose = 0
                print('~' * 79)
                print('Training model'.center(79))
                print('~' * 79)
                model.fit(
                    X_train,
                    y_train,
                    validation_split=.2,
                    epochs=args.num_epochs,
                    callbacks=callbacks,
                    verbose=verbose,
                    batch_size=args.batch_size,
                )
                predictions_nn = model.predict(X_test)

                #print(np.unique(model.predict(X_test), return_counts=True))

                # NN results
                five_sec_eval_auc = roc_auc_score(y_test_ps, predictions_nn)
                five_sec_eval_aucs[fold] = five_sec_eval_auc
                trained_auc = get_auc_for_full_page(
                    predictions_nn, X_test, X_test_screen_id, y_test_ps, y_test, N_test_split,
                )
                trained_aucs[fold] = trained_auc
                results_dict['model_name'] = model_name + '_fine-tune'
                results_dict[f'{problem_setting}_auc_fold_{fold}'] = trained_auc
                results_dict[f'{problem_setting}_five_sec_eval_auc_fold_{fold}'] = five_sec_eval_auc
                results_dict[f'{problem_setting}_mean_auc'] = np.mean(list(trained_aucs.values()))
                results_dict[f'{problem_setting}_five_sec_eval_mean_auc'] = np.mean(list(five_sec_eval_aucs.values()))

                # distance results
                five_sec_eval_auc = roc_auc_score(y_test_ps, predictions_distances)
                five_sec_eval_aucs_distance[fold] = five_sec_eval_auc
                trained_auc = get_auc_for_full_page(
                    predictions_distances, X_test, X_test_screen_id, y_test_ps, y_test, N_test_split,
                )
                trained_aucs_distance[fold] = trained_auc
                results_distance['model_name'] = model_name + '_distance'
                results_distance[f'{problem_setting}_auc_fold_{fold}'] = trained_auc
                results_distance[f'{problem_setting}_five_sec_eval_auc_fold_{fold}'] = five_sec_eval_auc
                results_distance[f'{problem_setting}_mean_auc'] = np.mean(list(trained_aucs_distance.values()))
                results_distance[f'{problem_setting}_five_sec_eval_mean_auc'] = np.mean(list(five_sec_eval_aucs_distance.values()))

                # rf results
                five_sec_eval_auc = roc_auc_score(y_test_ps, predictions_rf[:,1])
                five_sec_eval_aucs_rf[fold] = five_sec_eval_auc
                trained_auc = get_auc_for_full_page(
                    predictions_rf[:,1], X_test, X_test_screen_id, y_test_ps, y_test, N_test_split,
                )
                trained_aucs_rf[fold] = trained_auc
                results_rf['model_name'] = model_name + '_rf'
                results_rf[f'{problem_setting}_auc_fold_{fold}'] = trained_auc
                results_rf[f'{problem_setting}_five_sec_eval_auc_fold_{fold}'] = five_sec_eval_auc
                results_rf[f'{problem_setting}_mean_auc'] = np.mean(list(trained_aucs_rf.values()))
                results_rf[f'{problem_setting}_five_sec_eval_mean_auc'] = np.mean(list(five_sec_eval_aucs_rf.values()))

                '''
                print(f'{five_sec_eval_aucs.values()}')
                print(f'{trained_aucs.values()}')
                print(
                    f'{np.mean(list(five_sec_eval_aucs.values())):.2f}'
                    f'+/-{np.std(list(five_sec_eval_aucs.values()))/len(five_sec_eval_aucs):.2f}',
                )
                print(
                    f'{np.mean(list(trained_aucs.values())):.2f}'
                    f'+/-{np.std(list(trained_aucs.values()))/len(trained_aucs):.2f}',
                )
                if args.base_model:
                    model_name = 'base_model_' + model_name
                '''

                #model.save_weights(
                #    config.TRAINED_CLASSIFICATION_MODELS_DIR,
                #)
            else:
            
                y_train = np.array(y_train[:, label_dict[problem_setting]], dtype=int)
                
                param_grid = {
                    'n_estimators': [500, 1000],
                    'max_features': ['sqrt', 'log2'],
                    'max_depth' : [2,4,8,16,32, None],
                    'criterion' :['entropy'],
                    'n_jobs': [-1]
                }
                
                print('X_train: ' + str(X_train.shape))
                print('y_train: ' + str(y_train.shape))
                
                y_test_ps = np.array(y_test[:, label_dict[problem_setting]], dtype=int)
                
                grid_search_verbosity = 1

                # rf
                rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, verbose = grid_search_verbosity, cv = args.inner_cv_loops)
                rf.fit(X_train, y_train)

                best_parameters = rf.best_params_
                predictions_rf = rf.predict_proba(X_test)

                # rf results
                five_sec_eval_auc = roc_auc_score(y_test_ps, predictions_rf[:,1])
                five_sec_eval_aucs_rf[fold] = five_sec_eval_auc
                trained_auc = get_auc_for_full_page(
                    predictions_rf[:,1], X_test, X_test_screen_id, y_test_ps, y_test, N_test_split,
                )
                trained_aucs_rf[fold] = trained_auc
                results_rf['model_name'] = model_name + '_rf'
                results_rf[f'{problem_setting}_auc_fold_{fold}'] = trained_auc
                results_rf[f'{problem_setting}_five_sec_eval_auc_fold_{fold}'] = five_sec_eval_auc
                results_rf[f'{problem_setting}_mean_auc'] = np.mean(list(trained_aucs_rf.values()))
                results_rf[f'{problem_setting}_five_sec_eval_mean_auc'] = np.mean(list(five_sec_eval_aucs_rf.values()))

    results_csv_path = config.CSV_RESULTS_FILE
    if os.path.exists(results_csv_path):
        results_csv = pd.read_csv(results_csv_path)
        results_csv = pd.concat(
            [
                results_csv,
                pd.DataFrame(results_dict, index=[0]),
                pd.DataFrame(results_distance, index=[0]),
                pd.DataFrame(results_rf, index=[0]),
            ], ignore_index=True,
        )
    else:
        results_csv = pd.DataFrame(results_dict, index=[0])
        results_csv = pd.concat(
            [
                results_csv,
                pd.DataFrame(results_distance, index=[0]),
                pd.DataFrame(results_rf, index=[0]),
            ], ignore_index=True,
        )
    if args.encoder_name == 'rf':
        results_csv.to_csv(results_csv_path.replace('.csv','_rf.csv'), index=None)
    else:
        results_csv.to_csv(results_csv_path, index=None)

    return 0


def main() -> int:
    args = get_argument_parser()
    print(' === Configuring GPU ===')

    if args.gpu == -1:
        args.gpu = select_gpu(7700)
    print('~' * 79)
    print(f'{args.gpu=}')
    print('~' * 79)

    configure_gpu(args)
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
    # create dummy augmentatoin
    args.contrastive_augmentation = {'window_size': args.pretrain_window_size, 'channels':args.pretrain_channels, 'name':'random','sd':args.pretrain_sd}
    if args.pretrain_model_path is not None:
        args.pretrained_model_name = args.pretrain_model_path.split('/')[-1]
    else:
        if args.encoder_name == 'clrgaze':
            args.pretrained_model_name = 'CLRGAZE'
        elif args.encoder_name == 'ekyt':
            args.pretrained_model_name = 'EKYT'
        elif args.encoder_name == 'rf':
            args.pretrained_model_name = 'RF'

    print(f'evaluating {args.pretrained_model_name=}'.center(79, '~'))
    four_problem_setting_eval_on_sb(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
