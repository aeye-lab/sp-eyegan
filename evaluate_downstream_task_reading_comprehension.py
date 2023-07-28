from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from gpu_selection import select_gpu
from load_sb_sat_data import get_sb_sat_data
from Model import contrastive_learner
import config


def help_roc_auc(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return .5
    else:
        return roc_auc_score(y_true, y_pred)


def auroc(y_true, y_pred):
    return tf.py_function(help_roc_auc, (y_true, y_pred), tf.double)


def get_auc_for_full_page(
        model: Model,
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
                    mean_x_arr.append(np.mean(model.predict(x_arr[and_cond], verbose=False)))
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
    parser.add_argument('--EKYT', action='store_true')
    parser.add_argument('--CLRGaze', action='store_true')
    parser.add_argument('--sd', type=float, default=0.1)
    parser.add_argument('--sd-factor', type=float, default=1.25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--check-point-saver', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=1000)
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
    
    
    parser.add_argument(
        '--problem-setting', type=str, default='native',
        choices=['acc', 'difficulty', 'subj_acc', 'native'],
    )
    args = parser.parse_args()
    return args


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
    x_data, y_label, label_dict = get_sb_sat_data(args.window_size,
                                  verbose = 1,
                                  )
    if args.split_criterion == 'subj':
        split_criterion = np.array(sorted(list(set(y_label[:, label_dict[args.split_criterion]]))))
        n_splits = 5

    problem_settings = ['native', 'difficulty', 'acc', 'subj_acc']
    results_dict = {}
    for problem_setting in problem_settings:
        print(f' Evaluation  on {problem_setting=} '.center(79, '='))
        problem_setting_mean = np.mean(y_label[:, label_dict[problem_setting]])
        y_label[:, label_dict[problem_setting]] = np.array(
            y_label[:, label_dict[problem_setting]] > problem_setting_mean, dtype=int,
        )
        kfold = KFold(n_splits=n_splits)
        five_sec_eval_aucs = {}
        trained_aucs = {}
        np.random.seed(12)
        random.seed(12)
        tf.random.set_seed(12)
        if args.pretrained_model_name:
            model_name = args.pretrained_model_name.split('/')[-1]
        elif args.EKYT:
            model_name = 'EKYT'
        elif args.CLRGaze:
            model_name = 'CLRGaze'
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
                X_train = x_data[
                    np.isin
                    (
                        y_label[:, label_dict[args.split_criterion]],
                        N_train_split,
                    ),
                ][:, :, [0, 1]]
                y_train = y_label[
                    np.isin
                    (
                        y_label[:, label_dict[args.split_criterion]],
                        N_train_split,
                    ),
                ]
                X_test = x_data[
                    np.isin
                    (
                        y_label[:, label_dict[args.split_criterion]],
                        N_test_split,
                    ),
                ][:, :, [0, 1]]
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

            # zero-shot

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
            print(np.unique(model.predict(X_test), return_counts=True))
            five_sec_eval_auc = roc_auc_score(y_test_ps, model.predict(X_test))
            five_sec_eval_aucs[fold] = five_sec_eval_auc
            trained_auc = get_auc_for_full_page(
                model, X_test, X_test_screen_id, y_test_ps, y_test, N_test_split,
            )
            trained_aucs[fold] = trained_auc
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
            results_dict['model_name'] = model_name
            results_dict[f'{problem_setting}_auc_fold_{fold}'] = trained_auc
            results_dict[f'{problem_setting}_five_sec_eval_auc_fold_{fold}'] = five_sec_eval_auc
            results_dict[f'{problem_setting}_mean_auc'] = np.mean(list(trained_aucs.values()))
            results_dict[f'{problem_setting}_five_sec_eval_mean_auc'] = np.mean(list(five_sec_eval_aucs.values()))
            model.save_weights(
                config.TRAINED_CLASSIFICATION_MODELS_DIR,
            )

    results_csv_path = config.CSV_RESULTS_FILE
    results_csv = pd.read_csv(results_csv_path)
    results_csv = pd.concat(
        [
            results_csv,
            pd.DataFrame(results_dict, index=[0]),
        ], ignore_index=True,
    )
    results_csv.to_csv(results_csv_path, index=None)

    return 0


def main() -> int:
    args = get_argument_parser()  
    print(' === Configuring GPU ===')
    
    args.gpu = select_gpu(7700)
    print('~' * 79)
    print(f'{args.gpu=}')
    print('~' * 79)
    
    configure_gpu(args)
    if args.EKYT:
        args.pretrained_model_name = None
        args.pretrain_model_path = None
        four_problem_setting_eval_on_sb(args)
    elif args.CLRGaze:
        args.pretrained_model_name = None
        args.pretrain_model_path = None
        four_problem_setting_eval_on_sb(args)
    else:
        if args.pretrain_augmentation_mode is None:
            args.pretrain_model_path = None
            if args.encoder_name == 'ekyt':
                args.pretrained_model_name = 'ekyt_scratch'
            elif args.encoder_name == 'clrgaze':
                args.pretrained_model_name = 'cleargaze_scratch'
            print(f'evaluating {args.encoder_name=}'.center(79, '~'))
            four_problem_setting_eval_on_sb(args)            
        else:
            if args.pretrain_encoder_name == 'clrgaze':
                args.embedding_size = 512
            elif args.pretrain_encoder_name == 'ekyt':
                args.embedding_size = 128

            if args.pretrain_augmentation_mode == 'crop':
                args.contrastive_augmentation = {'window_size': args.pretrain_window_size, 'overall_size': args.overall_size,'channels':args.pretrain_channels, 'name':'crop'}
                args.pretrain_model_path = args.pretrain_model_dir + args.pretrain_encoder_name + '_' + args.pretrain_augmentation_mode + '_window_size_' + str(args.pretrain_window_size) +\
                                    '_overall_size_' + str(args.overall_size) +\
                                    '_embedding_size_' + str(args.embedding_size) + '_stimulus_' + str(args.pretrain_stimulus) +\
                                    '_model_' + str(args.pretrain_scanpath_model) + '_' + str(args.pretrain_num_pretrain_instances)
                args.per_process_gpu_memory_fraction = 1.
            elif args.pretrain_augmentation_mode == 'random':
                args.contrastive_augmentation = {'window_size': args.pretrain_window_size, 'channels':args.pretrain_channels, 'name':'random','sd':args.pretrain_sd}
                args.pretrain_model_path = args.pretrain_model_dir + args.pretrain_encoder_name + '_' + args.pretrain_augmentation_mode + '_window_size_' + str(args.pretrain_window_size) +\
                                    '_sd_' + str(args.pretrain_sd) + '_sd_factor_' + str(args.pretrain_sd_factor) +\
                                    '_embedding_size_' + str(args.embedding_size) + '_stimulus_' + str(args.pretrain_stimulus) +\
                                    '_model_' + str(args.pretrain_scanpath_model) + '_' + str(args.pretrain_num_pretrain_instances)
                args.per_process_gpu_memory_fraction = 1.
            elif args.pretrain_augmentation_mode == 'rotation':
                args.contrastive_augmentation = {'window_size': args.pretrain_window_size, 'channels':args.pretrain_channels, 'name':'rotation','max_rotation':args.pretrain_max_rotation}
                args.pretrain_model_path = args.pretrain_model_dir + args.pretrain_encoder_name + '_' + args.pretrain_augmentation_mode + '_window_size_' + str(args.pretrain_window_size) +\
                                    '_max_rotation_' + str(args.pretrain_max_rotation) +\
                                    '_embedding_size_' + str(args.embedding_size) + '_stimulus_' + str(args.pretrain_stimulus) +\
                                    '_model_' + str(args.pretrain_scanpath_model) + '_' + str(args.pretrain_num_pretrain_instances)
            args.pretrain_model_path = args.pretrain_model_path + args.pretrain_data_suffix
            args.pretrained_model_name = args.pretrain_model_path.split('/')[-1]
            print(f'evaluating {args.pretrained_model_name=}'.center(79, '~'))
            four_problem_setting_eval_on_sb(args)
            
            
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
