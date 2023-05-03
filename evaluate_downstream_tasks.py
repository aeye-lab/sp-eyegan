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


def get_model(args: argparse.Namespace) -> Model:
    print(' === Loading model ===')

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

    if args.pretrained_model_name:
        if args.base_model:
            print('~=== Using Basemodel evaluation~')
        else:
            print(' === Using pretrained  model ===')
            try:
                pretraining_model.load_encoder_weights(args.pretrained_model_name)
            except FileNotFoundError:
                pretraining_model.load_encoder_weights(
                    f"{args.model_dir}/{args.pretrained_model_name.split('.index')[0]}",
                )

    dense_out = Dense(
        1, activation='sigmoid', name='dense_out',
    )(pretraining_model.encoder.get_layer('dense').output)
    classification_model = Model(
                inputs=pretraining_model.encoder.get_layer('velocity_input').input,
                outputs=[dense_out], name='classifier',
    )
    if args.print_model_summary:
        classification_model.summary()
    return classification_model


def four_problem_setting_eval_on_sb(args: argparse.Namespace) -> int:
    print(' === Loading data ===')
    x_data, y_label, label_dict = get_sb_sat_data(args.window_size)
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
        four_problem_setting_eval_on_sb(args)
    elif args.CLRGaze:
        args.pretrained_model_name = None
        four_problem_setting_eval_on_sb(args)
    else:
        args.model_dir = config.CONTRASTIVE_PRETRAINED_MODELS_DIR
        if args.pretrained_model_name:
            if args.pretrained_model_name.startswith('clrgaze'):
                args.encoder_name = 'CLRGaze'
                args.embedding_size = 512
            else:
                args.encoder_name = 'EKYT'
            print(f'evaluating {args.pretrained_model_name=}'.center(79, '~'))
            four_problem_setting_eval_on_sb(args)
        else:
            print(f'looking for all models in batch with idx {args.models_batch}'.center(79, '~'))
            for idx, pretrained_model_name in enumerate(os.listdir(args.model_dir)):
                if pretrained_model_name.endswith('.index'):
                    if idx == args.models_batch:
                        args.pretrained_model_name = pretrained_model_name
                        four_problem_setting_eval_on_sb(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
