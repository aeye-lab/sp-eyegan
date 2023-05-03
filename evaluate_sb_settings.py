from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from load_sb_sat_data import get_sb_sat_data
from Model import contrastive_learner


def help_roc_auc(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return .5
    else:
        return roc_auc_score(y_true, y_pred)


def auroc(y_true, y_pred):
    return tf.py_function(help_roc_auc, (y_true, y_pred), tf.double)


def get_auc(
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
                    # all labels are conditioned on book-page-combination -- hence can take first
                    new_y_test.append(y_test[and_cond][0])
    rocauc = roc_auc_score(new_y_test, mean_x_arr)
    return rocauc


def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--window-size', type=int, default=5000)
    parser.add_argument('--overall-size', type=int, default=5000)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--sd', type=float, default=0.1)
    parser.add_argument('--sd-factor', type=float, default=1.25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--check-point-saver', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--per-process-gpu-memory-fraction', type=float, default=1.)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--pretrained', '-p', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--pretrained-model-name', '-m', type=str)
    parser.add_argument('--split-criterion', '-s', type=str, default='subj')
    parser.add_argument('--print-model-summary', action='store_true')
    parser.add_argument('--normalization', action='store_true')
    parser.add_argument(
        '--problem-setting', type=str, default='native',
        choices=[
            'acc', 'confidence', 'difficulty', 'familiarity', 'recognition', 'interest',
            'pressured', 'sleepiness', 'subj_acc', 'native', 'book', 'subj',
        ],
    )
    parser.add_argument(
        '--model-dir', type=str,
        default='/home/prasse/work/Projekte/AEye/aeye_synthetic_data/classification_model/',
    )
    parser.add_argument(
        '--augmentation-mode', type=str, default='random', choices=['random', 'crop'],
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
    if args.pretrained:
        # pretrained model (example)
        if args.pretrained_model_name:
            print(' === Using pretrained  model ===')
            load_model_name = args.pretrained_model_name
        else:
            load_model_name = f'{args.augmentation_mode}_window_size_{args.window_size}' \
                              f'_sd_{args.sd}_sd_factor_{args.sd_factor}' \
                              f'_embedding_size_{args.embedding_size}' \
                              f'_checkpoint_{args.check_point_saver}'
    else:
        # None if we dont want to use pretraining model
        load_model_name = None

    if args.augmentation_mode == 'crop':
        contrastive_augmentation = {
            'window_size': args.window_size,
            'overall_size': args.overall_size,
            'channels': args.channels,
            'name': 'crop',
        }
    elif args.augmentation_mode == 'random':
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
    )

    if load_model_name:
        pretraining_model.load_encoder_weights(args.model_dir + load_model_name)

    dense_out = Dense(1, activation='sigmoid')(pretraining_model.encoder.get_layer('dense').output)
    classification_model = Model(
                inputs=pretraining_model.encoder.get_layer('velocity_input').input,
                outputs=[dense_out], name='classifier',
    )
    if args.print_model_summary:
        classification_model.summary()
    return classification_model


def evaluate_model_on_sb(args: argparse.Namespace) -> int:
    print(' === Loading data ===')
    x_data, y_label, label_dict = get_sb_sat_data(args.window_size)
    if args.split_criterion == 'subj':
        split_criterion = np.array(sorted(list(set(y_label[:, label_dict[args.split_criterion]]))))
        n_splits = 5
    elif args.split_criterion == 'book':
        n_splits = 4
        ...
    elif args.split_criterion == 'random':
        n_splits = 5
        ...

    problem_setting_mean = np.mean(y_label[:, label_dict[args.problem_setting]])
    y_label[:, label_dict[args.problem_setting]] = np.array(
        y_label[:, label_dict[args.problem_setting]] > problem_setting_mean, dtype=int,
    )
    kfold = KFold(n_splits=n_splits)
    zero_shot_aucs = {}
    trained_aucs = {}
    for fold, (train_idx, test_idx) in enumerate(kfold.split(split_criterion)):
        print(f' {fold=}')
        N_train_split = split_criterion[train_idx]
        N_test_split = split_criterion[test_idx]
        if args.split_criterion == 'subj':
            X_train = x_data[np.isin(y_label[:, label_dict[args.split_criterion]], N_train_split)][:, :, [0, 1]]
            y_train = y_label[np.isin(y_label[:, label_dict[args.split_criterion]], N_train_split)]
            X_test = x_data[np.isin(y_label[:, label_dict[args.split_criterion]], N_test_split)][:, :, [0, 1]]
            X_test_screen_id = x_data[np.isin(y_label[:, label_dict[args.split_criterion]], N_test_split)][:, :, 2]
            y_test = y_label[np.isin(y_label[:, label_dict[args.split_criterion]], N_test_split)]
        elif args.split_criterion == 'book':
            ...
        elif args.split_criterion == 'random':
            ...

        # formatting
        if args.normalization:
            x_scaler = StandardScaler()
            X_train[:, :, 0] = x_scaler.fit_transform(X_train[:, :, 0])
            X_test[:, :, 0] = x_scaler.transform(X_test[:, :, 0])
            y_scaler = StandardScaler()
            X_train[:, :, 1] = y_scaler.fit_transform(X_train[:, :, 1])
            X_test[:, :, 1] = y_scaler.transform(X_test[:, :, 1])

        X_train = tf.cast(X_train, dtype=tf.float32)
        value_not_nan = tf.dtypes.cast(
            tf.math.logical_not(tf.math.is_nan(X_train)), dtype=tf.float32,
        )
        X_train = tf.math.multiply_no_nan(X_train, value_not_nan)
        y_train = np.array(y_train[:, label_dict[args.problem_setting]], dtype=int)
        X_test = tf.cast(X_test, dtype=tf.float32)
        value_not_nan = tf.dtypes.cast(
            tf.math.logical_not(tf.math.is_nan(X_test)), dtype=tf.float32,
        )
        X_test = tf.math.multiply_no_nan(X_test, value_not_nan)
        y_test_ps = np.array(y_test[:, label_dict[args.problem_setting]], dtype=int)

        # train model
        model = get_model(args)

        # zero-shot
        zero_shot_auc = get_auc(model, X_test, X_test_screen_id, y_test_ps, y_test, N_test_split)
        zero_shot_aucs[fold] = zero_shot_auc

        optimizer = Adam(learning_rate=args.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', auroc],
        )
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10),
        ]
        history = model.fit(
            X_train,
            y_train,
            validation_split=.2,
            epochs=100,
            callbacks=callbacks,
            batch_size=args.batch_size,
        )
        trained_auc = get_auc(model, X_test, X_test_screen_id, y_test_ps, y_test, N_test_split)
        trained_aucs[fold] = trained_auc
        print(f'{trained_aucs.values()}')
        print(f'{zero_shot_aucs.values()}')
        print(f'{np.mean(list(zero_shot_aucs.values())):.2f}+/-{np.std(list(zero_shot_aucs.values()))/len(zero_shot_aucs):.2f}')
        print(f'{np.mean(list(trained_aucs.values())):.2f}+/-{np.std(list(trained_aucs.values()))/len(trained_aucs):.2f}')

    breakpoint()
    return 0


def main() -> int:
    args = get_argument_parser()
    print(' === Configuring GPU ===')
    configure_gpu(args)
    evaluate_model_on_sb(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
