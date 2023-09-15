#!/usr/bin/env python
# coding: utf-8

### Mini Prediction Using Bidirectional LSTM
import os
import argparse

parser = argparse.ArgumentParser(description='Input required arguments.')
parser.add_argument('--train_csv', required=True, type=str, default=None,
                    help='path to the training dataset file eg: /home/user/mini_prediction/train.csv')
parser.add_argument('--valid_csv', type=str, default=None,
                    help='path to the validation dataset file eg: /home/user/mini_prediction/valid.csv')
parser.add_argument('--test_csv', required=True, type=str, default=None,
                    help='path to the test dataset file eg: /home/user/mini_prediction/train.csv')
parser.add_argument('--output_dir', type=str, default='',
                    help='path to the results and log output folder. \
                    eg: /home/user/mini_prediction/main_ng/bi_lstm/outputs/Run1')
parser.add_argument('--kmer_length', type=int, default=9,
                    help='size of kmer')
parser.add_argument('--kmer_count', type=int, default=11,
                    help='total number of kmers in a sequence')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size for training')
parser.add_argument('--patience', type=int, default=15, help='Number of epochs to wait before early stopping')
parser.add_argument('--do_test', action='store_true', help='whether to evaluate on test dataset or not')
parser.add_argument('--gpu', type=int, default=0, help='gpu id to use')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import time
from datetime import date
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import logging

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class TrainingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        log_str = ""
        for k, v in logs.items():
            log_str = log_str + str(k) + ': ' + str(v) + ', '
        log_str = "epoch: " + str(epoch) + ', ' + log_str
        logging.info(log_str)


def get_model(voc_size, embedding_vector_features, sentence_length):

    model = Sequential()
    model.add(Embedding(voc_size + 1, embedding_vector_features, input_length=sentence_length))
    model.add(Bidirectional(LSTM(embedding_vector_features, return_sequences=True)))
    model.add(Bidirectional(LSTM(embedding_vector_features)))
    model.add(Dense(1, activation='sigmoid'))

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])
    model.summary()
    return model


def generate_all_kmers(kmer_length):
    """ Function to generate all possible kmers of given length"""
    base_pairs = ['A', 'T', 'G', 'C']

    # Generate all 4 ^ 7 possible combinations of Kmers and lookup
    all_kmers = {"".join(x): idx + 1 for idx, x in enumerate(list(product(base_pairs, repeat=kmer_length)))}
    return all_kmers


def process_sequence(df, sentence_length, kmer_mapping_dict):
    """ Function to map kmers in a given sequence to a unique number"""

    # Encode kmers to dense one hot encoding
    one_hot_en_text = [([kmer_mapping_dict[x] for x in w.split()]) for w in list(df['sequence'])]

    # Add padding at the end if the sequence is not of given sentence_length
    padded_dataset = pad_sequences(one_hot_en_text, padding='post', maxlen=sentence_length)
    return padded_dataset


def get_train_valid_df(train_file):
    df_train = pd.read_csv(train_file, header=None, usecols=[0, 1, 2], names=['sequence', 'strain', 'label'])
    df_train.dropna()

    # Remove duplicates. It deletes all instances of duplicate rows. This is because some duplicate rows belong to
    # core and some belong to mini
    df_train = df_train.drop_duplicates(subset=['sequence'], keep=False)

    # Filter core and mini and remove duplicates
    core = df_train[df_train.label == 0].sample(frac=1)
    mini = df_train[df_train.label == 1].sample(frac=1)

    logging.info("Before handling class unbalance")
    logging.info("Core: {}".format(len(core)))
    logging.info("Mini: {}".format(len(mini)))
    logging.info("Ratio of core:mini = {}".format(len(core) / len(mini)))

    # To balance the size of mini and core
    # if len(core) > len(mini):
    #     mini = mini.sample(n=len(core))
    # else:
    #     core = core.sample(n=len(mini))

    df = pd.concat([core, mini], axis=0)

    # To shuffle the core and mini rows
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)

    valid_df_size = int(0.05 * len(df))
    df_valid, df_train = df.iloc[:valid_df_size].copy(), df.iloc[valid_df_size:].copy()

    # if args.retrain_model:
    #     if not args.retrain_dataset_file:
    #         print("Error! retrain_model is true but retraining dataset file is not given... exiting...")
    #         logging.error("Error! retrain_model is true but retraining dataset file is not given... exiting...")
    #         exit()
    #
    #     df_retrain = pd.read_csv(args.retrain_dataset_file, header=None, usecols=[0, 1, 2],
    #                              names=['sequence', 'strain', 'label'])
    #
    #     logging.info("new mini training dataset size: {}".format(len(df_retrain)))
    #     df_train = pd.concat([df_train, df_retrain], ignore_index=True)
    #     df_train.reset_index(drop=True, inplace=True)

    logging.info("Validation size: {}, train size: {}".format(len(df_valid), len(df_train)))
    df_valid.reset_index(drop=True, inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    return df_train, df_valid


def get_valid_dataset(df_valid):
    y_valid = df_valid['label']
    padded_valid = process_sequence(df_valid, kmer_count, kmers_dict)

    x_valid = np.array(padded_valid)
    y_valid = np.array(y_valid)

    return x_valid, y_valid


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, batch_size=32, num_classes=None, shuffle=True):
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X = self.df.iloc[batch]
        y = np.array(X['label'])
        padded_valid = process_sequence(X, kmer_count, kmers_dict)
        return np.array(padded_valid), y


if __name__ == "__main__":
    
    output_dir = args.output_dir
    kmer_count = args.kmer_count
    kmer_length = args.kmer_length
    voc_size = 4 ** kmer_length

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    # Path to save model during training
    model_save_path = os.path.join(output_dir, 'saved_model')

    os.makedirs(model_save_path, exist_ok=True)

    # Log file to save logs
    logging.basicConfig(filename=os.path.join(output_dir,
                                              'training_logs_{}.log'.format(date.today().strftime("%b_%d_%Y"))),
                        level=logging.DEBUG, filemode='w')

    # Tensorboard log directory
    tf_log_dir = os.path.join(output_dir, 'tboard_logs')
    os.makedirs(tf_log_dir, exist_ok=True)

    print("loading dataset...")
    # dataset_file = args.dataset_file
    # logging.info('dataset used: {}'.format(dataset_file))
    # Prepare dataset

    # All possible kmers lookup dict
    kmers_dict = generate_all_kmers(kmer_length)

    # df_train, df_valid = get_train_valid_df(dataset_file)
    df_train, df_valid = pd.read_csv(args.train_csv), pd.read_csv(args.valid_csv)
    X_valid, y_valid = get_valid_dataset(df_valid)

    train_generator = DataGenerator(df_train, batch_size=args.batch_size)

    # Creating model
    embedding_vector_features = 128

    print("creating model...")
    model = get_model(voc_size, embedding_vector_features, kmer_count)

    ## Finally Training
    print("training model...")
    logging.info('training model: batch_size: {}, early stopping on val_loss with patience {}'.format(args.batch_size, args.patience))
    epochs = args.epochs

    my_callback = TrainingCallback()
    tf_callback = tf.keras.callbacks.TensorBoard(log_dir=tf_log_dir)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True,
                                                      verbose=1)

    history = model.fit(x=train_generator, validation_data=(X_valid, y_valid), epochs=epochs,
                        callbacks=[my_callback, tf_callback, early_stopping],
                        validation_batch_size=args.batch_size)

    end_time = time.time()
    time_taken = "finished training model in: {:.4f} hrs".format((end_time - start_time) / 3600)
    logging.info(time_taken)
    print(time_taken)

    if args.do_test:
        if args.test_csv:
            logging.info("Results on test dataset: ")
            print("Testing on test set...")
            x_test, y_test = get_valid_dataset(pd.read_csv(args.test_csv))
            results = model.evaluate(x_test, y_test, batch_size=10240)
            print("Results on test dataset (Loss, Accu, Prec, Recall) :", results)
            logging.info(f"Results on test dataset (Loss, Accu, Prec, Recall) : {results}")
        else:
            print("--do_test is true but --test_csv file path is missing...")

    print('saving final model...')
    model.save(os.path.join(model_save_path, 'model_final.h5'))
    logging.info('final trained model saved to: '.format(os.path.join(model_save_path, 'model_final.h5')))



