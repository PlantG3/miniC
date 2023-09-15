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

import os
import tqdm
import gzip
import argparse
import logging

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from itertools import product, islice
from tensorflow.keras import models


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_all_kmers(kmer_length):
    """ Function to generate all possible kmers of given length"""
    base_pairs = ['A', 'T', 'G', 'C']

    # Generate all 4 ^ 7 possible combinations of Kmers and lookup
    all_kmers = {"".join(x): idx + 1 for idx, x in enumerate(list(product(base_pairs, repeat=kmer_length)))}
    return all_kmers


def get_ftp_client():
    import paramiko
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname='headnode.beocat.ksu.edu', username='gnikesh')
    ftp_client = ssh_client.open_sftp()
    return ftp_client


def process_sequence(sequences, sentence_length, kmer_mapping_dict):
    """ Function to map kmers in a given sequence to a unique number"""
    if args.random_length_seq:
        sentence_length = 35

    # Encode kmers to dense one hot encoding
    one_hot_en_text = [([kmer_mapping_dict[x] for x in w.split()]) for w in sequences]

    # Add padding at the end if the sequence is not of given sentence_length
    padded_dataset = pad_sequences(one_hot_en_text, padding='post', maxlen=sentence_length)
    return padded_dataset


def process_file(filename, kmer_length, kmer_count, lines_per_read=400000):
    """
    Function to read a .fq.gz file and extract the sequences present in each line. This function then read a sequences
    processed from .fq.gz file as a list and prepares test dataset based on the kmer length, and length of sequence
    required by the classifier model.

    It skips the sequences whose length is less than required to make a test data of required kmer size and sequence
    length. If the sequence length is too long, it takes the first <seq_length> kmers to create the required sentence
    length and ignores the rest of the sequences

    :param filename: .fq.gz file to read. Required the complete path to the filename
    :param kmer_length: length of kmer to break the sequences into
    :param kmer_count: length of the sequence of kmer.
    :return: a generator with the kmer sequences of required length
    """

    print("processing file: {}".format(filename))
    with gzip.open(filename, 'rb') as f_read:
        for n_lines in iter(lambda: tuple(islice(f_read, lines_per_read)), ()):
            sequences = set()
            for line in n_lines:
                line = line.decode('utf-8').strip()

                if line.startswith('@') or line.startswith('+'):
                    continue

                # Removing lines that has other characeters except ATGC
                if set([x.lower() for x in line]) != {'a', 't', 'g', 'c'}:
                    continue

                splitted_line = [line[i: i + kmer_length] for i in range(0, len(line), kmer_length)]
                splitted_line = [x for x in splitted_line if len(x) == kmer_length]

                if args.random_length_seq:
                    # sentence_length = 35
                    seq = ' '.join(splitted_line)

                else:
                    seq = splitted_line[:kmer_count]
                    seq = ' '.join(seq)

                sequences.add(seq)
    
            yield sequences


def test_file(remote_files_dir, strain):

    if args.save_pred_seq:
        save_pred_file = os.path.join(args.output_dir, strain + '.csv')
        p_file = open(save_pred_file, 'w')
        p_file.write('sequence,strain,pred,pred_proba\n')

    if args.remote_exec:
        ftp_client = get_ftp_client()
        print("downloading file: {} from remote host".format(strain))
        strain_full_path = os.path.join(remote_files_dir, strain)
        ftp_client.get(strain_full_path, os.path.join(args.base_dir, 'cache', strain))
        ftp_client.close()

    total_strain_seq, total_mini_pred, total_core_pred = 0, 0, 0
    batch_count = 0

    if args.remote_exec:
        filename = os.path.join(args.base_dir, 'cache', strain)
    else:
        filename = os.path.join(remote_files_dir, strain)

    for sequences in process_file(filename, kmer_length, kmer_count):
        batch_count += 1
        x_test = process_sequence(sequences, kmer_count, kmers_dict)
        y_pred = model.predict(x_test)
        pred_labels = (y_pred > args.prediction_threshold).astype('int32')

        total_seq = len(pred_labels)
        mini_pred = sum(pred_labels)
        core_pred = total_seq - mini_pred

        batch_result_str = "making predictions: strain: {} batch: {} total: {} core: {} mini: {} core_per: {} mini_per: {}".format(
            strain,
            batch_count, total_seq, core_pred, mini_pred, core_pred / total_seq, mini_pred / total_seq)

        logging.info(batch_result_str)
        print(batch_result_str)

        total_strain_seq += total_seq
        total_mini_pred += mini_pred
        total_core_pred += core_pred

        if args.save_pred_seq:
            for seq, pred, pred_proba in zip(sequences, pred_labels, y_pred):
                # sequence, strain, pred, pred_proba
                p_file.write('{},{},{},{}\n'.format(seq, strain, pred[0], pred_proba[0]))

        if args.test_on_sample_batch:
            break

    mini_percent = np.round(total_mini_pred / total_strain_seq, 6)
    core_percent = np.round(total_core_pred / total_strain_seq, 6)

    is_sample_batch = ''
    if args.test_on_sample_batch:
        is_sample_batch = '(Using sampled batch)'
    result_str = 'Final results:{}\n\
    Strain: {}, Total Seq: {}, mini: {}, core: {}, mini_percent: {}, core_percent: {}, core:mini: {}, prediction_threshold: {}\n'.format(
        is_sample_batch, strain, total_strain_seq, total_mini_pred, total_core_pred, mini_percent, core_percent,
        total_core_pred / total_mini_pred, args.prediction_threshold
    )

    print(result_str)
    logging.info(result_str)

    if args.remote_exec:
        if os.path.exists(filename):
            os.remove(filename)

    if args.save_pred_seq:
        p_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir', type=str, default='/data/gnikesh/mini_prediction',
    #                     help="the path to the project directory")
    parser.add_argument('--model_path', type=str, required=True, default=None, help="trained model name with full path")
    parser.add_argument('--output_dir', type=str, required=True, default='', help='path to the output folder')
    parser.add_argument('--data_file', type=str, default='', help="test datafile in beocat with full path")
    parser.add_argument('--kmer_length', type=int, default=7, help='size of kmer')
    parser.add_argument('--kmer_count', type=int, default=15, help='total number of kmers to use in a sequence')
    parser.add_argument('--prediction_threshold', type=float, default=0.95, help='threshold for classification')

    parser.add_argument('--remote_exec', action="store_true",
                        help="whether to run the test remotely from host other than beocat")
    parser.add_argument('--save_pred_seq', action="store_true",
                        help="whether to save all the predicted sequences and their prediction probabilities")
    parser.add_argument('--random_length_seq', action="store_true",
                        help="Whether to create dataset with random sequence lengths for testing")
    parser.add_argument('--test_on_sample_batch', action="store_true",
                        help="whether to test only on randomly sampled 100k fq sequences")
    parser.add_argument('--no_mini', action="store_true", help="whether strain contains mini or not")

    args = parser.parse_args()

    kmer_length = args.kmer_length
    kmer_count = args.kmer_count

    single_test_file = args.data_file
    os.makedirs(args.output_dir, exist_ok=True)
    files_dir = '/bulk/liu3zhen/research/projects/mini_prediction/data/Mo02_trimPE'

    if args.data_file:

        files_dir, strain = args.data_file.rsplit('/', 1)
        logging_file = os.path.join(args.output_dir, strain + '_results.log')
    else:
        logging_file = os.path.join(args.output_dir, 'all_strain_results.log')

    print("logging results to file: {}".format(logging_file))

    logging.basicConfig(filename=logging_file, level=logging.DEBUG, filemode='w')

    ### Vocabulary size
    voc_size = 4 ** kmer_length  # Because we use 7 mers and vocab size of all possible combination of 7 mers

    # All possible kmers lookup dict
    kmers_dict = generate_all_kmers(kmer_length)

    model = models.load_model(args.model_path)

    # data_path = "/bulk/liu3zhen/research/projects/mini_prediction/data/Mo02_trimPE"
    if not args.data_file:
        if args.remote_exec:
            import subprocess
            files = subprocess.Popen(f"ssh gnikesh@headnode.beocat.ksu.edu ls /bulk/liu3zhen/research/projects/mini_prediction/data/Mo02_trimPE",
                                 shell=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE).communicate()
            files = files[0].decode('utf-8').strip().split('\n')
        else:
            files = os.listdir(files_dir)

        for strain in files:
            if not strain.endswith('fq.gz'):
                continue

            test_file(files_dir, strain)
    else:
        test_file(files_dir, strain)


