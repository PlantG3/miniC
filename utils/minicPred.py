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


def generate_all_kmers(kmer_length):
    """ Function to generate all possible kmers of given length"""
    base_pairs = ['A', 'T', 'G', 'C']
    # Generate all 4 ^ 7 possible combinations of Kmers and lookup
    all_kmers = {"".join(x): idx + 1 for idx, x in enumerate(list(product(base_pairs, repeat=kmer_length)))}
    return all_kmers

def process_sequence(sequences, sentence_length, kmer_mapping_dict):
    """ Function to map kmers in a given sequence to a unique number"""
    if args.random_length_seq:
        sentence_length = 35
    one_hot_en_text = [([kmer_mapping_dict[x] for x in w.split()]) for w in sequences]

    # Add padding at the end if the sequence is not of given sentence_length
    padded_dataset = pad_sequences(one_hot_en_text, padding='post', maxlen=sentence_length)
    return padded_dataset


def process_file(filename, kmer_length, kmer_count, lines_per_read=100000):
    """
    Function to read a .fq.gz file and extract the sequences present in each line. This function then read a sequences

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



def test_file(data_dir, strain):

    total_strain_seq, total_mini_pred, total_core_pred = 0, 0, 0
    batch_count = 0

    filename = os.path.join(data_dir, strain)

    if args.save_pred_seq:
        save_pred_file = os.path.join(args.output_dir, strain + '.detail.csv')
        p_file = open(save_pred_file, 'w')
        p_file.write('strain,sequence,pred_prob\n')

    for sequences in process_file(filename, kmer_length, kmer_count):
        batch_count += 1
        x_test = process_sequence(sequences, kmer_count, kmers_dict)
        y_pred = model.predict(x_test)

        if args.save_pred_seq:
            for seq, pprob in zip(sequences, y_pred):
                p_file.write(f"{strain},{seq},{pprob[0]}\n")

        pred_labels = (y_pred > args.prediction_threshold).astype('int32')

        total_seq = len(pred_labels)
        mini_pred = sum(pred_labels)
        core_pred = total_seq - mini_pred

        batch_result_str = "making predictions: strain: {} batch: {} total: {} core: {} mini: {} core_per: {} mini_per: {}".format(
            strain, batch_count, total_seq, core_pred, mini_pred, core_pred / total_seq, mini_pred / total_seq)

        logging.info(batch_result_str)
        print(batch_result_str)

        total_strain_seq += total_seq
        total_mini_pred += mini_pred
        total_core_pred += core_pred

        if args.test_on_sample_batch:
            break

    mini_percent = np.round(total_mini_pred / total_strain_seq, 6)
    core_percent = np.round(total_core_pred / total_strain_seq, 6)

    is_sample_batch = ''
    if args.test_on_sample_batch:
        is_sample_batch = '(Using sampled batch)'
    
    # output the final prediction
    save_pred_file = os.path.join(args.output_dir, strain + '.final.csv')
    o_file = open(save_pred_file, 'w')
    o_file.write('data,totalSeq,mini,core,mini_percent,core_percent\n')
    o_file.write('{},{},{},{},{},{}\n'.format(strain, total_strain_seq, total_mini_pred[0], total_core_pred[0], mini_percent[0], core_percent[0]))
    o_file.close()

    if args.save_pred_seq:
        p_file.close()

    logging.info("data,totalSeq,mini,core,mini_percent,core_percent")
    logging.info("{},{},{},{},{},{}\n".format(strain, total_strain_seq, total_mini_pred[0], total_core_pred[0], mini_percent[0], core_percent[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, default=None, help="trained model name with full path")
    parser.add_argument('--output_dir', type=str, required=True, default='', help='path to the output folder')
    parser.add_argument('--data_file', type=str, default='', help="test datafile in beocat with full path")
    parser.add_argument('--kmer_length', type=int, default=7, help='size of kmer')
    parser.add_argument('--kmer_count', type=int, default=15, help='total number of kmers to use in a sequence')
    parser.add_argument('--prediction_threshold', type=float, default=0.95, help='threshold for classification')
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
    os.makedirs(args.output_dir, exist_ok=True)

    ### Vocabulary size
    voc_size = 4 ** kmer_length  # Because we use 7 mers and vocab size of all possible combination of 7 mers

    # All possible kmers lookup dict
    kmers_dict = generate_all_kmers(kmer_length)

    model = models.load_model(args.model_path)
    
    if args.data_file:
        if '/' in args.data_file:
            files_dir, strain = args.data_file.rsplit('/', 1)
        else:
            files_dir = "."
            strain = args.data_file
        logging_file = os.path.join(args.output_dir, strain + '.batch.log')
        logging.basicConfig(filename=logging_file, level=logging.DEBUG, filemode='w')
        # prediction
        test_file(files_dir, strain)
    else:
        print("No data_file was identified")

