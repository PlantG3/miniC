import os
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Input required arguments.')
parser.add_argument('--saved_model', type=str, required=True, help="trained model name with full path")
parser.add_argument('--test_csv', type=str, default='data/sample_data/test_set_sample.csv',
                    help='path to the test dataset file eg: /home/user/mini_prediction/train.csv')

parser.add_argument('--output_dir', type=str, default='',
                    help='path to the results and log output folder. \
                    eg: /home/user/results')
parser.add_argument('--save_pred_seq', action="store_true")
parser.add_argument('--prediction_threshold', type=float, default=0.98, help="threshold for classifying the prediction to positive class label")
parser.add_argument('--batch_size', type=int, default=2048, help="size of a batch to use for processing large dataset while making the prediction")
parser.add_argument('--kmer_length', type=int, default=9,
                    help='size of kmer')
parser.add_argument('--kmer_count', type=int, default=11,
                    help='total number of kmers in a sequence')
parser.add_argument('--gpu', type=int, default=0, help='gpu id to use')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

from itertools import product


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



def get_test_dataset(df_test):
    y_valid = df_test['label']
    padded_valid = process_sequence(df_test, kmer_count, kmers_dict)

    x_valid = np.array(padded_valid)
    y_valid = np.array(y_valid)

    return x_valid, y_valid


def test_model(model, x_test, y_test, args):  
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_pred_seq:
        seq_df = pd.read_csv(args.test_csv)
        results = []
        for i in range(0, len(seq_df), args.batch_size):
            df = seq_df.iloc[i: i + args.batch_size]
            padded = process_sequence(df, kmer_count, kmers_dict)
            preds = model.predict(padded)
            preds_label = [1 if p >= args.prediction_threshold else 0 for p in preds]
            results += list(zip(df['sequence'], df['strain'], df['label'], preds_label, [x[0] for x in preds]))

        result_df = pd.DataFrame(results, columns=['sequence', 'strain', 'label', 'predicted_label', 'prediction_score'])
        result_df.to_csv(os.path.join(args.output_dir, 'prediction_scores.csv'), index=False)
        print(f"prediction scores saved to: {os.path.join(args.output_dir, 'prediction_scores.csv')}")

    results = model.evaluate(x_test, y_test, batch_size=args.batch_size)
    acc, pr, r = results[1], results[2], results[3]
    print(f"Accuracy: {acc:.6f} Precision: {pr:.6f} Recall: {r:.6f}")

    with open(os.path.join(args.output_dir, 'prediction_results.txt'), 'w') as fwrite:
        fwrite.write(f'Total sequences: {len(x_test)}\n')
        fwrite.write(f'-' * 25)
        fwrite.write('\n')
        fwrite.write(f'Accuracy  : {acc:.6f}\n')
        fwrite.write(f'Precision : {pr:.6f}\n')
        fwrite.write(f'Recall    : {r:.6f}\n')
        fwrite.write(f'-' * 25)

    print(f"Final results written to file: {os.path.join(args.output_dir, 'prediction_results.txt')}")


if __name__ == "__main__":
    kmer_length = args.kmer_length
    kmer_count = args.kmer_count
    kmers_dict = generate_all_kmers(kmer_length)

    if not args.test_csv:
        print("No test file given... exiting...")
        exit()

    model = models.load_model(args.saved_model)
    x_test, y_test = get_test_dataset(pd.read_csv(args.test_csv))
    results = test_model(model, x_test, y_test, args)
    print('done...')

