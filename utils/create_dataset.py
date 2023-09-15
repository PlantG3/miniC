import os
import argparse
import pandas as pd
import multiprocessing
import subprocess
import tqdm

# base directory where the code is
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()

# Strains that have both core and mini fasta files: eg <strain>.core.fasta and <strain>.mini.fasta
STRAINS_FQ_NO_NEWLINE = ["B71", "LpKY", "O_O135-1", "TF05-1"]

# Strain that do not have mini: eg: <strain>.fasta or <strain>.fa
STRAINS_FQ_NEWLINE = ["70_15", "E_MZ5-1-6"]

def kmer2seq(kmers):
    """
    Convert kmers to original sequence
    Arguments:
    kmers -- str, kmers separated by space.
    Returns:
    seq -- str, original sequence.
    """
    kmers_list = kmers.split(" ")
    bases = [kmer[0] for kmer in kmers_list[0:-1]]
    bases.append(kmers_list[-1])
    seq = "".join(bases)
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq


def fasta2seq(base_dir, dataset_folder):
    """
    Converts sequences from FASTA file to a text file separating each chromosome. Only converts 'core'
    and 'mini' chromosomes from the FASTA file.

    Writes the processed genome sequences to the folder: <base_dir>/data/genome_sequence/

    Please add the new strains to "strains" or "other_strains" list if there are new strains.
        -> Add to "strains" list if the fasta file has whole chromosome as a single line.
        -> Add to "other_strains" list if the fast file has chromosome sequences broken in new lines

    :param base_dir: Absolute path to the mini_prediction project: eg: '/homes/user1/mini_prediction'
    :param dataset_folder: Absolute path to the data folder: eg: '/bulk/liu3zhen/research/projects/mini_prediction/data'
    :return: None
    """

    def write_file(ip_dir, op_dir):
        with open(ip_dir, 'r') as fp:
            lines = fp.readlines()
            lines = [x.strip() for x in lines]

            for idx in range(len(lines)):
                if '>' in lines[idx]:
                    with open(os.path.join(op_dir, lines[idx].replace('>', '')), 'w') as f_write:
                        f_write.write(lines[idx + 1].upper())

    for strain in os.listdir(dataset_folder):
        if strain not in STRAINS_FQ_NO_NEWLINE:
            continue
        print("processing strain... ", strain)
        genome_sequence_dir = os.path.join(base_dir, 'data', 'genome_sequence', strain)
        os.makedirs(genome_sequence_dir, exist_ok=True)
        for genome in os.listdir(os.path.join(dataset_folder, strain, '0_genome')):
            if genome.endswith('core.fasta'):
                write_file(os.path.join(dataset_folder, strain, '0_genome', genome), genome_sequence_dir)

            if genome.endswith('mini.fasta'):
                write_file(os.path.join(dataset_folder, strain, '0_genome', genome), genome_sequence_dir)

    for strain in os.listdir(dataset_folder):
        if strain not in STRAINS_FQ_NEWLINE:
            continue
        print("processing strain...", strain)
        genome_sequence_dir = os.path.join(base_dir, 'data', 'genome_sequence', strain)
        os.makedirs(genome_sequence_dir, exist_ok=True)
        for file in os.listdir(os.path.join(dataset_folder, strain, '0_genome')):
            with open(os.path.join(dataset_folder, strain, '0_genome', file), 'r') as fp:
                lines = fp.readlines()
                lines = [x.strip() for x in lines]
                chrms = []
                new_chrms = ''
                for l in lines:
                    if '>' in l:
                        if new_chrms:
                            chrms.append(new_chrms)
                        new_chrms = ''

                    if set([x.lower() for x in l]) != {'a', 't', 'g', 'c'}:
                        continue

                    new_chrms += l
                chrms.append(new_chrms)

                for idx, l in enumerate(chrms):
                    with open(os.path.join(base_dir, 'data', 'genome_sequence', strain, 'chr' + str(idx + 1)),
                              'w') as fwrite:
                        fwrite.write(l.upper())


def write_kmers2seq(base_dir, strain='TF05-1'):
    kmer_dir = os.path.join(base_dir, 'cache', strain)

    save_dir = os.path.join(base_dir, 'data', 'genome_sequence', strain)
    os.makedirs(save_dir, exist_ok=True)

    for f in os.listdir(kmer_dir):

        if not strain in f:
            continue
        print("processing file: ", f)
        with open(os.path.join(kmer_dir, f), 'r') as fp:
            kmers = fp.readlines()[0].strip()
            seq = kmer2seq(kmers)

        op_file = f.replace('_15mer', '')
        with open(os.path.join(save_dir, op_file), 'w') as fp:
            fp.write(seq)


def create_kmer_sequence(base_dir, kmer_length, kmer_count, mini_stride, core_stride):
    total_sequence_length = kmer_length * kmer_count
    sequences = os.path.join(base_dir, 'data/genome_sequence')

    save_path = os.path.join(base_dir, 'cache/kmer_sequences/{}mer-{}seq'.format(kmer_length, kmer_count))

    os.makedirs(save_path, exist_ok=True)
    all_strains = STRAINS_FQ_NEWLINE + STRAINS_FQ_NO_NEWLINE
    for strain in all_strains:
        files = os.listdir(os.path.join(sequences, strain))

        for f in files:
            f_name = f.replace('_15mer', '') + '{}_{}mer'.format(strain, kmer_length)
            rd_file = os.path.join(sequences, strain, f)
            op_file = os.path.join(save_path, f_name)

            if 'mt' in f.lower():
                continue

            with open(rd_file, 'r') as f_read:
                line = f_read.readlines()[0].strip()
                line = line.replace('N', '')
                rev_line = line[::-1]
                u_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
                rev_comp = "".join([u_dict[x] for x in rev_line])

                if f.lower().startswith('mini'):
                    split_line = [line[i:i + total_sequence_length] for i in range(0, len(line), mini_stride)]
                    split_line = [x for x in split_line if len(x) == total_sequence_length]

                    split_rev_comp = [rev_comp[i:i + total_sequence_length] for i in
                                      range(0, len(rev_comp), mini_stride)]
                    split_rev_comp = [x for x in split_rev_comp if len(x) == total_sequence_length]

                if f.lower().startswith('chr'):
                    split_line = [line[i:i + total_sequence_length] for i in range(0, len(line), core_stride)]
                    split_line = [x for x in split_line if len(x) == total_sequence_length]

                    split_rev_comp = [rev_comp[i:i + total_sequence_length] for i in
                                      range(0, len(rev_comp), core_stride)]
                    split_rev_comp = [x for x in split_rev_comp if len(x) == total_sequence_length]

                seq = ' '.join(split_line)
                rev_comp_seq = ' '.join(split_rev_comp)
                with open(op_file, 'w') as f_write:
                    print("writing file: {}".format(f))
                    f_write.write(seq + '\n')
                    f_write.write(rev_comp_seq)


def get_core_stride_required_to_balance_dataset(kmer_length, kmer_count, mini_stride):
    sequences = os.path.join(BASE_DIR, 'data/genome_sequence')
    total_seq_len = kmer_length * kmer_count

    # Kmer formula = |(L - k) / n| + 1 where L = seq_length, k = kmer_len, n = stride_len
    core_counts, mini_counts = [], []
    all_strains = STRAINS_FQ_NEWLINE + STRAINS_FQ_NO_NEWLINE
    for strain in all_strains:
        # if not 'B71' in strain:
        #     continue


        files = os.listdir(os.path.join(sequences, strain))
        for f in files:

            rd_file = os.path.join(sequences, strain, f)
            with open(rd_file, 'r') as f_read:
                line = f_read.readlines()[0].strip()
                line = line.replace('N', '')
                
                if 'chr' in f.lower():
                    core_counts.append(len(line))

                if 'mini' in f.lower():
                    mini_kmers = set([line[i:i + total_seq_len] for i in range(0, len(line), mini_stride)])
                    mini_counts.append(len(mini_kmers))
    
    
    # total_mini = sum([int((seq_len - kmer_length) / mini_stride) + 1 for seq_len in mini_counts])
    total_mini = sum(mini_counts)
    core_stride = int(round((sum(core_counts) - kmer_length) / (total_mini - 1), 0))

    # print(core_counts)
    # print(mini_counts)
    # total_core = int((sum(core_counts) - kmer_length) / core_stride) + 1
    # print(f"total mini: {total_mini} total core: {total_core} core_stride: {core_stride} kmer_len: {kmer_length}")
    # print(f"diff: core - mini: {total_core - total_mini} x2: {(total_core - total_mini) * 2}")

    # total_core_after_stride = sum([int((seq_len - kmer_length) / core_stride) + 1 for seq_len in core_counts])
    # print(total_core_after_stride)
    return core_stride
                    
                

def preprocess_fasta_for_filtering(kmer_length, kmer_count, mini_stride, core_stride=None):
    """
    This function preprocesses the genome sequence by creating fasta file for filtering common reads from core and mini
    and also filter mt reads.

    :param kmer_length: length of kmer
    :param kmer_count: length of kmer words to make a sequence of
    :param mini_stride: length of mini steps size to take while creating sequence from whole genome
    :param core_stride: length of core steps size to take while creating sequence from whole genome
    :return: None
    """
    if not core_stride:
        core_stride = get_core_stride_required_to_balance_dataset(kmer_length, kmer_count, mini_stride)

    total_sequence_length = kmer_length * kmer_count
    sequences = os.path.join(BASE_DIR, 'data/genome_sequence')

    save_path = os.path.join(BASE_DIR, 'cache/kmer_sequences/{}mer-{}seq_fasta'.format(kmer_length, kmer_count))

    os.makedirs(save_path, exist_ok=True)

    print(
        "splitting whole genome sequences to {} sequence length and converting it to fasta file for preprocessing".format(
            total_sequence_length
        ))
    print("mini stride: {} core stride: {} kmer_length: {} sequence length: {}".format(mini_stride, core_stride,
                                                                                       kmer_length, kmer_count))
    mini_count, core_count = 0, 0
    all_strains = STRAINS_FQ_NEWLINE + STRAINS_FQ_NO_NEWLINE

    for strain in all_strains:
        files = os.listdir(os.path.join(sequences, strain))
        output_fasta_file = os.path.join(save_path, strain + '.fasta')
        f_write = open(output_fasta_file, 'w')
        files.sort()
        print("Processing strain: {}".format(strain))
        for f in tqdm.tqdm(files):
            rd_file = os.path.join(sequences, strain, f)
            # print("writing to fasta file: {} {}".format(strain, f))
            if 'mt' in f.lower():
                continue

            with open(rd_file, 'r') as f_read:
                line = f_read.readlines()[0].strip()
                line = line.replace('N', '')
                rev_line = line[::-1]
                u_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
                rev_comp = "".join([u_dict[x] for x in rev_line])

                if f.lower().startswith('mini'):
                    split_line = [line[i:i + total_sequence_length] for i in range(0, len(line), mini_stride)]
                    split_line = [x for x in split_line if len(x) == total_sequence_length]

                    split_rev_comp = [rev_comp[i:i + total_sequence_length] for i in
                                      range(0, len(rev_comp), mini_stride)]
                    split_rev_comp = [x for x in split_rev_comp if len(x) == total_sequence_length]
                    mini_count = mini_count + len(split_line) + len(split_rev_comp)

                if f.lower().startswith('chr'):
                    split_line = [line[i:i + total_sequence_length] for i in range(0, len(line), core_stride)]
                    split_line = [x for x in split_line if len(x) == total_sequence_length]

                    split_rev_comp = [rev_comp[i:i + total_sequence_length] for i in
                                      range(0, len(rev_comp), core_stride)]
                    split_rev_comp = [x for x in split_rev_comp if len(x) == total_sequence_length]
                    core_count = core_count + len(split_line) + len(split_rev_comp)

                i = 0
                for s, rs in zip(split_line, split_rev_comp):
                    f_write.write('>{}_{}\n'.format(f, i))
                    f_write.write(s + '\n')
                    f_write.write('>{}_{}_r\n'.format(f, i))
                    f_write.write(rs + '\n')
                    i += 1

        f_write.close()
    print("Total mini count: {}\nTotal core count: {}\n".format(mini_count, core_count))
    print("Preprocessed fasta for filtering common core and mni is saved at: {}".format(save_path))


def _get_sequence(sequences, strain, label, kmer_length, kmer_count):
    result = []
    for whole_seq in sequences:
        whole_seq = whole_seq.strip().split(' ')
        for seq in whole_seq:
            kmers = [seq[i:i + kmer_length] for i in range(0, len(seq), kmer_length)]
            kmers = [x for x in kmers if len(x) == kmer_length]

            if len(kmers) == kmer_count:
                result.append(' '.join(kmers) + ', ' + strain + ', ' + label + '\n')
    return result


def create_dataset(base_dir, kmer_count, kmer_length):
    save_dir = os.path.join(base_dir, 'main_ng', 'bi_lstm', 'training_dataset',
                            '{}mer-{}seq'.format(kmer_length, kmer_count))
    os.makedirs(save_dir, exist_ok=True)

    train_file = os.path.join(save_dir, 'train_dataset_lstm_{}-seq_{}-kmer.csv'.format(kmer_count, kmer_length))
    tests_file_tf05 = os.path.join(save_dir,
                                   'test_dataset_lstm_tf05_{}-seq_{}-kmer.csv'.format(kmer_count, kmer_length))
    tests_file_emz25 = os.path.join(save_dir,
                                    'test_dataset_lstm_emz25_{}-seq_{}-kmer.csv'.format(kmer_count, kmer_length))

    f_train = open(train_file, 'w')
    f_tests_tf05 = open(tests_file_tf05, 'w')
    f_tests_emz25 = open(tests_file_emz25, 'w')

    kmer_dir = os.path.join(base_dir, 'cache/kmer_sequences/{}mer-{}seq'.format(kmer_length, kmer_count))

    for kmer_file in os.listdir(kmer_dir):
        if not '_{}mer'.format(kmer_length) in kmer_file:
            continue

        print("Processing file: ", kmer_file)
        with open(os.path.join(base_dir, 'cache', 'kmer_sequences', '{}mer-{}seq'.format(kmer_length, kmer_count),
                               kmer_file), 'r') as fp:
            kmer_seq = fp.readlines()

            if 'TF05' in kmer_file:
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_tests_tf05.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_tests_tf05.writelines(seq)

            if 'E_MZ5' in kmer_file.upper():
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_tests_emz25.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_tests_emz25.writelines(seq)

            if 'O135' in kmer_file.upper():
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_train.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_train.writelines(seq)

            if 'LpKY'.lower() in kmer_file.lower():
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_train.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_train.writelines(seq)

            if '70-15' in kmer_file:
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_train.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_train.writelines(seq)

            if 'B71' in kmer_file.upper():
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_train.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_train.writelines(seq)

    f_train.close()
    f_tests_tf05.close()
    f_tests_emz25.close()
    print("train/test dataset created successfully at {}".format(save_dir))


def create_dataset_from_filtered_fasta(base_dir, kmer_count, kmer_length):
    save_dir = os.path.join(base_dir, 'main_ng', 'bi_lstm', 'training_dataset',
                            '{}mer-{}seq'.format(kmer_length, kmer_count))
    os.makedirs(save_dir, exist_ok=True)

    train_file = os.path.join(save_dir, 'train_dataset_lstm_{}-seq_{}-kmer.csv'.format(kmer_count, kmer_length))
    tests_file_tf05 = os.path.join(save_dir,
                                   'test_dataset_lstm_tf05_{}-seq_{}-kmer.csv'.format(kmer_count, kmer_length))
    tests_file_emz25 = os.path.join(save_dir,
                                    'test_dataset_lstm_emz25_{}-seq_{}-kmer.csv'.format(kmer_count, kmer_length))

    f_train = open(train_file, 'w')
    f_tests_tf05 = open(tests_file_tf05, 'w')
    f_tests_emz25 = open(tests_file_emz25, 'w')

    kmer_dir = os.path.join(base_dir, 'cache/kmer_sequences/{}mer-{}seq_fasta'.format(kmer_length, kmer_count))

    for kmer_file in os.listdir(kmer_dir):
        if not '_{}mer'.format(kmer_length) in kmer_file:
            continue

        print("Processing file: ", kmer_file)
        with open(os.path.join(kmer_dir, kmer_file), 'r') as fp:
            kmer_seq = fp.readlines()

            if 'TF05' in kmer_file:
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_tests_tf05.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_tests_tf05.writelines(seq)

            if 'E_MZ5' in kmer_file.upper():
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_tests_emz25.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_tests_emz25.writelines(seq)

            if 'O135' in kmer_file.upper():
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_train.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_train.writelines(seq)

            if 'LpKY'.lower() in kmer_file.lower():
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_train.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_train.writelines(seq)

            if '70-15' in kmer_file:
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_train.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_train.writelines(seq)

            if 'B71' in kmer_file.upper():
                if kmer_file.lower().startswith('chr'):
                    seq = _get_sequence(kmer_seq, kmer_file, '0', kmer_length, kmer_count)
                    f_train.writelines(seq)

                if kmer_file.lower().startswith('mini'):
                    seq = _get_sequence(kmer_seq, kmer_file, '1', kmer_length, kmer_count)
                    f_train.writelines(seq)

    f_train.close()
    f_tests_tf05.close()
    f_tests_emz25.close()
    print("train/test dataset created successfully at {}".format(save_dir))


def process_filtered_fasta(kmer_length, kmer_count, mask_common):
    output_dir = os.path.join(BASE_DIR, 'data', 'training_dataset', 
                    'masked_common_core_mini' if mask_common else 'no_masked_common_core_mini',
                    f'{kmer_length}mer-{kmer_count}seq')
    
    os.makedirs(output_dir, exist_ok=True)

    fasta_dir = os.path.join(BASE_DIR, 'cache', 'kmer_sequences', '{}mer-{}seq_fasta'.format(kmer_length, kmer_count))

    train_file = os.path.join(output_dir, 'processed_dataset_{}mer.csv'.format(kmer_length))

    f_writer = open(train_file, 'w')

    for file in os.listdir(fasta_dir):
        if mask_common:
            if not file.endswith('fasta.filt.fastq'):
                continue
        else:
            if not file.endswith('fasta'):
                continue
        
        strain = file.split('.')[0]
        print("processing file: {}".format(file))
        with open(os.path.join(fasta_dir, file), 'r') as f_read:

            lines = f_read.readlines()
            for i in range(len(lines) - 1):
                line = lines[i]
                if line.lower().startswith('>chr'):
                    seq_line = lines[i + 1].strip()
                    kmers = [seq_line[x:x + kmer_length] for x in range(0, len(seq_line), kmer_length)]
                    kmers = [x for x in kmers if len(x) == kmer_length]

                    seq = ' '.join(kmers)
                    f_writer.write(seq + ',' + strain + ',' + '0' + '\n')

                if line.lower().startswith('>mini'):
                    seq_line = lines[i + 1].strip()
                    kmers = [seq_line[x:x + kmer_length] for x in range(0, len(seq_line), kmer_length)]
                    kmers = [x for x in kmers if len(x) == kmer_length]

                    seq = ' '.join(kmers)
                    f_writer.write(seq + ',' + strain + ',' + '1' + '\n')

    f_writer.close()


def create_train_splits(args, create_eval_splits):
    print("creating data splits...")
    output_dir = os.path.join(BASE_DIR, 'data', 'training_dataset',
                'masked_common_core_mini' if args.mask_common_core_mini else 'no_masked_common_core_mini',
                f'{args.kmer_length}mer-{args.kmer_count}seq')
    
    train_file = os.path.join(output_dir, 'processed_dataset_{}mer.csv'.format(args.kmer_length))

    df_train = pd.read_csv(train_file, header=None, usecols=[0, 1, 2], names=['sequence', 'strain', 'label'])
    df_train.dropna()

    # Remove duplicates. It deletes all instances of duplicate rows. This is because some duplicate rows belong to
    # core and some belong to mini
    df_train.drop_duplicates(subset=['sequence', 'label'], inplace=True) #, keep=False)

    # Filter core and mini and remove duplicates
    core = df_train[df_train.label == 0].sample(frac=1)
    mini = df_train[df_train.label == 1].sample(frac=1)

    # To balance the size of mini and core
    # if len(core) > len(mini):
    #     mini = mini.sample(n=len(core))
    # else:
    #     core = core.sample(n=len(mini))

    df = pd.concat([core, mini], axis=0)

    # To shuffle the core and mini rows
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)
    stats_writer = open(os.path.join(output_dir, 'dataset_stats.txt'), 'w')
    stats_writer.write(f"Total rows: {len(df)}\n")

    if create_eval_splits:
        valid_df_size = int(args.split_size * len(df))
        test_df_size = int(args.split_size * len(df))

        df_valid, df_test, df_train = df.iloc[:valid_df_size].copy(), df.iloc[valid_df_size: valid_df_size + test_df_size].copy(), df.iloc[valid_df_size + test_df_size:].copy()
        df_valid.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
        df_train.reset_index(drop=True, inplace=True)

        df_train.to_csv(os.path.join(output_dir, 'train.csv'), sep=',', index=False)
        df_test.to_csv(os.path.join(output_dir, 'test.csv'), sep=',', index=False)
        df_valid.to_csv(os.path.join(output_dir, 'valid.csv'), sep=',', index=False)

        stats_writer.write(f"Datast \t | Core Count | Mini Count | Total \n")
        stats_writer.write(f"--------------------------------------------------\n")
        stats_writer.write(f"Train \t | {len(df_train[df_train.label == 0])} | {len(df_train[df_train.label == 1])} | {len(df_train)}\n")
        stats_writer.write(f"Test  \t | {len(df_test[df_test.label == 0])} | {len(df_test[df_test.label == 1])} | {len(df_test)}\n")
        stats_writer.write(f"Valid \t | {len(df_valid[df_valid.label == 0])} | {len(df_valid[df_valid.label == 1])} | {len(df_valid)}\n")
        stats_writer.write(f"--------------------------------------------------\n")
        stats_writer.write(f"Total \t | {len(df[df.label == 0])} | {len(df[df.label == 1])} | {len(df)} \n")
    else:
        df_train = df
        df_train.to_csv(os.path.join(output_dir, 'train.csv'), sep=',', index=False)
        stats_writer.write(f"Datast \t | Core Count | Mini Count | Total \n")
        stats_writer.write(f"--------------------------------------------------\n")
        stats_writer.write(f"Train \t | {len(df_train[df_train.label == 0])} | {len(df_train[df_train.label == 1])} | {len(df_train)}\n")
        stats_writer.write(f"No test or validation sets were created\n")
        stats_writer.write(f"--------------------------------------------------\n")
        stats_writer.write(f"Total \t | {len(df[df.label == 0])} | {len(df[df.label == 1])} | {len(df)} \n")
    
    stats_writer.close()
    
    if os.path.exists(train_file):
        os.remove(train_file)

    

def mask_common_core_mini(kmer_length, kmer_count):
    # def fasta2seq(filename):
    #     with open(filename, 'r') as fread:
    #         with open(filename + '_final', 'w') as fwrite:
    #             lines = fread.readlines()
    #             for i in range(len(lines) - 1):
    #                 line = lines[i]
    #                 if line.startswith('>'):
    #                     fwrite.write(lines[i + 1].strip() + '\n')

    def run_filter_script(fname):
        with open(fname, 'rb') as file:
            script = file.read()

        print("filtering common core, mini sequences... {}".format(fname))

        rc = subprocess.call(script, shell=True)
        if os.path.exists(fname):
            os.remove(fname)

        print("preparing filtered sequence file...")
        # fasta2seq(filename + '.fasta.filt.fastq')

    all_strains = STRAINS_FQ_NEWLINE + STRAINS_FQ_NO_NEWLINE
    processes = []
    for strain in all_strains:
        print("Filtering common core, mini, and mt sequences for strain: {}".format(strain))

        filename = os.path.join(BASE_DIR, 'cache/kmer_sequences/{}mer-{}seq_fasta'.format(kmer_length, kmer_count),
                                strain)

        # bowtie command
        commands = [
            # "ref=/data/gnikesh/mini_prediction/data/filter_ref/bowtie2/cm.common.mt",
            "ref=/bulk/liu3zhen/research/projects/mini_prediction/data/C-Mcommon/bowtie2/cm.common.mt",
            "reads={}".format(filename + '.fasta'),
            # "out=`basename $reads`",
            "out=$reads",
            "# bowtie alignment",
            # "/data/gnikesh/softwares/bowtie2/bowtie2 --no-head --all \\",
            "/homes/liu3zhen/local/bin/bowtie2 --no-head --all \\",
            "-x $ref \\",
            "-f $reads | awk '$2==4' | cut -f 1 \\",
            "> ${out}.unaligned.reads",
            "# extract reads",
            # "/data/gnikesh/softwares/seqtk/seqtk subseq $reads ${out}.unaligned.reads > ${out}.filt.fastq",
            "/homes/liu3zhen/local/bin/seqtk subseq $reads ${out}.unaligned.reads > ${out}.filt.fastq",
            "# cleanup",
            "rm ${out}.unaligned.reads"
        ]

        fname = filename + '_filtersh.sh'
        with open(fname, 'w') as fp:
            for c in commands:
                fp.write(c + '\n')

        processes.append(multiprocessing.Process(target=run_filter_script, args=(fname,)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()



if __name__ == "__main__":
    """
    To add new strain data, please add the strain's name in the function 'fasta2seq()' in this file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--genome_fasta_dir', default='/bulk/liu3zhen/research/projects/mini_prediction/data', type=str,
                        help="path to the directory where genome fasta files for all strains are located")
    # parser.add_argument('--genome_fasta_dir', default='/data/gnikesh/projects/mini_pred/mini_pred_data', type=str,
                        # help="path to the directory where genome fasta files for all strains are located")
    parser.add_argument('--kmer_length', default=9, type=int, help="length of kmer")
    parser.add_argument('--kmer_count', default=11, type=int, help="sequence length for input to bidirectional LSTM")
    parser.add_argument('--mini_stride', default=1, type=int, help="stride length for mini to overlap")
    parser.add_argument('--core_stride', default=27, type=int, help="stride length for core to overlap if 0, selects best stride length to create balanced dataset")
    parser.add_argument('--mask_common_core_mini', action='store_true', help='provide this flag if you want to mask common core and mini sequences')
    parser.add_argument('--split_size', type=float, default=0.1, help='percentage size of the test and validation splits')
    
    args = parser.parse_args()

    fasta2seq(BASE_DIR, args.genome_fasta_dir)
    preprocess_fasta_for_filtering(kmer_length=args.kmer_length, kmer_count=args.kmer_count,
                                    mini_stride=args.mini_stride, core_stride=None)
    
    mask_common = args.mask_common_core_mini
    if mask_common:
        print("Masking the mt and common core and mini sequences using bash script")
        mask_common_core_mini(kmer_length=args.kmer_length, kmer_count=args.kmer_count)

    process_filtered_fasta(kmer_length=args.kmer_length, kmer_count=args.kmer_count, mask_common=mask_common)
    create_train_splits(args, create_eval_splits=True)


