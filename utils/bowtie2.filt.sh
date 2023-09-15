#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00

ref=/bulk/liu3zhen/research/projects/mini_prediction/data/C-Mcommon/bowtie2/cm.common.mt
reads=$1
outbase=$2
#out=`basename $reads`

# bowtie alignment
/homes/liu3zhen/local/bin/bowtie2 --no-head --all \
	-x $ref \
	-U $reads | awk '$2==4' | cut -f 1 \
	| /homes/liu3zhen/local/bin/seqtk subseq $reads - \
	| gzip > $outbase

# if the input file is a fasta file, use -f to replace -U

