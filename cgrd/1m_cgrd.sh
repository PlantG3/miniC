#!/bin/bash

# installation of CGRD can refer to:
# https://github.com/liu3zhenlab/CGRD.git

# activate CGRD conda
conda activate cgrd

# query reads:
qry=query
qfq1=query.R1.fq.gz
qfq2=query.R2.fq.gz

# subject reads: B71
subj=B71
sfq1=B71.R1.fq.gz
sfq2=B71.R2.fq.gz

# reference: B71Ref2
ref=B71Ref2.fasta

# bins using knum of 1000
binbed=MoT_B71Ref2.knum1000.bin.bed

# thread number
ncpu=8

# CGRD run
perl ~/scripts2/CGRD/cgrd \
	--ref $ref \
	--binbed $binbed \
	--subj $subj \
	--sfq1 $sfq1 --sfq2 $sfq2 \
	--qry $qry \
	--qfq1 $qfq1 --qfq2 $qfq2 \
	--adj0 --cleanup \
	--groupval "-5 -0.4 0.4 0.8" \
	--prefix $qry --threads $ncpu

