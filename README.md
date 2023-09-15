# miniC
A LSTM deep learning for predicting if a Pyricularia strain carries supernumerary chromosomes (or mini-chromosomes)

# Installation


# Preparation
Here, the example is inputing read data in FASTQ format. We will first removed sequences commonly found in both core and mini. The step is referred to be filtering.  
```
k=9
kcount=11
h5model=models/modelv1.0.k9c11.h5
outbase=`basename $infq | sed 's/.fq.gz//g'; s/.fq//g'; s/.fastq//g'`
filtfq=${outbase}.filt.fq.gz

# filter
infq_base=`basename $infq`
bash utils/bowtie2.filt.sh $infq $filtfq
```

# Prediction
Use the trained model for the prediction.
```
pred_prob_threshold=0.99
python ../../../utils/minicPred.py \
   --model_path $h5model \
   --data_file $filtfq \
   --kmer_length $k \
   --kmer_count $kcount \
   --output_dir . \
   --prediction_threshold $pred_prob_threshold \
   --save_pred_seq
```

