# miniC
A LSTM deep learning for predicting if a *Pyricularia* strain carries supernumerary chromosomes (or mini-chromosomes)

# Installation
Python3.9 was tested. A virtual environment can be set up by using the software list in [requirements.txt](installation/requirements.txt).
```
pip install -r installation/requirements.txt
```

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
Use the trained model for the prediction. The current model can be downloaded from [here](https://people.beocat.ksu.edu/~liu3zhen/models/model_final.h5)

```
h5model=<path_to_model>
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

# Prediction result
Here is an example result for the prediction using a FASTQ dataset. "mini_percent" is the value used to assess the presence of minis. The threshold is 0.015 using the current model. The value larger than the threshold evidences the presence of mini(s) in the strain.   
```
data,totalSeq,mini,core,mini_percent,core_percent
B71.R1.filt.fq.gz,14386540,473280,13913260,0.032897,0.967103
```

# Citation
<to be added>

