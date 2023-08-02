# minicPRED
the LSTM deep learning for predicting if a Pyricularia strain carries mini-chromosomes

# Test
To test the sequences, convert the sequence as a csv file with ```"sequence,strain,label"``` columns. For example: look into ```data/sample_data/test_set_sample.csv```.
Run the ```test.py``` file as follows  

```python test.py --saved_model path_to_saved_model --test_csv path_to_csv_file --kmer_length <kmer_length> --kmer_count <kmer_count>```  

The parameters ```test.py``` file accepts are:  

```--saved_model```: trained model name with full path  
```--test_csv```: path to the test dataset file eg: /home/user/mini_prediction/train.csv  
```--output_dir```: path to the results and log output folder  
```--save_pred_seq```: whether to save the prediction score  
```--prediction_threshold```: threshold for classifying the prediction to positive class label  
```--batch_size```: size of a batch to use for processing large dataset while making the prediction  
```--kmer_length```: size of kmer  
```--kmer_count```: total number of kmers in a sequence  
```--gpu```: gpu id to use  

For e.g.  

```python test.py --saved_model "saved_model/model_final_k9-11.h5" --test_csv "data/sample_data/test_set_sample.csv" --kmer_length 9 --kmer_count 11 --save_pred_seq```
