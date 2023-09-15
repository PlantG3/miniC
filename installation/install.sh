cd ~/venvs/
module load Python/3.9.6-GCCcore-11.2.0
virtualenv minicLSTM
source ~/venvs/minicLSTM/bin/activate
# requirement
requirements=/bulk/liu3zhen/research/projects/mini_prediction/main_minicLSTM/installation/requirements.txt
pip install -r $requirements 

