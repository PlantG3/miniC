# Python version 3.9.6
# module load Python/3.9.6-GCCcore-11.2.0
cd ~/venvs/
virtualenv minicLSTM
source ~/venvs/minicLSTM/bin/activate
# requirement
requirements=requirements.txt
pip install -r $requirements 

