#!/bin/bash

echo 'Starting tests...'

export PYTHONPATH="$PWD"

cd $PYTHONPATH/Transparency

#QA
python3 train_and_run_experiments_qa.py --dataset babi_1 --data_dir . --output_dir outputs/ --attention all --encoder lstm
python3 train_and_run_experiments_qa.py --dataset babi_1 --data_dir . --output_dir outputs/ --attention none --encoder lstm

echo 'Done!'