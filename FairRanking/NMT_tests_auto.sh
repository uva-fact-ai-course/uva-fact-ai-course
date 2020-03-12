#!/bin/bash

echo 'Starting tests...'

export PYTHONPATH="$PWD"

cd $PYTHONPATH/NMT

python3 main.py --source eng
python3 main.py --source eng --use_uniform True

echo 'Done!'