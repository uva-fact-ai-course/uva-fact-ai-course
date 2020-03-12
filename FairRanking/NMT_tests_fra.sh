#!/bin/bash

echo 'Starting tests...'

export PYTHONPATH="$PWD"

cd $PYTHONPATH/NMT

python3 main.py
python3 main.py --use_uniform True

echo 'Done!'