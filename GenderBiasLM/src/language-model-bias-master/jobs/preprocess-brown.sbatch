#!/usr/bin/env bash

#SBATCH --job-name=preprocess-brown
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb6416@nyu.edu
#SBATCH --output="preprocess-brown-%j.out"
#SBATCH --err="preprocess-brown-%j.err"

#source ~/.bashrc

source /home/sb6416/pyenv/py3.6.3/bin/activate


SRCDIR=$HOME

DATASET_DIR=/beegfs/sb6416/ds1012/brown/training
OUTPUT_DIR=/beegfs/sb6416/ds1012/preprocessed_data/brown/training

module purge

python $SRCDIR/preprocess.py $DATASET_DIR $OUTPUT_DIR .txt -n 10
