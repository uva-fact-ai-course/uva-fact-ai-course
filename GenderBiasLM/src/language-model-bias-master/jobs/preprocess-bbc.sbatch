#!/usr/bin/env bash

#SBATCH --job-name=preprocess-bbc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jtc440@nyu.edu
#SBATCH --output="preprocess-bbc-%j.out"
#SBATCH --err="preprocess-bbc-%j.err"


source ~/.bashrc
source activate ds1012

SRCDIR=$HOME/dev/language-model-bias
DATASET_DIR=/beegfs/yw3004/projects/language_bias/data/bbc
OUTPUT_DIR=/beegfs/jtc440/ds1012/preprocessed_data/bbc

module purge

python $SRCDIR/preprocess.py $DATASET_DIR $OUTPUT_DIR .txt -n 8
