#!/usr/bin/env bash

#SBATCH --job-name=preprocess-dailymail
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=125GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jtc440@nyu.edu
#SBATCH --output="preprocess-dailymail-%j.out"
#SBATCH --err="preprocess-dailymail-%j.err"


source ~/.bashrc
source activate ds1012

SRCDIR=$HOME/dev/language-model-bias
DATASET_DIR=/beegfs/jtc440/ds1012/dailymail
OUTPUT_DIR=/beegfs/jtc440/ds1012/preprocessed_data/dailymail

module purge

python $SRCDIR/preprocess.py $DATASET_DIR $OUTPUT_DIR .story -n 28
