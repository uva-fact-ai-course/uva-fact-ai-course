#!/bin/bash

rsync -avz --include='*ASGD_True' --exclude='*TEMP*' LISA-FACT-20:~/FACT-replicate/models ./FACT-replicate/models