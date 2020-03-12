#!/bin/bash

echo 'Starting tests...'

export PYTHONPATH="$PWD"

echo 'SST'
cd $PYTHONPATH/Transparency/preprocess/SST
ipython SST.py

echo '20News'
cd $PYTHONPATH/Transparency/preprocess/20News
ipython 20News.py

echo 'IMDB'
cd $PYTHONPATH/Transparency/preprocess/IMDB
ipython IMDB.py

echo 'AGNews'
cd $PYTHONPATH/Transparency/preprocess/ag_news
ipython AGNews.py

echo 'bAbI'
cd $PYTHONPATH/Transparency/preprocess/Babi
ipython Babi.py

echo 'SNLI'
cd $PYTHONPATH/Transparency/preprocess/SNLI
ipython SNLI.py

echo 'Finished!'
