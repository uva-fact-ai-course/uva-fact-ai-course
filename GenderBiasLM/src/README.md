

# FACT-AI


We will be reproducing and extending upon ["Identifying and Reducing Gender Bias in Word-Level Language Models"](https://arxiv.org/abs/1904.03035). The orignal repository of the paper can be found at https://github.com/BordiaS/language-model-bias.

# Library versions

The paper repository mentions usage of `pytorch=0.4`.

Furthermore `spacy` and `scipy` are needed to run everything inside the repository, we first install recent versions as versions are not specified.


# Usage:


`conda env create -f FACT_env.yml`

`conda activate FACT`

`cd language-model-bias-master/`

`cd awd-lstm/`

the paper includes a version of the AWS language model, which lacks the required "getdata" file (both for that repository's dataset, as well as the data required for the given paper), for this reason we **have to set up an equivalent**, which should include the datasets:

* Penn Treebank
* WikiText-2     
* [CNN/Daily Mail](https://github.com/deepmind/rc-data/)

The Penn treebank & wikitext-2 were sourced from the getdata.sh file which we conclude is missig from the original repo, but which is [refferred](https://github.com/BordiaS/language-model-bias/tree/master/awd-lstm) to in their explanation/readme on how to run the code. Since they refer to a different repository, with a [shell-file](https://github.com/salesforce/awd-lstm-lm/blob/32fcb42562aeb5c7e6c9dec3f2a3baaaf68a5cb5/getdata.sh) with that name, of which we copied the relevant parts into our repository.

`./getdataLin.sh`


then it should be possible to run with [recommended](https://github.com/BordiaS/language-model-bias/tree/master/awd-lstm) arguments:

## pen treebank

`python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt --gender_pair_file penn`

## wikitext 2

`python main.py --epochs 750 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --seed 1882 --gender_pair_file wiki`


to use our "reimplementation" copy the data directory:

language-model-bias-master/awd-lstm/data

to FACT-replicate, such that you have a resulting directory:

FACT-replicate/data ,which includes "penn" and "wikitext-2" directories

once you've got the data you can call:

`python train.py --dataset penn`

or (for the other dataset)

`python train.py --dataset wikitext-2`

alternatively with the extra argument `--device cpu` if you want to run the mdoel on cpu
