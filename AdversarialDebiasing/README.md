### Adversarial Debiasing

In the context of the course FACT in AI an attempt is made at reproducing a paper on fairness, namely
Mitigating Unwanted Biases with Adversarial Learning ([Zhang et al., 2018](https://arxiv.org/abs/1801.07593)). This is an in-processing method for the mitigation of unwanted biases in learning algorithms and is applicable to both continuous and discrete domains.

This repository contains code for running the experiment from the paper on the UCI Adult dataset,
as well as extensions that implement the proposed adversarial network in order to debias the
UTKFace and UCI Communities and Crime datasets. To run the experiments,
please follow the instructions below.

### Prerequisites
Anaconda: https://www.anaconda.com/distribution/

### Getting Started
First open the Anaconda prompt, clone the entire repository and move to this directory:
```bash
git clone https://github.com/uva-fact-ai-course/uva-fact-ai-course
cd AdversarialDebiasing
```

Then create and activate the environment necessary for running the experiments, using the following commands:
```bash
conda env create -f environment.yml
conda activate advdebias
```

To deactivate the environment, use:
```bash
conda deactivate
```

To view the notebook with our experimental results, run:
```bash
jupyter notebook results.ipynb
```

### Running the experiments
New experiments can be conducted using the main.py file. The dataset to be used can be specified by passing it as an argument on the command line. The usage of the file is specified as follows:

```bash
usage: main.py [-h] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE]
               [--predictor_lr PREDICTOR_LR] [--adversary_lr ADVERSARY_LR]
               [--debias] [--val] [--dataset_name DATASET_NAME] [--seed SEED]
               [--save_model_to SAVE_MODEL_TO] [--lr_scheduler {exp,lambda}]
               [--alpha ALPHA]

optional arguments:
  -h, --help            show this help message and exit
  --n_epochs N_EPOCHS   Number of epochs
  --batch_size BATCH_SIZE
                        Batch size
  --predictor_lr PREDICTOR_LR
                        Predictor learning rate
  --adversary_lr ADVERSARY_LR
                        Adversary learning rate
  --debias              Whether to perform debiasing or not
  --val                 Whether to use a validation set during training
  --dataset_name DATASET_NAME
                        Name of the dataset to be used: adult, crime, images
  --seed SEED           Fixed seed to train with
  --save_model_to SAVE_MODEL_TO
                        Output path for saved model
  --lr_scheduler {exp,lambda}
                        Learning rate scheduler to use
  --alpha ALPHA         Value of alpha for the adversarial training
```

Some of the arguments listed above might be irrelevant for some experimental set-ups or inappropriate for some data. Below you will find commands for running an experiment with the default settings on a specific dataset, including arguments specific to it. Commands for reproducing the results presented in the notebook are also provided.

#### UCI Adult dataset
The UCI adult dataset is the default data for doing the debiasing experiments. To train and test without debiasing, run:
```bash
python main.py
```

To debias, run:
```bash
python main.py --debias
```

To replicate our results, run:
```bash
python main.py --batch_size 128 --predictor_lr 0.1 --n_epochs 10
python main.py --debias --batch_size 128 --predictor_lr 0.01 --adversary_lr 0.001 --n_epochs 30 --alpha 0.3
```

#### UCI Communities and Crime dataset
To train, validate and test on the UCI Communities and Crime dataset, run:

```bash
python main.py --dataset_name crime --val
```

To replicate our results, run:
```bash
python main.py --dataset_name crime --n_epochs 50
python main.py --dataset_name crime --debias --batch_size 64 --predictor_lr 0.002 --adversary_lr 0.005 --n_epochs 210 --alpha 1
```

#### UTKFace dataset
The UTKFace set is not present in the data folder of this directory. The data is downloaded and placed into the right local folder when experimenting with it for the first time. To train, validate and test on the UTK Face dataset, run:

```bash
python main.py --dataset_name images --val
```

To replicate our results, run:
```bash
python main.py --dataset_name images --batch_size 128 --predictor_lr 0.001 --n_epochs 30
python main.py --dataset_name images --batch_size 128 --debias --predictor_lr 0.001 --adversary_lr 0.001 --n_epochs 30 --alpha 0.1
```

### Authors
- [Vanessa Botha](https://github.com/VanessaBotha) - vanessa.botha@student.uva.nl
- [Nithin Holla](https://github.com/Nithin-Holla) - nithin.holla@student.uva.nl
- [Azamat Omuraliev](https://github.com/azamatomu) - azamat.omuraiev@student.uva.nl
- [Leila Talha](https://github.com/LaMouilleBoucle) - leila.talha@student.uva.nl

### Acknowledgements
We would like to express great appreciation to IBM for releasing the [AI Fairness 360 toolkit](https://github.com/IBM/AIF360) that has been of inspiration to us, when parameter settings required to reproduce results were not mentioned by Zhang and colleagues. In addition we are grateful for the datasets made publicly available by UCI and [susanqq](https://github.com/susanqq). Finally we would like to thank Leon Lang for providing us with advice and feedback and for swiftly responding to our e-mails, answering our questios.
