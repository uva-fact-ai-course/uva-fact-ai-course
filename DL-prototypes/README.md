# FACT-AI: Towards Hierarchical Explanation
Private Github repository for the course Fairness, Accountability, Confidentiality and Transparency in AI at the University of Amsterdam. This repository contains the reproduction source code, results and extended methods on the ["Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network that Explains Its Predictions"](https://arxiv.org/abs/1710.04806) paper. 

## Authors
* Albert Harkema (albert.harkema@student.uva.nl)
* Anna Langedijk (annalangedijk@gmail.com)
* Christiaan van der Vlist (christiaan.vandervlist@student.uva.nl)
* Hinrik Snær (hinriksnaer@gmail.com)

## Based on
Our implementation is based on the tensorflow code in https://github.com/OscarcarLi/PrototypeDL.
It extends the original implementation by using hierarchical prototypes.

## Evaluate or reproduce results
The notebook `results.ipynb` can be run to evaluate the existing algorithm and the training scripts are provided for reproducibility. To retrain the model or use different parameters, see instructions for `run.py` below.

# Instructions
First, (create and then) activate the correct environment:
```
[conda env create -f environment_prototype.yml]
source activate prototype 
```

Then, run the code either from the IPython notebook, or by running `run.py`: 
```
python run.py [--hier true] [--seed <int>] [--dir <directory name>] ...
```
This will run the code with default parameters/seed for reproduction.
Additional parameters can be set according to their descriptions: run 
```
python run.py --help
```
for more information about all the different parameters.

# Code base 
All of our non-wrapper code is included in the `src/` directory. The basic modules are in `src/network`. They are combined within the `src/model.py` file, together with all the files necessary for training.

