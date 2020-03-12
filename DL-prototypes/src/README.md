# FACT-AI: Towards Hierarchical Explanation
Private Github repository for the course Fairness, Accountability, Confidentiality and Transparency in AI at the University of Amsterdam. 

## Authors
* Albert Harkema (12854794) (albert.harkema@student.uva.nl)
* Anna Langedijk (12297402) (annalangedijk@gmail.com)
* Christiaan van der Vlist (12876658) (christiaan.vandervlist@student.uva.nl)
* Hinrik Snær (12675326) (hinriksnaer@gmail.com)

## Based on
Our implementation is based on the tensorflow code in https://github.com/OscarcarLi/PrototypeDL.
It extends the original implementation by using hierarchical prototypes.

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
Additional parameters can be set according to their descriptions, run 
```
python run.py --help
```
for more information about all the different parameters.

# Code base 
All of our non-wrapper code is included in the `src/` directory. The basic modules are in `src/network`. They are combined within the `src/model.py` file, together with all the files necessary for training.

## Environment
I based this environment on the environment provided by the DL course and added jupyter, matplotlib for easy IPython notebooks.
This includes an older version of `pillow`, see https://github.com/python-pillow/Pillow/issues/4130. This issue is encountered on older versions of packages (for instance on Lisa).
