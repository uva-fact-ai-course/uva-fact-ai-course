# Bias in Face Detection? Yes, then how to deal with it

##### Authors

| Name                          | Email                         |
|-------------------------------|:----------:|
| Luisa Ebner                   |l.t.ebner@student.vu.nl        |
| Frederic Robert Chamot        |  f.chamot@student.vu.nl           |
| Julio Joaquín López González  |j.j.lopezgonzalez@student.vu.nl|
| Maximilian Knaller            |max.knaller@student.uva.nl     |

Teaching Assistant: Simon Passenheim

## Summary

In this Repo we aim to reproduce an extend the ideas presented in the paper
[Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure](https://lmrt.mit.edu/sites/default/files/AIES-19_paper_220.pdf).

For reproduction we will mainly follow an implementation that was provided in an
[course](https://github.com/aamini/introtodeeplearning) that seems to be closely affiliated with the authors,
as both are published from MIT (Massachusetts Institute of Technology). However, the code was removed from their
Github on 25.01.2020, so we can't link to the source anymore.  Extensions focus mainly on the reevaluation of the
findings based on a bigger and especially for the topic of bias in face detection created dataset
called [FairFace](https://github.com/joojs/fairface).

All details, experiments, results and discussions can be found in the notebook `Bias_in_face_detection.ipynb`, that
follows this order. We will start by looking at the data used in the project. Then evaluate if there is bias in
classic face detection systems. Lastly, we will look into the proposed mehtod for reducing this probable bias.

Please make sure to check out the Requirements section before starting.

## Requirements

### Virtual Environment

There is a conda environment for ubuntu fact_ai_ubuntu.yml supplied. Given you have Conda installed and using ubuntu you can install the environment using

```bash
conda env create -f fact_ai_ubuntu.yml
```

There is conda environment for mac fact_ai_mac.yml. Given you have Conda installed and using mac you can install the environment using

```bash
conda env create -f fact_ai_mac.yml
```

Then you have to run

```bash
conda install nb_conda_kernels
```

to automatically make all environments available as a Jupyter Kernel.
Afterwards you should be able to select the environment as kernel in a started Jupyter Notebook.

If you should have problems seeing the kernel in the Jupyter Notebook, you can also manually add it
to the available kernels, by running:

```bash
python -m ipykernel install --user --name FACT-AI --display-name "Python (FACT-AI)"
```
