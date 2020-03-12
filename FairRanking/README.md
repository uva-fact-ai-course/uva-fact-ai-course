# Relationship between Attention, Complexity and Explainability

This repository is a part of a project work for Fairness, Accountability, Confidentiality and Transparency course of AI Master's at University of Amsterdam. 

The results displayed in our report Relationship between Attention, Complexity and Explainability show that when attention is being used by the model, which is the case for QA and autoencoding, permuting attention distributions alter the model's prediction. However, we cannot conclude that interpretable attention correlates with the complexity of the task. Future work on NMT modelling might change this. 

Based on work from:
https://github.com/successar/AttentionExplanation

Check out their README for prerequisite steps for the core code from them. As the code for the seq2seq part requires some of the same dependencies (seaborn, pytorch, matplotlib, numpy, etc.), please check their dependencies first.

We recommend to run this project in Linux environment, as difficulties were encountered when running the core code under Win10.

Use the FACT_project_API notebook as a user interface. From there you should be able to execute all code and visualize all results.

The NMT folder contains code for the seq2seq experiments, whereas the "Transparency" folder is the modified codebase from the original paper.

Team Info:
- David Cerny, david.cerny@student.uva.nl
- Oliviero Nardi, oliviero.nardi@student.uva.nl
- Liselore Borel Rinkes
- TA: Marco Heuvelman
