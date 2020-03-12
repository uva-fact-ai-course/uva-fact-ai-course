# list jupyter kernels
jupyter kernelspec list
jupyter kernelspec uninstall unwanted-kernel

# add env to jupyter 
pip install ipython jupyter ipykernel
python -m ipykernel install --user --name senn

# remove conda environment
conda remove --name myenv --all

# list all packages with version in current environment
conda list