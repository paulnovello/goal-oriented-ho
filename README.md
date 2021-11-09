
# Example experiments for the paper "Goal-Oriented Sensitivity Analysis of Hyperparameters in Deep Learning"

## Requirements
---

The jupyter notebook `paper_experiments.ipnyb` and the python script `paper_experiments.py` have been run with 
* scipy = 1.6.2
* numpy = 1.20
* python = 3.9.4

Alternatively, you can create a conda environment with all dependencies satisfied using the provided `.yml` file, `explainable_ho.yml`:
~~~
conda env create --file explainable_ho.yml
~~~

Then, activate the environment:

~~~
conda activate explainable_ho
~~~

## Content

---

The notebook `paper_experiments.ipynb` reproduces the experiments of sections 4.1, 4.2, and 4.3 of the paper. It also aims to help the reader to get familiar with HSIC usage: do not hesitate to modify the examples.

The file `paper_experiments.py` is simply the python script version of the notebook.
