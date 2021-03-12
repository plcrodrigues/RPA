# Riemannian Procrustes Analysis

(Installation procedure is described below)

## Description

This repository contains Python code for using the Riemannian Procrustes Analysis method proposed in [1]. This procedure is a Transfer Learning approach for dealing with the statistical variability of EEG signals recorded on different sessions and/or from different subjects. This is a common problem faced by Brain-Computer Interfaces (BCI) and poses a challenge for systems that try to reuse data from previous recordings to avoid a calibration phase for new users.

RPA is based on the concept of Procrustes analysis [2]. It works by matching the statistical distributions of two datasets using simple geometrical transformations (translation, scaling and rotation) over the data points in both datasets. We use symmetric positive definite matrices (SPD) as statistical features for describing the EEG signals, so the geometrical operations on the data points respect the intrinsic geometry of the SPD manifold.

We have included two notebooks with examples of application of the RPA on data from two BCI paradigms : motor imagery and SSVEP. Both notebooks are self-explanatory and illustrate the improvement in cross-subject classification when RPA is used as a step for matching the statistical distributions of the datasets.

[1] Pedro L. C. Rodrigues, Christian Jutten, and Marco Congedo - "*Riemannian Procrustes Analysis : Transfer Learning for Brain-Computer Interfaces*", DOI : 10.1109/TBME.2018.2889705 ([link](https://ieeexplore.ieee.org/document/8588384))

[2] J. C. Gower and G. B. Dijksterhuis, "*Procrustes Problems*". Oxford University Press, January 2004

## Installation

**Must be running Python 3**

To install, you can fork or clone the repository and go to the downloaded directory, then run

```
pip install -r requirements.txt
python setup.py develop # because no stable release yet
```

This will ensure that whenever you do ```import rpa``` your code will find the scripts (you might prefer to do this on a virtual environment).

**Requirements we use**

- autograd==1.3
- pyriemann==0.2.6
- pymanopt==0.2.5
- scipy==1.6.1
- matplotlib==3.2.2
- joblib==1.0.1
- tqdm==4.56.0
- numpy==1.20.1
