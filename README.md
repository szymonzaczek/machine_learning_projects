# __Machine learning projects__

### Currently published projects:
- __Predicting USD/PLN exchange rates with recurrent neural networks__
- __Reducing dimensionality of molecular simulations__


## __Predicting USD/PLN exchange rates__
Predicting USD/PLN exchange rates with Long-Short Term Memory Recurrent Neural Networks.

### Technologies used
- Python
- Deep learning - LSTM Recurrent Neural Networks

#### Packages used
- sklearn
- keras
- numpy
- matplotlib

### Overview

This repository contains Jupyter notebook, in which I applied Recurrent Neural Networks for predicting USD/PLN exchange rates.

Dataset contains true data for USD/PLN exchange rates from 2.01.2014 to 27.09.2019 obtained from http://fx.sauder.ubc.ca/.

For making predictions, I constructucted multi-layered Long-Short Term Memory (LSTM) Recurrent Neural Networks.

There are 6 LSTM layer, each one has 50 neurons and a dropout rate equal to 20%.

For making predictions, I constructed multi-layered (6 layers) Long-Short Term Memory Recurrent Neural Networks. Each layer has dropout rate specified as 0.2 (random 20% of the neurons are not used during each update while training to prevent overfitting).

## __Dimensionality reduction (PCA and k-means clustering) of molecular simulations__
Application of unsupervised machine learning algorithm (PCA) allowed for identification of correlated atomic motions during molecular dynamics (MD) simulations of UbiX enzymes.

Presented herein application of unsupervised machine learning was used in the published scientific paper:

[![DOI:10.1002/cbic.201800389](https://zenodo.org/badge/DOI/10.1002/cbic.201800389.svg)](https://doi.org/10.1002/cbic.201800389)

### Technologies used
- Python
- Unsupervised Machine Learning - PCA and k-means clustering

#### Packages used
- pyemma
- matplotlib
- numpy
- molpx

### Overview
This notebook presents an application of PCA and k-means clustering that aided me in providing an insight into identifying dominant movements of an enzyme during performed molecular dynamics simulations.

Dataset consists of results of molecular dynamics simulations of an enzymatic system, carried out for 500 ns in Amber software. Those simulations were performed with Tesla K40 XL GPU on Prometheus Supercomputer, that is a part of PLGrid initiative. Those simulations took about 30 days to complete.

There are two files available. 'topology_4zaf.prmtop' is the topology file, which contains all information regarding the definition of molecular system (i. e. which atom is bonded to which), whereas 'traj_short.nc' is the trajectory file, which is a binary file containing info about xyz coordinates of each atom. Simula

Herein, PyEmma package was used due to its convenient handling of molecular data (it does not require converting topology and trajectory files to numpy array).

Principal Component Analysis (PCA) helped identifying dominant movements in the molecular system. K-means clustering allowed identifying mostly populated molecular states after PCA transformation.
