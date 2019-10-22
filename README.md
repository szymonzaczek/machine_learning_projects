# __Machine learning projects__

### Currently published projects:
- __NLP: Sentiment analysis of reviews using artificial neural networks__
- __Predicting USD/PLN exchange rates with recurrent neural networks__
- __Reducing dimensionality of molecular simulations__

## __NLP: Sentiment analysis of reviews using artificial neural networks__
Basing on the text of the review, constructed Artificial Neural Network predicts if a given opinion is negative, neutral or positive.

## Type of machine learning problem
- Supervised Machine Learning
- Multi-label classification

### Technologies used
- Python
- NLP (Natural Language Processing) - Sentiment Analysis
- Deep learning - Artificial Neural networks

#### Packages used
- sklearn
- keras
- tensorflow
- nltk
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud

### Overview
This notebook illustrates an application of  one of Natural Language Processing technologies - Sentiment Analysis. Given the written review of a product bought in an online shop, constructed Deep Neural Network predicts whether this opinion is negative, neutral or positive.

Dataset contains reviews of Cell Phones and Accessories collected by Julian McAuley from Amazon web shop. Dataset was downloaded from: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz

Significant preprocessing of data was conducted in order to construct ANN. First, reviews from json file must have been properly loaded. Dataset was transformed to Pandas Dataframe format, which made further processing easier. Dataset was skimmed to have equal number of negative, neutral and positive reviews. Then, text of reviews was tokenized and only tokens that were found more often than 20 times in the whole dataset were retained for model construction and subsequent predictions.

Wordcloud was used to illustrate which words were most frequent in the dataset.

Artificial Neural Network contains 2 deep layers (20 nodes each) and one output layer (since there are 3 labels, 3 output nodes were used). Several attempts were made to avoid overfitting - high dropout rate (0.4), weight and activation function regularization.

Constructed confusion matrix illustrates that the model is competent with differentiating between positive and negative reviews though it struggles with neutral reviews. This is due to the fact that even though the rating for a given product might be neutral, the overall undertone of the review is usually polarized either towards negative or positive attributes.


## __Predicting USD/PLN exchange rates__
Predicting USD/PLN exchange rates with Long-Short Term Memory Recurrent Neural Networks.

## Type of machine learning problem
- Supervised Machine Learning
- Regression

### Technologies used
- Python
- Deep learning - Long Short Term Memory (LSTM) Recurrent Neural Networks

#### Packages used
- sklearn
- keras
- tensorflow
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

## Type of machine learning problem
- Unsupervised Machine Learning

### Technologies used
- Python
- Principal Component Analysis (PCA)
- k-means clustering

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
