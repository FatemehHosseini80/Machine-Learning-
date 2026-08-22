
# Custom ID3 Decision Tree on Iris Dataset

## Overview
This repository contains a complete, from-scratch implementation of the **ID3 (Iterative Dichotomiser 3)** decision tree algorithm in Python. The model is trained and evaluated on the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). 

Since the standard ID3 algorithm requires categorical features, a major highlight of this project is the custom **Entropy-based Discretization** algorithm. It automatically converts continuous numerical features into categorical bins using Information Gain to find the optimal cut-points.

## Key Features
* **Entropy & Information Gain Calculation**: Core mathematical components built from scratch using NumPy.
* **Continuous to Categorical Binning**: Features an automated ID3-style greedy algorithm (`best_k_cutpoints`) to segment continuous variables into discrete labels (`Very Short`, `Short`, `Medium`, `Long`).
* **ID3 Algorithm**: A recursive tree-building function that splits the dataset based on the highest Information Gain.
* **Prediction & Evaluation**: Custom inference engine to traverse the generated dictionary-based tree and calculate accuracy.

## Prerequisites
To run this project, you need Python 3.x and the following libraries:
* `pandas`
* `numpy`
* `scikit-learn` (used strictly for loading the Iris dataset and `train_test_split`)
