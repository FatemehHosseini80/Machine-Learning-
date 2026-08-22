# Noise Reduction Using PCA and Autoencoders

This project demonstrates and compares two approaches for **image denoising**:

* **Principal Component Analysis (PCA)**
* **Deep Learning Autoencoder**

The **Fashion MNIST** dataset is used as a benchmark to evaluate the ability of both methods to reconstruct clean images from noisy inputs.

## 📌 Project Overview

The project follows four main stages:

### 1. Data Loading and Noise Generation

* Loads the **Fashion MNIST** dataset using `torchvision`.
* Normalizes the 28×28 grayscale images to the range **[-1, 1]**.
* Adds artificial **Gaussian noise** to the images to simulate noisy input data.

### 2. PCA-Based Denoising

PCA is applied to reduce the dimensionality of the images and retain their most important features.

The project includes:

* A **custom PCA implementation** using NumPy and eigendecomposition.
* A standard implementation using `scikit-learn`.
* Analysis of **cumulative explained variance** to determine the number of components required to preserve a desired percentage of the information, such as **95% variance**.
* Image reconstruction using the selected principal components.

### 3. Autoencoder-Based Denoising

A fully connected **Autoencoder** is implemented using PyTorch.

The architecture consists of:

* An **Encoder** that compresses the 784-dimensional input into a **100-dimensional latent representation**.
* A **Decoder** that reconstructs the original 28×28 image.
* `Tanh` activation for the reconstructed output.

The model is trained using:

* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam
* **Epochs:** 10

### 4. Evaluation and Comparison

The denoising performance of PCA and the Autoencoder is evaluated using **reconstruction error**.

The project also provides visual comparisons between:

1. Original images
2. Noisy images
3. PCA-reconstructed images
4. Autoencoder-reconstructed images

This makes it possible to qualitatively and quantitatively compare the two approaches.

## 🧠 Methods

| Method      | Main Idea                                                              | Implementation       |
| ----------- | ---------------------------------------------------------------------- | -------------------- |
| PCA         | Dimensionality reduction and reconstruction using principal components | NumPy + Scikit-learn |
| Autoencoder | Nonlinear feature learning and image reconstruction                    | PyTorch              |

## 🛠️ Tech Stack

* **Python**
* **PyTorch** – Building and training the Autoencoder
* **Torchvision** – Loading the Fashion MNIST dataset
* **NumPy** – Matrix operations and custom PCA implementation
* **Matplotlib** – Visualization and reconstruction error plots
* **Scikit-learn** – Standard PCA implementation
* **Torchviz** – Optional visualization of the Autoencoder architecture

