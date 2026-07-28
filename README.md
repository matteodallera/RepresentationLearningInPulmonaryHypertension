# Representation Learning In Pulmonary Hypertension

This repository contains the code required to implement the framework proposed in the paper "Representation Learning for Pulmonary Hypertension Subtypes: Task-Driven Biases vs. Vision Foundation Models"

Abstract. - Coming Soon -

# Model Training & Evaluation Pipeline

Repository for preprocessing, model training on public datasets, inference on proprietary data, and model benchmarking.

---

## 📁 Repository Structure


## 📌 Modules Overview

### 1. Training (`/Training`)
* **Preprocessing & Task-driven Preprocessing**: Prepares public dataset inputs.
* **Training**: Network architectures (`unetmod.py`) and training scripts [To disable the retained Skip Connection, please set flag = 0 in un unetmod.py]

### 2. Evaluation (`/Evaluation`)
* **Inference & Proprietary Data Processing**: Handles proprietary dataset preprocessing and executes inference across models. Generates `.xlsx` spreadsheets containing patient rows and extracted feature columns.
* **Benchmarks**:
  * **Linear Separability**: Linear analysis of the feature space.
  * **Non-linear Classification**: MLP probing and cross-validation scripts.
  * **2D-UMAP**: Dimensionality reduction and latent space visualization.

### 🔹Evaluation Benchmarks
- Folder contains the scipts needed for the comparison of the latent representations (the .xlsx files of the different models are needed)


## 📌 Notes 
