# Representation Learning In Pulmonary Hypertension

This repository contains the code required to implement the framework proposed in the paper "Representation Learning for Pulmonary
Hypertension Subtypes Identification:Comparison of Task-Driven and Vision Foundation Modelss"

Accurate discrimination between Pulmonary
Arterial Hypertension (PAH) and Chronic Thromboembolic
Pulmonary Hypertension (CTEPH) using Computed
Tomography Pulmonary Angiography (CTPA) is a critical
diagnostic challenge characterized by low disease
prevalence and high inter-reader variability. While generalist
Vision Foundation Models (FMs) have recently gained
prominence in medical imaging, their pretraining and
architecture are not explicitly designed to encode domainspecific clinical knowledge. In this work, we investigate
whether domain-specific inductive biases incorporated into
a 3D Autoencoder can yield competitive representations with
state-of-the-art FMs (Curia and TAP-CT) for PH subtype
classification. We evaluate two decoupled inductive biases:
a data-level bias via task-driven preprocessing (P) and an
architectural bias introducing a learnable, high-pass filtered
skip connection (A). These inductive biases are
systematically evaluated across four model variants (𝑩𝑺,
𝑩𝑺𝑷, 𝑩𝑺𝑨, 𝑩𝑺𝑷𝑨). Evaluated on 183 CTPA scans across
linear probing, non-linear classification capacity, and latent
topology analysis, variants incorporating P achieved
superior performance in terms of separability, while
structuring a feature space better aligned with the clinical
task. These findings demonstrate that task-tailored
architectures enriched with clinical knowledge can
outperform generalist foundation models in specialized
diagnostic applications, combining superior representation
quality with full data transparency

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
