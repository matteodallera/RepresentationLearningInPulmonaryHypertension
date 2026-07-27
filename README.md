# Representation Learning In Pulmonary Hypertension

This repository contains the code required to implement the framework proposed in the paper "Representation Learning for Pulmonary Hypertension Subtypes: Task-Driven Biases vs. Vision Foundation Models"

Abstract. - Coming Soon -

---

## 📂 Repository Structure

### 🔹 Preprocessing

- This folder contains the script needed to peform the task-driven preprocessing of data; preprocessing needs the usage of the Pyhton Library TotalSegmentator (Please refer to the official TotalSegmentator GitHub repository for installation details and further information)
- taskprocessingS.py allows the task-driven processing of the proprietary dataset
- taskprocessingR.py allows the task-driven processing of the RSNA dataset
- processingS.py allows the processing of the proprietary dataset used for the variants of the task-driven framework that do not use task-driven preprocessing 
- processingR.py allows the processing of the RSNA dataset used for the variants of the task-driven framework that do not use task-driven preprocessing 
-----------

### 🔹 Training task-driven framework
- The folder contains the scripts needed for the training of the task-driven-processing (TraningMG.py) which needs the training data and the validation data.
- To zero-gate the skip connection, flag need to be set equal to 0 in Unetmod.py
---

### 🔹 Computation of the embeddings 
- **`Inference*.py`**
allows the computation of the representations being compared; the input are the volumes used for the comparison of the embeddings while
the output is a 2D-matrix in .xlsx format containing patients as rows and the features as columns. the last column is the class label
------

### 🔹Fourier analysis of the skip connection 
- **`FFT_skip_connection.py`** FFT analysis of the role of the skip connection in the analysis
-------

### 🔹Evaluation Benchmarks
- Folder contains the scipts needed for the comparison of the latent representations (the .xlsx files of the different models are needed)


## 📌 Notes 
