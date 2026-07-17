# Representation Learning In Pulmonary Hypertension

This repository contains the code required to implement the framework proposed in the paper "Representation Learning for Pulmonary Hypertension Subtypes: Task-Driven Biases vs. Vision Foundation Models"

Abstract. - Coming Soon -

---

## 📂 Repository Structure

### 🔹 Task-Driven-Framework 

- This folder contains the script needed to peform the task-driven preprocessing of data (task-driven-preprocessing folder); preprocessing needs the usage of the Pyhton Library TotalSegmentator (Please refer to the official TotalSegmentator GitHub repository for installation details and further information)
- Preprocessing needs the source input folder of the volumes and the output folder for the processed ones.
- The folder also contains the script with the modified version of the 3D-Unet (Unetmod.py) and the script to be used for the training of the task-driven-framework 
---

### 🔹 Computation of the embeddings 
- **`Inference*.py`**
The script named inference*.py allow the computation of the representations being compared; the input are the volumes used for the comparison of the mebeddings while
the output is a 2D-matrix in .xlsx format containing patients as rows and the features as columns. the last column is the class label
---

### 🔹Fourier analysis of the skip connection 
- **`FFT_skip_connection.py`** FFT analysis of the role of the skip connection in the analysis
---



## 📌 Notes 
