# 🌌 Galaxy Morphology Classification — CNN From Scratch (NumPy)

This project implements a Convolutional Neural Network **from scratch using NumPy only** to classify galaxy morphologies from the Galaxy Zoo dataset.

No PyTorch.  
No TensorFlow.  
No high-level deep learning frameworks.

All convolution, backpropagation, pooling, and optimization were implemented manually.
Dataset uses:Galaxy Zoo Dataset (link:https://www.kaggle.com/code/ansuman30sahu/galaxy-zoo-dataset)
---

## 📌 Project Overview

The goal is to classify galaxy images into 5 morphology categories:

- Cigar-shaped smooth
- Completely round smooth
- In between smooth and spiral
- Edge-on
- Spiral

The model was trained on 64×64 RGB images.

---

## 🧠 Architecture

### CNN Feature Extractor

Input: `64 × 64 × 3`
Conv2D (`3 → 16`, kernel=`3x3`, padding=`1`)
ReLU
MaxPool (`2x2`)

Conv2D (`16 → 32`, kernel=`3x3`, padding=`1`)
ReLU
MaxPool (`2x2`)

Conv2D (`32 → 64`, kernel=`3x3`, padding=`1`)
ReLU
MaxPool (`2x2`)

Final feature map size:

`64 × 8 × 8` → Flatten → `4096` features

### MLP Classifier

`4096 → 1024 → 128 → 5`
ReLU activations
Softmax output
Cross-entropy loss

## ⚙️ Implementation Details

### ✔ Custom Convolution Implementation
- Manual forward propagation
- Manual backward propagation
- Gradient computation for:
  - Weights
  - Bias
  - Input tensor

### ✔ im2col Optimization
Convolution was optimized using the **im2col / col2im** technique, converting convolution into efficient matrix multiplication.

This reduced training time from minutes per epoch to seconds per epoch.

### ✔ Custom Modules Implemented
- Conv2D
- MaxPool
- ReLU
- Flatten
- Fully Connected Layers
- Cross Entropy Loss
- Mini-batch Gradient Descent

---

## 📊 Training Results

Dataset size: **2000 images (64×64)**  
Train/Test Split: 80/20  

Final Accuracy:

- Training Accuracy: **84%**
- Test Accuracy: **84%**

The model shows strong generalization with minimal overfitting.

---

## 🧮 Parameter Count

Total parameters ≈ ~600k

Most parameters are in the MLP layer, while convolution layers remain lightweight.

---

## 🔍 What This Project Demonstrates

- Deep understanding of convolution mathematics
- Backpropagation derivation and implementation
- Memory optimization using im2col
- Practical ML pipeline development
- Architecture experimentation
- Overfitting analysis



---

## 🚀 Future Improvements

- Add Batch Normalization
- Add Dropout
- Data augmentation (rotation invariance for galaxies)
- Grad-CAM visualization
- Compare 32×32 vs 64×64 performance
- Train on full 29k dataset

---
## 📂 Files

- CNN_mod->contains the CNN forward propagation backward propagationa and all other related fucntions inclusing training and testiing fucntions
- mlp->contains the code for the mlp which is also one of my custom made proejct it contaisn its own forward and backward propagaion and training fucntions
- data_load->Contains code to randomly select data from dataset and provide the training and testing set
## 📂 How to Run

1. Install dependencies:
   ```bash
   pip install numpy opencv-python matplotlib seaborn
2. Run Training
```bash
  python CNN_mod.py
```
## 📚 Learning Outcome

This project was built to deeply understand:

How convolution actually works

How gradients flow in CNNs

Why im2col makes convolution efficient

How architecture design affects generalization

## Output
<img width="360" height="360" alt="image" src="https://github.com/user-attachments/assets/f2606703-7ea5-44d7-b2bf-51cd578b47e6" />
<img width="360" height="360" alt="image" src="https://github.com/user-attachments/assets/5f1d1529-8af0-4d74-a1fe-2ae7c56b44cc" />
<img width="360" height="360" alt="image" src="https://github.com/user-attachments/assets/615c4585-0bad-441f-b7ca-0ce9a35d552e" />
<img width="360" height="360" alt="image" src="https://github.com/user-attachments/assets/1ed45865-3d6f-4d32-aeb9-acceefd266d4" />

## 📞 Contact Me
Email:yashasvi21022005@gmail.com

Linkedin:https://www.linkedin.com/in/yashasvi-kumar-tiwari/

