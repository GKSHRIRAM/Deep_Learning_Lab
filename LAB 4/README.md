# Experiment 4 – Comparative Study of Deep CNN Architectures Using Transfer Learning

**Course:** CS3807 – Deep Learning Laboratory
**Program:** B.Tech Artificial Intelligence & Data Science, Semester V
**Institution:** Shiv Nadar University Chennai

## Overview

This experiment studies the evolution of deep convolutional neural network architectures (LeNet-5, AlexNet, VGG16, GoogLeNet, ResNet) and implements transfer learning using a pretrained CNN fine-tuned on the CIFAR-10 dataset.

## Contents

| File | Description |
|---|---|
| `Experiment_4.ipynb` | Jupyter notebook with full implementation |
| `Experiment_4.tex` | Lab report source (LaTeX) |

## Objectives

- Study the evolution of deep CNN architectures
- Compare LeNet-5, AlexNet, VGG16, GoogLeNet and ResNet
- Understand and implement transfer learning
- Fine-tune a pretrained CNN model
- Compare classification performance across architectures

## Dataset

**CIFAR-10** — 60,000 32×32×3 color images across 10 classes (50,000 train / 10,000 test).

## Approach

1. **Dataset preparation** – load CIFAR-10, normalize pixel values, visualize samples
2. **Transfer learning** – load a pretrained ImageNet model (MobileNetV2 by default; VGG16/ResNet50 also supported), remove the top classifier, freeze the convolutional base, and add a GlobalAveragePooling → Dense(ReLU) → Dense(Softmax) head
3. **Training** – Adam optimizer, lr = 0.001, batch size = 32, 10 epochs, categorical cross-entropy loss
4. **Fine-tuning** – unfreeze the last convolutional block, train for 5 more epochs at a lower learning rate (1e-5)
5. **Evaluation** – accuracy, precision, recall, F1-score, confusion matrix, classification report

## Requirements

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage

```bash
jupyter notebook Experiment_4.ipynb
```

Run all cells top to bottom. GPU is recommended but not required.

## Results

Fill in after running:

| Metric | Value |
|---|---|
| Training Accuracy | |
| Testing Accuracy | |
| Precision | |
| Recall | |
| F1-score | |

## References

- LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, 1998
- Krizhevsky et al., *ImageNet Classification with Deep CNNs*, NeurIPS 2012
- Simonyan & Zisserman, *Very Deep Convolutional Networks*, ICLR 2015
- Szegedy et al., *Going Deeper with Convolutions*, CVPR 2015
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
