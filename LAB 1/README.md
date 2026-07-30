# Single Layer Perceptron — Banknote Authentication

A from-scratch implementation of a Single Layer Perceptron for binary classification, built as part of the CS3807 Deep Learning Laboratory (Shiv Nadar University Chennai, B.Tech AI & DS, AY 2026–27).

The perceptron is trained to distinguish authentic vs. forged banknotes using the [UCI Banknote Authentication Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication), and its results are benchmarked against scikit-learn's `Perceptron`.

## Overview

This project implements the classical perceptron learning algorithm end-to-end — weighted sum, step activation, and the perceptron weight/bias update rule — without relying on any machine learning framework for the core model. It covers:

- Exploratory data analysis (histograms, correlation heatmap, scatter plot, boxplots)
- Data preprocessing (standardization, train/test split)
- A from-scratch `Perceptron` class
- Training with error tracking across epochs
- Evaluation (accuracy, precision, recall, F1-score, confusion matrix)
- Learning rate comparison (0.01, 0.1, 0.5)
- Comparison against scikit-learn's `Perceptron`
- Discussion of why a single layer perceptron cannot solve XOR

## Dataset

| Property | Value |
|---|---|
| Instances | 1372 |
| Features | 4 (Variance, Skewness, Curtosis, Entropy — from Wavelet Transform of banknote images) |
| Classes | 2 (0 = Authentic, 1 = Forged) |
| Missing values | None |

## Model

The perceptron uses the classical **step activation function**:

```
f(z) = 1 if z ≥ 0
f(z) = 0 if z < 0
```

Weights and bias are updated using the perceptron learning rule:

```
w_new = w_old + η(y − ŷ)x
b_new = b_old + η(y − ŷ)
```

where `η` is the learning rate.

## Results

Trained for 20 epochs, learning rate = 0.1:

| Metric | Scratch Perceptron | Scikit-learn Perceptron |
|---|---|---|
| Accuracy | 0.9818 | 0.9891 |
| Precision | 0.9593 | 0.9832 |
| Recall | 1.0000 | 0.9915 |
| F1-score | 0.9793 | 0.9873 |

Training error dropped from 52 misclassifications in epoch 1 to 15 by epoch 20, showing steady convergence. All three tested learning rates (0.01, 0.1, 0.5) converged to the same final test accuracy, differing mainly in the smoothness of the error curve.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication) and place it as `data_banknote_authentication.txt` in the project directory.
2. Run the notebook/script to reproduce EDA plots, train the perceptron, and generate evaluation metrics.

```python
from perceptron import Perceptron

p = Perceptron([0, 0, 0, 0], 0)
p.backwardpass(X_train[i], Y_train[i], learning_rate=0.1)
prediction = p.predict(X_test[i])
```

## Key Takeaways

- Feature standardization is essential for stable perceptron convergence when features are on very different scales.
- The banknote dataset is close to linearly separable, which is why a simple perceptron performs so well on it (~98% accuracy).
- The step activation function has no usable gradient, which is why it cannot be used in gradient-based (backpropagation) training — motivating smooth activations like Sigmoid, Tanh, and ReLU.
- A single layer perceptron cannot solve non-linearly-separable problems like XOR; this motivates the multilayer perceptron.

## References

1. F. Rosenblatt, "The Perceptron," *Psychological Review*, 1958.
2. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
3. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
4. S. Haykin, *Neural Networks and Learning Machines*, Pearson, 2009.
5. UCI Machine Learning Repository — Banknote Authentication Dataset.
6. [Scikit-learn Documentation: Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)

## Author

CS3807 Deep Learning Laboratory — Experiment 1
B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
