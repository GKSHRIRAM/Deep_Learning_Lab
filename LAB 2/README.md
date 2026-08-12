# Multi-Layer Perceptron — Fashion-MNIST Classification

A TensorFlow/Keras implementation of a Multi-Layer Perceptron (MLP) for multi-class image classification, built as part of the CS3807 Deep Learning Laboratory (Shiv Nadar University Chennai, B.Tech AI & DS, AY 2026–27).

The model classifies grayscale clothing images from the **Fashion-MNIST** dataset into 10 categories, with a baseline architecture compared against a model tuned via automated hyperparameter search.

## Overview

This project covers the full pipeline for a dense feed-forward image classifier:

- Dataset exploration (sample images, class distribution)
- Preprocessing (flattening 28×28 images to 784-length vectors, pixel normalization, one-hot encoding)
- Building and training a Sequential MLP in Keras
- Evaluation using accuracy, macro precision/recall/F1, confusion matrix, and a per-class classification report
- Automated hyperparameter optimization using `RandomizedSearchCV` (via SciKeras)
- Comparison of baseline vs. optimized model performance

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — a drop-in replacement for MNIST digits, consisting of grayscale images of clothing items.

| Property | Value |
|---|---|
| Training images | 60,000 |
| Testing images | 10,000 |
| Classes | 10 |
| Image size | 28 × 28 (grayscale) |

Classes: `T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`. The training set is perfectly balanced (6,000 images per class).

## Model Architecture

Baseline model — a `Sequential` Keras MLP:

```
Input(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
```

- **Loss:** categorical cross-entropy
- **Optimizer:** Adam
- **Metric:** accuracy
- **Training:** 20 epochs, batch size 32, 20% validation split
- **Total parameters:** 109,386

## Results

### Baseline Model — Test Set

| Metric | Value |
|---|---|
| Accuracy | 0.8772 |
| Precision (macro) | 0.8820 |
| Recall (macro) | 0.8772 |
| F1-score (macro) | 0.8777 |

Training accuracy reached ~92.4% and validation accuracy plateaued around ~88.3% by epoch 20, with the growing train/validation gap indicating mild overfitting. The confusion matrix shows most misclassifications occurring between visually similar categories: **Shirt, T-shirt/top, Coat, and Pullover**.

### Hyperparameter Optimization

A `RandomizedSearchCV` (via the SciKeras wrapper) was used to search over:

| Hyperparameter | Candidate Values |
|---|---|
| Hidden Layers | 1, 2, 3 |
| Hidden Neurons | 32, 64, 128, 256 |
| Learning Rate | 0.1, 0.01, 0.001 |
| Batch Size | 16, 32, 64, 128 |
| Epochs | 10, 20, 30 |
| Optimizer | SGD, Adam, RMSProp |
| Activation | ReLU, Tanh, Sigmoid |
| Dropout Rate | 0.0, 0.2, 0.5 |

Search was run on a subsampled dataset (2-fold CV, 3 epochs/candidate) due to the ~11,664-combination search space being computationally infeasible to search exhaustively; the best candidate was then retrained on the full dataset.

**Best configuration found:** 2 hidden layers × 64 neurons, Tanh activation, RMSProp (lr=0.001), no dropout.

| Metric | Baseline | Optimized |
|---|---|---|
| Accuracy | 0.8772 | 0.8744 |
| Precision (macro) | 0.8820 | 0.8775 |
| Recall (macro) | 0.8772 | 0.8744 |
| F1-score (macro) | 0.8777 | 0.8750 |
| Training time (20 epochs) | ~47s | ~100s |

The optimized model performed marginally lower than the baseline — likely because the search itself was constrained to a reduced subsample with only 3 epochs per candidate, which doesn't perfectly transfer to full-scale training.

## Requirements

```
tensorflow
scikeras
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install tensorflow scikeras numpy pandas matplotlib scikit-learn
```

## Usage

```python
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess: flatten + normalize
train_images = train_images.reshape(60000, 784) / 255.0
test_images = test_images.reshape(10000, 784) / 255.0

# Build and train the baseline MLP
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, tf.keras.utils.to_categorical(train_labels), epochs=20, batch_size=32, validation_split=0.2)
```

Run the hyperparameter search script to reproduce the `RandomizedSearchCV` results, or load the notebook to regenerate all plots (class distribution, accuracy/loss curves, confusion matrix, search results).

## Key Takeaways

- **Grid Search vs. Randomized Search:** Grid Search is exhaustive but computationally infeasible for large search spaces (11,664 combinations here); Randomized Search samples a fixed budget of combinations and finds near-optimal configurations much faster.
- **Most impactful hyperparameter:** the optimizer/learning-rate combination had the largest effect on cross-validation accuracy — SGD with a low learning rate underperformed within the limited epoch budget compared to RMSProp or Adam.
- **Optimization didn't beat the baseline** in this run, illustrating a practical limitation: rankings found on a reduced subsample with fewer epochs don't always transfer cleanly to full-scale training.
- **Recommended configuration:** the original baseline (`Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)` with Adam) remains the best-performing configuration actually evaluated at full scale, though Tanh + RMSProp is a promising alternative worth further exploration.

## References

1. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
2. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
3. S. Haykin, *Neural Networks and Learning Machines*, Pearson, 2009.
4. Xiao, H., Rasul, K., Vollgraf, R., "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms," 2017.
5. [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
6. [SciKeras Documentation](https://www.adriangb.com/scikeras/)

## Author

CS3807 Deep Learning Laboratory — Experiment 2
B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
