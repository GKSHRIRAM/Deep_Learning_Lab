# Convolutional Neural Network — CIFAR-10 Image Classification

A TensorFlow/Keras implementation of a Convolutional Neural Network (CNN) for multi-class image classification, built as part of the CS3807 Deep Learning Laboratory (Shiv Nadar University Chennai, B.Tech AI & DS, AY 2026–27).

The model classifies colour images from the **CIFAR-10** dataset into 10 categories, covering the convolution operation, pooling, feature map visualization, and full CNN training/evaluation from the ground up.

## Overview

This project covers the full pipeline for a CNN-based image classifier:

- Dataset exploration (sample images, class distribution)
- Manual/numerical walkthroughs of convolution, max pooling, and parameter counting
- Convolution experiments — effect of kernel size, stride, and padding on output shape
- Feature map visualization after a convolution layer
- Max pooling vs. average pooling comparison
- Building and training a two-convolution-layer CNN in Keras
- Evaluation using accuracy, macro precision/recall/F1, confusion matrix, and a per-class classification report
- Additional exercises: output-size/parameter formulas, ReLU vs. Sigmoid, pooling-type comparison, filter-count comparison

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — small colour photographs of real-world objects and animals.

| Property | Value |
|---|---|
| Training images | 50,000 |
| Testing images | 10,000 |
| Classes | 10 |
| Image size | 32 × 32 × 3 (RGB) |

Classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`. The training set is perfectly balanced (5,000 images per class).

## Model Architecture

Final model — a `Sequential` Keras CNN:

```
Input(32,32,3) → Conv2D(32, 3×3, ReLU, same) → MaxPool(2×2)
              → Conv2D(64, 3×3, ReLU, same) → MaxPool(2×2)
              → Flatten → Dense(64, ReLU) → Dense(10, Softmax)
```

- **Loss:** categorical cross-entropy
- **Optimizer:** Adam
- **Metric:** accuracy
- **Training:** 20 epochs, batch size 32, 10% validation split
- **Total parameters:** 282,250

## Results

### Final CNN — Test Set

| Metric | Value |
|---|---|
| Accuracy | 0.6799 |
| Precision (macro) | 0.6891 |
| Recall (macro) | 0.6799 |
| F1-score (macro) | 0.6834 |

Training accuracy reached ~91.7% by epoch 20, while validation accuracy plateaued around ~69–71% after epoch 6, with the growing train/validation gap indicating overfitting — confirmed by validation loss rising again after epoch 6–7 even as training loss kept dropping. The confusion matrix shows most misclassifications occurring between visually similar categories: **cat and dog**, and to a lesser extent **bird and deer**.

### Convolution & Pooling Experiments

| Experiment | Key Finding |
|---|---|
| Kernel size (3×3, 5×5, 7×7) | Every +2 increase in kernel size shrinks output spatial dims by 2 on each side (valid padding) |
| Stride (1 vs 2) | Stride 2 roughly halves the output spatial dimensions vs. stride 1 |
| Padding (`same` vs `valid`) | `same` preserves input size; `valid` shrinks output based on kernel size |
| Max pooling vs. average pooling | Max pooling preserves the strongest activations; average pooling smooths/blurs them — max pooling reached higher validation accuracy (0.6976 vs 0.6742 at 5 epochs) |
| Filters (16 vs 64) | Widening from 16 → 64 filters improved accuracy by ~4.3 points (0.5672 → 0.6099) for a modest time increase (17.29s → 18.90s) |

### Additional Exercises

| Exercise | Result |
|---|---|
| Output size (N=64, F=5, P=2, S=2) | 32 |
| Parameters (64 filters, 3×3, RGB input) | 1,792 |
| ReLU vs. Sigmoid | ReLU avoids saturation/vanishing gradients, making it the default for hidden layers; Sigmoid is mostly reserved for binary output layers |

## Requirements

```
tensorflow
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Usage

```python
import tensorflow as tf
from tensorflow.keras import layers

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess: normalize + one-hot encode
x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# Build and train the CNN
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_norm, y_train_cat, epochs=20, batch_size=32, validation_split=0.1)
```

Run the notebook to reproduce all plots (sample images, class distribution, kernel/stride/padding experiments, feature map visualizations, pooling comparisons, accuracy/loss curves, and the confusion matrix).

## Key Takeaways

- **Convolution vs. fully connected layers:** convolution uses small, shared kernels instead of one weight per pixel, drastically cutting parameter count and exploiting translation invariance — the two conv layers here used only 19,392 parameters vs. 196,000+ for an equivalent dense layer.
- **Stride and padding control output size:** stride skips positions and shrinks the feature map; `same` padding keeps spatial size constant while `valid` padding lets it shrink, both verified directly against the output-size formula.
- **Pooling type matters:** max pooling retains the strongest activations and outperformed average pooling in this run, though both reduce spatial size and computation identically.
- **More filters help, cheaply:** going from 16 to 64 filters gave a meaningful accuracy boost for only a small increase in training time, at least at this shallow depth.
- **Overfitting emerged by epoch 20:** the diverging training/validation curves suggest early stopping or dropout would improve generalization on this architecture.

## References

1. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
2. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
3. S. Haykin, *Neural Networks and Learning Machines*, Pearson, 2009.
4. [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
5. [CIFAR-10 Dataset Documentation](https://www.cs.toronto.edu/~kriz/cifar.html)

## Author

CS3807 Deep Learning Laboratory — Experiment 3
B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
