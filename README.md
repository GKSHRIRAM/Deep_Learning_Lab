# CS3807 – Deep Learning Laboratory

Lab reports and notebooks for the Deep Learning Laboratory (CS3807), B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai, AY 2026–27.

Each experiment includes the Jupyter notebook, the compiled LaTeX report (PDF + `.tex` source), and all generated plots.

## Experiments

| # | Title | Dataset | Notebook | Report |
|---|-------|---------|----------|--------|
| 1 | Single Layer Perceptron for Binary Classification | Banknote Authentication (UCI) | [`LAB1.ipynb`](./experiment-1/LAB1.ipynb) | [`report.pdf`](./experiment-1/report.pdf) |
| 2 | Multi-Layer Perceptron (MLP) for Multi-Class Image Classification | Fashion-MNIST | [`experiment_2.ipynb`](./experiment-2/experiment_2.ipynb) | [`report3.pdf`](./experiment-2/report3.pdf) |
| 3 | Convolutional Neural Networks (CNNs) for Image Classification | CIFAR-10 | [`Experiment_3_CNN_final.ipynb`](./experiment-3/Experiment_3_CNN_final.ipynb) | [`report4.pdf`](./experiment-3/report4.pdf) |

## Repository Structure

```
.
├── experiment-1/
│   ├── LAB1.ipynb
│   ├── report.tex
│   ├── report.pdf
│   └── plots/
├── experiment-2/
│   ├── experiment_2.ipynb
│   ├── report3.tex
│   ├── report3.pdf
│   └── plots/
├── experiment-3/
│   ├── Experiment_3_CNN_final.ipynb
│   ├── report4.tex
│   ├── report4.pdf
│   └── plots/
└── README.md
```

## Summary of Results

| Experiment | Model | Test Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| 1 – Perceptron | Scratch-built Perceptron | 0.9818 | 0.9593 | 1.0000 | 0.9793 |
| 1 – Perceptron | Scikit-learn Perceptron | 0.9891 | 0.9832 | 0.9915 | 0.9873 |
| 2 – MLP | Baseline (128→64→10) | 0.8713 | 0.8745 | 0.8713 | 0.8693 |
| 2 – MLP | Optimized (RandomizedSearchCV) | 0.8696 | 0.8689 | 0.8696 | 0.8684 |
| 3 – CNN | Conv→Pool→Conv→Pool→Dense | 0.6799 | 0.6891 | 0.6799 | 0.6834 |

## Tech Stack

- **Python 3**, **TensorFlow / Keras**, **scikit-learn**, **SciKeras**
- **NumPy**, **pandas**, **matplotlib**, **seaborn**
- **LaTeX** (reports compiled with `pdflatex`)

## Building the Reports

Each `report*.tex` file is self-contained (assumes the sibling `plots/` folder is present) and compiles with a standard TeX distribution:

```bash
pdflatex report.tex
pdflatex report.tex   # run twice for references/ToC to resolve
```

## Datasets

- **Experiment 1:** [Banknote Authentication Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication) — UCI Machine Learning Repository
- **Experiment 2:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — Zalando Research
- **Experiment 3:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — Krizhevsky, 2009

## References

1. F. Rosenblatt, "The Perceptron," *Psychological Review*, 1958.
2. I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning*, MIT Press, 2016.
3. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
4. S. Haykin, *Neural Networks and Learning Machines*, Pearson, 2009.
5. Xiao, H., Rasul, K., Vollgraf, R., "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms," 2017.
6. TensorFlow/Keras Documentation, <https://www.tensorflow.org/api_docs>
7. Scikit-learn Documentation, <https://scikit-learn.org/stable/>

## Author

B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
