# Few-Shot kNN Classification with SSL Visual Features

## Overview
This repository contains the codebase for evaluating and comparing small self-supervised pretrained visual backbones as frozen feature extractors for low-shot image classification. Instead of training neural networks end-to-end, this project evaluates the intrinsic quality of features extracted from models like DINOv2 by using a non-parametric k-nearest-neighbors (kNN) classifier. 

We evaluate these frozen representations on the CIFAR-10 and CIFAR-100 datasets across varying sizes of labeled examples per class (N-shot learning), ranging from one-shot up to the full dataset.

## Supported Models
This repository is designed to evaluate a lineage of prominent self-supervised visual learning models:
*   **DINOv1** (ViT-S/16 or ViT-S/8)
*   **DINOv2** (ViT-S/14, with optional registers ablation)
*   **DINOv3** (ViT-S/16 distilled)
*   **AttMask** (ViT-S/16, 300-epoch High)
*   **iBOT** (ViT-S/16 teacher checkpoint)
*   **MoCo v3** (ViT-Small, 300 epochs)

### Baselines
*   **Fully Supervised ResNet Baseline**: Scripts are provided in `resnet_baseline/` to train and evaluate a fully supervised ResNet model from scratch, or to use it as a feature extractor. This provides a traditional supervised benchmark against the SSL-pretrained features.

## Datasets
*   **CIFAR-10:** 10 classes, evaluated with N ∈ {1, 2, 5, 10, 20, 50, 100, 500, full}
*   **CIFAR-100:** 100 classes, evaluated with N ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500}

Images are resized from their native 32x32 resolution to the expected 224x224 input size using bicubic interpolation before feature extraction.

## Experimental Protocol
The experimental workflow follows a strict **"extract once, probe many"** pattern to allow rapid experimentation without GPU overhead after the initial feature extraction step.

1.  **Feature Extraction (`src/extract.py`)**:
    *   Forward pass on the datasets using a frozen backbone.
    *   Extracts the `[CLS]` token (or mean-pooled patch tokens).
    *   Applies L2-normalization to map features to a unit hypersphere.
    *   Caches the resulting tensors to disk in the `features/` directory.
2.  **kNN Classification (`src/evaluate_knn.py`)**:
    *   Uses Cosine Distance (or Euclidean distance post-normalization).
    *   Runs $k$-NN with distance-weighted voting (k ∈ {1, 5, 20}).
    *   Averages predictions over at least 3 random seeds for low-shot sampling.
    *   Computes Top-1 Accuracy (primary) and Macro-F1 (secondary).

## Setup & Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Note: PyTorch, torchvision, and optionally FAISS or scikit-learn are required. You might also need the `timm` library for model loading.)*

## Usage

The project provides a unified Command Line Interface (CLI) for running the full pipeline based on the settings in `configs/default.yaml`.

### 1. Extract Features (GPU Recommended)
Run the extraction script to generate `.pt` feature files. This step downloads the datasets, processes them through the frozen models, and saves the output to the `features/` directory.

```bash
python src/cli.py extract
```

### 2. Evaluate with kNN (CPU)
Run the kNN evaluation grid across the extracted features. This step evaluates the N-shot classification tasks over several random seeds and saves the results to `results/results.csv`.

```bash
python src/cli.py evaluate
```

Alternatively, to run a single quick test on a preset $k$:
```bash
python run_knn.py
```

## Generating Results & Charts

After the evaluation grid completes, generate comparative Accuracy-vs-Shots curves evaluating low-shot data efficiency and differences between model architectures. Plots are saved as PNG files in `results/figures/`.

```bash
python src/cli.py plot
```

You can specify plotting arguments to visualize different pooling modes or $k$ neighbors:
```bash
python src/cli.py plot --k 5 --pooling cls
```

## Fully Supervised Baselines

To compare SSL representations against a traditional fully supervised model, train and evaluate the ResNet baseline from scratch:

```bash
# 1. Train the ResNet baseline
python resnet_baseline/train.py --epochs 50

# 2. Extract features using the trained baseline
python resnet_baseline/extract.py

# 3. Evaluate kNN on the baseline features
python resnet_baseline/evaluate_knn.py
```
