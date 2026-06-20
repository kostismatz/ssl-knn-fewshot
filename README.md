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

### 1. Extract Features (GPU Recommended)
Run the extraction script to generate `.pt` feature files. This step downloads the dataset, processes it through the frozen model, and saves the output to the `features/` directory.

```bash
python -m src.extract --device cuda --batch_size 16
```

### 2. Evaluate with kNN (CPU)
Run the kNN evaluation grid across the extracted features. This step evaluates the N-shot classification tasks over several random seeds.

```bash
python -m src.evaluate_knn
```

Alternatively, to run a single test on a preset $k$:
```bash
python run_knn.py
```

## Results & Plotting
The `evaluate_knn.py` script will output the mean accuracies and plot an Accuracy-vs-Shots curve for the selected dataset and model. All plotting logic handles the generation of comparative figures evaluating low-shot data efficiency and differences between model architectures.
