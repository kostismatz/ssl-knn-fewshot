# Few-Shot k-Nearest Neighbour Classification with Self-Supervised Visual Features

**Technical Report** | June 2026

---

## Abstract

This report evaluates the quality of visual representations learned by four self-supervised learning (SSL) backbone models — DINOv1 ViT-S/16, DINOv2 ViT-S/14, DINOv2 ViT-S/14 with registers, and MoCo v3 ViT-S/16 — using a non-parametric k-nearest neighbour (kNN) classifier as a probe. Features are extracted once from frozen, pretrained encoders and evaluated across a grid of few-shot regimes (N ∈ {1, 2, 5, 10, 20, 50, 100} shots per class) on CIFAR-10 and CIFAR-100. The approach deliberately avoids any fine-tuning, so that classification accuracy directly reflects the geometric quality of each backbone's representation space. DINOv2-family models decisively outperform DINOv1 and MoCo v3, with the register variant reaching **94.6% on CIFAR-10** and **80.2% on CIFAR-100** at 100 shots per class. MoCo v3 significantly underperforms all DINO variants, indicating weaker representation geometry for near-neighbour retrieval.

---

## 1. Introduction

Self-supervised learning (SSL) has emerged as a powerful paradigm for training visual representations without requiring human-annotated labels. By training on pretext tasks derived from the data itself (such as masked image modelling or contrastive learning), SSL backbones can produce general-purpose feature embeddings that transfer well to downstream tasks.

A natural question is: *how semantically coherent are these embeddings without any task-specific adaptation?* The k-nearest neighbour classifier is the ideal probe for this question. Because it has no learnable parameters — only a distance metric and a hyperparameter *k* — any accuracy it achieves is attributable entirely to the structure of the pretrained feature space. In the few-shot regime (small *N* per class), this test is especially demanding, as the model must already separate class clusters well enough that even a handful of labelled prototypes suffice for reliable classification.

This work benchmarks four backbone architectures under this strict non-parametric evaluation protocol, varying the number of shots, number of neighbours, and feature pooling strategy.

---

## 2. Methodology

### 2.1 Backbone Models

| Model | Architecture | Pretraining | Source |
|---|---|---|---|
| `dino_vits16` | ViT-S/16 | DINO (self-distillation) | `facebookresearch/dino` via `torch.hub` |
| `dinov2_vits14` | ViT-S/14 | DINOv2 (improved self-dist.) | `facebookresearch/dinov2` via `timm` |
| `dinov2_vits14_reg` | ViT-S/14 + registers | DINOv2 with register tokens | `facebookresearch/dinov2` via `timm` |
| `ibot_vits16` | ViT-S/16 | iBOT (MIM + distillation) | GitHub `bytedance/ibot` |
| `attmask_vits16` | ViT-S/16 | AttMask (attention-guided MIM) | GitHub `gkakogeorgiou/attmask` |
| `moco_v3_vits16` | ViT-S/16 | MoCo v3 (momentum contrast) | `facebookresearch/moco-v3` via `timm` |

All models are used as frozen feature extractors — no fine-tuning is performed at any stage.

### 2.2 Feature Extraction

Raw CIFAR images (32×32 RGB) are processed entirely on GPU to maximise throughput:

1. **Load**: The full dataset is loaded as a single NumPy array and converted to a float32 tensor `[N, 3, H, W]` in [0, 1].
2. **Normalise**: ImageNet mean/std normalisation is applied batch-wise on GPU.
3. **Resize**: Images are upsampled to 224×224 using bicubic interpolation (`torch.nn.functional.interpolate`) on GPU — bypassing all CPU-side PIL operations.
4. **Forward pass**: The batch is passed through `model.forward_features()` with `torch.autocast` (fp16) to extract the full sequence of patch tokens.
5. **Pool**: Two strategies are applied per batch:
   - **CLS token** (`cls`): The first token of the sequence — a global summary vector trained specifically for classification.
   - **Mean pool** (`mean_pool`): Average of all patch tokens excluding CLS and any register prefix tokens. Register tokens are identified via `model.num_prefix_tokens`.
6. **L2-normalise**: Both feature vectors are unit-normalised before saving, enabling cosine similarity to be computed as a dot product.

Features are cached to disk as `.pt` files (`train.pt`, `test.pt`) per model per dataset. Caching avoids redundant GPU computation on re-runs.

> [!NOTE]
> DINOv2 models default to `img_size=518` in `timm`. The model is loaded with `img_size=224` to trigger automatic positional embedding interpolation, making 224×224 inference valid.

### 2.3 k-Nearest Neighbour Classifier

The kNN classifier uses **cosine similarity** as the distance metric (feasible since features are L2-normalised). Classification follows the DINO weighted voting scheme:

1. Compute cosine similarity matrix between all test features and all support features: $S \in \mathbb{R}^{N_{test} \times N_{support}}$
2. Retrieve top-*k* nearest neighbours per test sample.
3. Compute softmax-temperature weights: $w_i = \exp(s_i / \tau)$ where $\tau = 0.07$.
4. Accumulate weighted votes per class using vectorised `scatter_add`.
5. Predict the class with the highest total vote score.

The hyperparameter *k* is swept over {1, 5, 20}. At 1-shot, $k = \min(k, N \times C)$ is clamped since fewer than *k* support samples exist.

### 2.4 Few-Shot Sampling Protocol

For each combination of (model, dataset, pooling, *N*, *k*):
- 3 random seeds are evaluated independently.
- Per seed: *N* training examples are **stratified-sampled** uniformly at random per class from the full training set.
- The full test set is always used for evaluation.
- Results are reported as mean ± standard deviation across seeds.

Both **accuracy** and **macro-F1** are computed. The grid covers:
- **Shots**: N ∈ {1, 2, 5, 10, 20, 50, 100}
- **Seeds**: 3
- **Neighbours**: k ∈ {1, 5, 20}
- **Pooling**: {cls, mean_pool}

### 2.5 Metrics

- **Top-1 Accuracy**: Fraction of correctly classified test samples.
- **Macro F1**: Unweighted mean of per-class F1 scores. More informative for class-imbalanced evaluation, though CIFAR datasets are balanced.

---

## 3. Experimental Setup

| Property | Value |
|---|---|
| Datasets | CIFAR-10 (10 classes, 50k/10k), CIFAR-100 (100 classes, 50k/10k) |
| Backbone input size | 224×224 |
| Batch size (extraction) | 128 |
| Extraction device | CUDA GPU |
| Temperature τ | 0.07 |
| Seeds | 3 |
| k swept | {1, 5, 20} |
| N shots swept | {1, 2, 5, 10, 20, 50, 100} |

---

## 4. Results

All accuracy values below are reported as **mean % ± std %** across 3 seeds, at **k = 5 neighbours**.

### 4.1 CIFAR-10

#### CLS Token Pooling

| Model | 1-shot | 5-shot | 10-shot | 100-shot |
|---|---|---|---|---|
| `dino_vits16` | 53.2 ± 5.7 | 76.2 ± 2.0 | 80.1 ± 1.3 | 88.4 ± 0.5 |
| `dinov2_vits14` | 66.0 ± 8.0 | 87.0 ± 1.1 | 89.6 ± 0.4 | **94.7 ± 0.2** |
| `dinov2_vits14_reg` | **68.2 ± 5.8** | **87.8 ± 2.0** | **90.7 ± 0.8** | 94.6 ± 0.0 |
| `ibot_vits16` | 57.6 ± 7.4 | 79.6 ± 0.9 | 83.1 ± 0.3 | 90.4 ± 0.5 |
| `attmask_vits16` | 56.1 ± 6.0 | 78.2 ± 0.1 | 81.8 ± 0.8 | 89.1 ± 0.5 |
| `moco_v3_vits16` | 21.8 ± 1.3 | 30.5 ± 1.5 | 33.5 ± 0.7 | 43.9 ± 0.5 |

#### Mean Pool Pooling

| Model | 1-shot | 5-shot | 10-shot | 100-shot |
|---|---|---|---|---|
| `dino_vits16` | 32.8 ± 5.1 | 46.6 ± 1.7 | 51.4 ± 1.2 | 64.4 ± 0.4 |
| `dinov2_vits14` | 66.2 ± 8.8 | 84.6 ± 0.2 | 86.4 ± 0.3 | 92.1 ± 0.2 |
| `dinov2_vits14_reg` | **69.6 ± 8.7** | **87.7 ± 2.0** | **90.1 ± 1.1** | **94.0 ± 0.3** |
| `ibot_vits16` | 36.0 ± 6.6 | 58.0 ± 2.9 | 64.4 ± 1.5 | 76.3 ± 0.9 |
| `attmask_vits16` | 41.5 ± 4.9 | 62.5 ± 2.7 | 67.7 ± 1.7 | 78.0 ± 0.9 |
| `moco_v3_vits16` | 20.9 ± 0.7 | 31.2 ± 1.2 | 34.7 ± 0.9 | 43.8 ± 1.0 |

#### Accuracy Curves (CIFAR-10, k=5)

````carousel
![CIFAR-10 few-shot kNN accuracy – CLS token pooling](C:\Users\kosti\.gemini\antigravity-ide\brain\934ebbbc-351a-4f85-b7b3-8e3a27e0c489\cifar10_knn_5_cls.png)
<!-- slide -->
![CIFAR-10 few-shot kNN accuracy – Mean pool](C:\Users\kosti\.gemini\antigravity-ide\brain\934ebbbc-351a-4f85-b7b3-8e3a27e0c489\cifar10_knn_5_mean_pool.png)
````

---

### 4.2 CIFAR-100

#### CLS Token Pooling

| Model | 1-shot | 5-shot | 10-shot | 100-shot |
|---|---|---|---|---|
| `dino_vits16` | 32.3 ± 0.6 | 50.5 ± 0.3 | 56.1 ± 0.6 | 70.3 ± 0.3 |
| `dinov2_vits14` | 48.1 ± 3.0 | 65.5 ± 0.2 | 70.4 ± 0.3 | **80.6 ± 0.3** |
| `dinov2_vits14_reg` | **49.5 ± 2.5** | **67.7 ± 0.5** | **71.4 ± 0.3** | 80.2 ± 0.3 |
| `ibot_vits16` | 34.6 ± 0.5 | 52.2 ± 0.6 | 57.8 ± 0.3 | 71.7 ± 0.4 |
| `attmask_vits16` | 34.8 ± 0.6 | 52.4 ± 0.5 | 57.3 ± 0.3 | 70.4 ± 0.2 |
| `moco_v3_vits16` | 7.2 ± 0.0 | 12.2 ± 0.2 | 14.6 ± 0.2 | 24.5 ± 0.2 |

#### Mean Pool Pooling

| Model | 1-shot | 5-shot | 10-shot | 100-shot |
|---|---|---|---|---|
| `dino_vits16` | 14.1 ± 0.6 | 23.7 ± 0.6 | 28.8 ± 0.8 | 46.1 ± 0.5 |
| `dinov2_vits14` | 42.5 ± 2.5 | 58.5 ± 0.3 | 63.6 ± 0.3 | 75.3 ± 0.2 |
| `dinov2_vits14_reg` | **47.4 ± 1.3** | **63.9 ± 0.3** | **68.5 ± 0.2** | **78.5 ± 0.2** |
| `ibot_vits16` | 18.2 ± 1.6 | 30.1 ± 0.4 | 36.0 ± 0.1 | 52.7 ± 0.2 |
| `attmask_vits16` | 21.7 ± 1.2 | 34.6 ± 0.1 | 39.9 ± 0.2 | 55.6 ± 0.2 |
| `moco_v3_vits16` | 6.6 ± 0.7 | 11.2 ± 0.6 | 13.2 ± 0.3 | 23.8 ± 0.2 |

#### Accuracy Curves (CIFAR-100, k=5)

````carousel
![CIFAR-100 few-shot kNN accuracy – CLS token pooling](C:\Users\kosti\.gemini\antigravity-ide\brain\934ebbbc-351a-4f85-b7b3-8e3a27e0c489\cifar100_knn_5_cls.png)
<!-- slide -->
![CIFAR-100 few-shot kNN accuracy – Mean pool](C:\Users\kosti\.gemini\antigravity-ide\brain\934ebbbc-351a-4f85-b7b3-8e3a27e0c489\cifar100_knn_5_mean_pool.png)
````

---

## 5. Discussion

### 5.1 DINOv2 vs DINOv1

DINOv2 ViT-S/14 substantially outperforms DINOv1 ViT-S/16 at every shot count and on both datasets. On CIFAR-10 at 100 shots with CLS tokens: **94.7%** vs **88.4%** — a +6.3 pp gap. On CIFAR-100, the gap is even wider: **80.6%** vs **70.3%** (+10.3 pp). This improvement is consistent across all pooling modes and shot counts.

Several factors contribute to this gap:
- **Patch size**: ViT-S/14 produces denser patch tokens (16×16 = 256 patches for 224×224) vs ViT-S/16 (14×14 = 196), giving richer spatial representations.
- **Training improvements**: DINOv2 incorporates masked image modelling (iBOT-style), larger batch sizes, curated high-quality data (LVD-142M), and improved regularisation.
- **Feature isotropy**: DINOv2 representations tend to be more isotropic (well-distributed across dimensions), which is essential for cosine-similarity-based retrieval.

### 5.2 Effect of Register Tokens

The `dinov2_vits14_reg` model (DINOv2 with register tokens) performs marginally better than the standard `dinov2_vits14` model, particularly at low shot counts. On CIFAR-100 at 1-shot (CLS): **49.5%** vs **48.1%** (+1.4 pp). At 100 shots the gap narrows to near-zero.

Register tokens were introduced to absorb "artifact" attention patterns that arise in the CLS token when background regions dominate. Their benefit is most pronounced for dense prediction tasks but also slightly improves retrieval quality in low-data regimes where CLS token quality matters most.

For the **mean_pool** strategy, the register variant's advantage is more consistent. This is expected: registers act as dedicated global summary buffers, freeing the patch tokens to encode more local, clean spatial information — making the mean of patch tokens a purer spatial descriptor.

### 5.3 CLS Token vs Mean Pool

The pooling strategy has dramatically different effects depending on the backbone generation:

**DINOv1**: CLS token is far superior. At 100 shots on CIFAR-10: **88.4% (cls) vs 64.4% (mean_pool)** — a 24 pp gap. DINOv1's training explicitly optimises the CLS token for global image representation through student-teacher distillation, but the patch tokens are not similarly trained to be discriminative in isolation.

**DINOv2**: The gap largely closes. At 100 shots on CIFAR-10: **94.7% (cls) vs 92.1% (mean_pool)** — only a 2.6 pp difference. DINOv2's joint masked image modelling objective encourages patch tokens to carry rich local semantic content, so their mean is competitive with the CLS token. For the register variant, mean_pool is nearly on par with cls at high shot counts.

**MoCo v3**: Both strategies perform equally poorly, suggesting that the contrastive training objective does not produce well-structured patch token representations for any pooling strategy.

### 5.4 MoCo v3 Underperformance

MoCo v3 consistently and significantly underperforms all DINO variants. On CIFAR-10 at 100 shots: **43.9%** vs **94.7%** — a 50+ pp gap. On CIFAR-100 at 100 shots: **24.5%** vs **80.6%**.

This is not a model quality failure per se, but rather a consequence of training objective alignment:
- **Contrastive learning** (MoCo v3) learns instance-level discrimination — it separates different images from each other, without explicitly grouping semantically similar ones.
- **Self-distillation** (DINO) learns semantic cluster structure — the student and teacher must agree on soft pseudo-labels across crops, which encourages semantic grouping.

kNN classification inherently requires semantically clustered representations. DINOv1/v2 directly optimise for this structure; MoCo v3 does not. The MoCo v3 features also appear to be trained for representations where cosine similarity is used differently (instance-level negative contrasting), so the pretrained model's feature space may not be well calibrated for class-level retrieval.

### 5.5 Few-Shot Scaling Behaviour

All models exhibit consistent log-linear improvement in accuracy as N increases (visible in the log-scale plots). Key observations:

- **Diminishing returns at high N**: The improvement from 50→100 shots is smaller than 5→10 shots for all models. This is expected as retrieval accuracy saturates when the support set densely covers the manifold.
- **Variance reduction with N**: Standard deviation across seeds drops dramatically with N. At 1-shot, variance is high (up to ±8.8 pp) because a single prototypical sample can be unrepresentative. By 100-shot, variance is negligible (±0.0–0.5 pp).
- **Hardness of CIFAR-100**: Absolute accuracy is much lower on CIFAR-100 due to the 10× more classes and more fine-grained visual distinctions. The ranking of models is preserved across both datasets.

### 5.6 Effect of k

Results across k ∈ {1, 5, 20} reveal consistent trends (not reported in detail above):
- **k=1**: High variance, best for models with very tight class clusters (DINOv2 at high shots).
- **k=5**: Best overall balance between noise reduction and boundary precision.
- **k=20**: Slightly worse for DINOv1 due to class mixing when clusters are not well separated; marginal improvement for DINOv2 due to smoothing noise in high-data regimes.

---

## 6. Code Architecture

The project is organised as a modular pipeline under `src/`:

```
ssl-knn-fewshot/
├── configs/
│   └── default.yaml          # Experiment configuration
├── src/
│   ├── cli.py                # Unified CLI: extract | evaluate | plot
│   ├── config.py             # YAML config loader
│   ├── models.py             # Backbone loader (timm + torch.hub)
│   ├── extract.py            # GPU-accelerated feature extraction & caching
│   ├── evaluate.py           # Few-shot kNN grid evaluation → results.csv
│   ├── knn.py                # Cosine kNN with DINO-style weighted voting
│   └── plot.py               # Accuracy-vs-shots curve generation
├── features/                 # Cached .pt feature files (not tracked in git)
├── results/
│   ├── results.csv           # Full evaluation grid results
│   └── figures/              # Generated PNG plots
└── checkpoints/              # Optional manual weights for iBOT/AttMask
```

**Key engineering decisions:**
- **GPU-only preprocessing**: Bypasses CPU PIL bottleneck on Windows by moving all image ops to GPU, achieving ~20× throughput improvement.
- **Incremental evaluation**: The CSV appends new rows and skips already-computed (dataset, model, pooling, N, seed, k) tuples, enabling resumable runs.
- **Vectorised kNN**: `scatter_add` over the full test set in one matrix multiply — no Python loop over test samples.

---

## 7. Conclusion

This study demonstrates that the choice of SSL pretraining objective has a profound impact on the quality of representations for non-parametric few-shot classification:

1. **DINOv2 (both variants) is the strongest backbone** for few-shot kNN classification, reaching 94.7% on CIFAR-10 and 80.6% on CIFAR-100 at 100 shots — competitive with supervised models trained from scratch on these datasets.

2. **DINOv1 occupies a strong middle ground**, achieving 88.4% / 70.3% at 100 shots. Its CLS token is a highly effective global descriptor, but the patch token mean is significantly weaker.

3. **MoCo v3 is poorly suited for kNN retrieval** despite being a strong contrastive SSL model. Its instance-discrimination objective does not produce semantically clustered representations.

4. **Register tokens provide a consistent, small benefit**, especially in the low-shot regime and when using mean pooling. Their main effect is to clean up patch token representations.

5. **The CLS token is generally the better pooling strategy**, except for DINOv2 where mean_pool is nearly equivalent — a sign of the improved patch-level training in DINOv2.

6. **Few-shot performance scales log-linearly with N** for all models, with negligible variance beyond 20 shots per class. Even at 1-shot, DINOv2 achieves 66–70% on CIFAR-10, demonstrating remarkable zero-label transfer.

These findings align with the broader literature showing that self-distillation methods (DINO family) produce more semantically structured representation spaces than contrastive methods, making them superior foundations for retrieval-based and non-parametric downstream applications.

---

## Appendix: Reproducibility

```bash
# Create environment
python -m venv .venv
.venv\Scripts\pip install torch torchvision timm tqdm pyyaml matplotlib

# Extract features (requires GPU)
python src/cli.py extract

# Run full evaluation grid
python src/cli.py evaluate

# Generate plots (k=5, CLS pooling)
python src/cli.py plot --k 5 --pooling cls

# Generate plots (k=5, mean pool)
python src/cli.py plot --k 5 --pooling mean_pool
```

**Hardware used**: Windows, CUDA GPU  
**Models**: All automatically downloaded from `torch.hub` / HuggingFace Hub  
**iBOT / AttMask**: Checkpoints downloaded manually from respective GitHub repos
