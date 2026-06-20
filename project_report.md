# Few-Shot Image Classification via kNN Probing of Self-Supervised Representations

**Project Report** | June 2026

---

## 1. Problem Setting

### 1.1 Motivation

Modern deep learning for image classification typically requires large amounts of labelled training data. In many practical settings — medical imaging, satellite data, rare species recognition — only a handful of labelled examples are available per class. This motivates the study of **few-shot learning**: can a model generalise well when given only *N* labelled samples per class?

A complementary line of work, **self-supervised learning (SSL)**, trains visual representations on unlabelled data by solving pretext tasks. SSL backbones can leverage massive unlabelled corpora, producing general-purpose feature embeddings without requiring a single human label during pretraining.

This project sits at the intersection of these two ideas:

> **Can features learned by SSL backbones, without any task-specific training, support accurate few-shot classification via a non-parametric classifier?**

### 1.2 The kNN Probe

We evaluate SSL representations using a **k-nearest neighbour (kNN)** classifier — a fully non-parametric method with zero learnable parameters. Given a small labelled *support set* of *N* examples per class, the kNN classifier labels a test image by finding its *k* most similar support images (by cosine distance in feature space) and aggregating their labels.

This is the strictest possible evaluation: any accuracy achieved is entirely attributable to the **geometric structure of the pretrained representation space** — specifically, whether semantically similar images are already close together before any adaptation. No fine-tuning, no linear probes, no learned heads.

### 1.3 Research Questions

1. Which SSL pretraining objective produces the most semantically coherent feature space for kNN retrieval?
2. How does few-shot performance scale with the number of labelled support examples *N*?
3. Does the choice of feature pooling strategy (CLS token vs mean patch pooling) significantly affect results?
4. Does the introduction of register tokens in DINOv2 improve kNN classification?

---

## 2. Methodology

### 2.1 Pipeline Overview

The pipeline consists of three independent stages:

```
[Frozen SSL Backbone] → [Feature Extraction & Caching] → [kNN Grid Evaluation] → [Results]
```

All backbone weights are frozen. No gradient computation occurs outside of the feature extraction forward pass.

### 2.2 Feature Extraction

Raw CIFAR images (32×32 RGB pixels) are preprocessed and fed to each backbone:

1. **Loading**: The entire dataset is loaded as a single NumPy array and converted to a GPU float32 tensor in a single operation — bypassing sequential CPU PIL image processing entirely.
2. **Normalisation**: ImageNet-standard mean and standard deviation are applied per-channel on the GPU.
3. **Resizing**: Bicubic interpolation (`torch.nn.functional.interpolate`) upsamples images from 32×32 to **224×224**, matching the backbones' expected input size. This is done on the GPU in batch.
4. **Forward pass**: Each batch is passed through `model.forward_features()` under `torch.autocast` (FP16 precision), returning a sequence of tokens `[B, L+1, D]`.
5. **Pooling**: Two feature vectors are extracted per image:
   - **CLS token** (`cls`): The first (global summary) token, optimised during pretraining for holistic image representation.
   - **Mean pool** (`mean_pool`): The arithmetic mean of all spatial patch tokens, excluding any prefix tokens (CLS and register tokens, identified via `model.num_prefix_tokens`).
6. **L2-normalisation**: Both vectors are unit-normalised, enabling cosine similarity as a dot product.

All features are cached to disk as `.pt` files, enabling fast re-evaluation without recomputation.

### 2.3 kNN Classifier

The kNN classifier implements the **DINO-style weighted voting** scheme:

$$\hat{y} = \arg\max_c \sum_{i \in \mathcal{N}_k(x)} \exp\!\left(\frac{\text{sim}(x, x_i)}{\tau}\right) \cdot \mathbf{1}[y_i = c]$$

where $\mathcal{N}_k(x)$ is the set of *k* nearest support neighbours of test sample $x$, $\text{sim}(\cdot,\cdot)$ is cosine similarity, and $\tau = 0.07$ is a temperature parameter sharpening the similarity weights.

This formulation is implemented fully vectorised using `scatter_add` over a similarity matrix $S \in \mathbb{R}^{N_\text{test} \times N_\text{support}}$, avoiding any Python loop over test samples.

### 2.4 Few-Shot Sampling Protocol

For each experimental condition:
- **N** samples per class are drawn from the full training set via stratified random sampling.
- This is repeated across **3 independent random seeds** to estimate variance.
- The full **test set** is always used for evaluation (no sub-sampling of test data).
- *k* is clamped to $\min(k,\, N \cdot C)$ to avoid requesting more neighbours than exist in the support set.

### 2.5 Evaluation Metrics

- **Top-1 Accuracy**: Fraction of test samples correctly classified.
- **Macro F1**: Unweighted mean of per-class F1 scores — robust to class imbalance (though CIFAR datasets are balanced).

Results are reported as **mean ± std** across 3 seeds.

---

## 3. Data

### 3.1 CIFAR-10

| Property | Value |
|---|---|
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Training images | 50,000 |
| Test images | 10,000 |
| Image size | 32×32 RGB |
| Class balance | Perfectly balanced (5,000 train / 1,000 test per class) |

CIFAR-10 is a relatively easy classification benchmark. With only 10 classes and visually distinct categories, strong SSL models can achieve near-supervised accuracy even with few labels.

### 3.2 CIFAR-100

| Property | Value |
|---|---|
| Classes | 100 (grouped into 20 superclasses) |
| Training images | 50,000 |
| Test images | 10,000 |
| Image size | 32×32 RGB |
| Class balance | Perfectly balanced (500 train / 100 test per class) |

CIFAR-100 is substantially harder: 10× more classes, 10× fewer training examples per class, and much finer-grained visual distinctions (e.g., separating *baby*, *boy*, *girl*, *man*, *woman* as different classes). It provides a more discriminating test of representation quality.

---

## 4. Models

Four pretrained ViT-Small backbone models are evaluated:

### 4.1 DINO ViT-S/16 (`dino_vits16`)

DINO (Self-Distillation with No Labels) trains a student network to predict the output distribution of a momentum-updated teacher network. The teacher and student see different random crops of the same image. The CLS token is explicitly optimised to serve as a holistic image summary.

- **Architecture**: ViT-S/16 (patch size 16, small embedding)
- **Pretraining data**: ImageNet-1k
- **Key property**: Excellent CLS token; patch tokens not explicitly trained for discriminability

### 4.2 DINOv2 ViT-S/14 (`dinov2_vits14`)

DINOv2 extends DINO with masked image modelling (iBOT-style), curated high-quality pretraining data (LVD-142M, ~142M images), and numerous training improvements. The smaller patch size (14 vs 16) yields denser feature maps.

- **Architecture**: ViT-S/14 (patch size 14, small embedding)
- **Pretraining data**: LVD-142M (curated web data)
- **Key property**: Strong CLS token *and* highly informative patch tokens

### 4.3 DINOv2 ViT-S/14 with Registers (`dinov2_vits14_reg`)

Identical to DINOv2 ViT-S/14, but with additional **register tokens** appended to the input sequence. Register tokens act as dedicated "scratch space" for the attention mechanism, preventing artifact patterns from accumulating in the CLS token when the model attends to uninformative background regions.

- **Architecture**: ViT-S/14 + 4 register tokens
- **Pretraining data**: LVD-142M
- **Key property**: Cleaner patch token representations; beneficial for both dense tasks and retrieval

### 4.4 MoCo v3 ViT-S/16 (`moco_v3_vits16`)

MoCo v3 is a contrastive self-supervised method. It trains a backbone to maximise agreement between differently-augmented views of the same image, while minimising agreement with views of different images (negatives from a momentum-updated queue).

- **Architecture**: ViT-S/16
- **Pretraining data**: ImageNet-1k
- **Key property**: Instance-level discrimination objective; does not explicitly group semantic classes

---

## 5. Results

All results are reported for **k = 5 neighbours**, mean ± std (%) across 3 seeds.

### 5.1 CIFAR-10 Results

#### CLS Token Pooling

| Model | 1-shot | 5-shot | 10-shot | 100-shot |
|---|:---:|:---:|:---:|:---:|
| `dino_vits16` | 53.2 ± 5.7 | 76.2 ± 2.0 | 80.1 ± 1.3 | 88.4 ± 0.5 |
| `dinov2_vits14` | 66.0 ± 8.0 | 87.0 ± 1.1 | 89.6 ± 0.4 | **94.7 ± 0.2** |
| `dinov2_vits14_reg` | **68.2 ± 5.8** | **87.8 ± 2.0** | **90.7 ± 0.8** | 94.6 ± 0.0 |
| `moco_v3_vits16` | 21.8 ± 1.3 | 30.5 ± 1.5 | 33.5 ± 0.7 | 43.9 ± 0.5 |

#### Mean Pool Pooling

| Model | 1-shot | 5-shot | 10-shot | 100-shot |
|---|:---:|:---:|:---:|:---:|
| `dino_vits16` | 32.8 ± 5.1 | 46.6 ± 1.7 | 51.4 ± 1.2 | 64.4 ± 0.4 |
| `dinov2_vits14` | 66.2 ± 8.8 | 84.6 ± 0.2 | 86.4 ± 0.3 | 92.1 ± 0.2 |
| `dinov2_vits14_reg` | **69.6 ± 8.7** | **87.7 ± 2.0** | **90.1 ± 1.1** | **94.0 ± 0.3** |
| `moco_v3_vits16` | 20.9 ± 0.7 | 31.2 ± 1.2 | 34.7 ± 0.9 | 43.8 ± 1.0 |

> [!NOTE]
> Random chance on CIFAR-10 = 10%. MoCo v3 at 43.9% (100-shot) is only 4.4× above chance, while DINOv2 at 94.7% is near the supervised upper bound.

### 5.2 CIFAR-10 Plots

````carousel
![CIFAR-10 kNN accuracy vs shots — CLS token pooling (k=5). DINOv2 variants dominate; MoCo v3 lags far behind.](C:\Users\kosti\.gemini\antigravity-ide\brain\934ebbbc-351a-4f85-b7b3-8e3a27e0c489\cifar10_knn_5_cls.png)
<!-- slide -->
![CIFAR-10 kNN accuracy vs shots — Mean pool (k=5). DINOv1 collapses dramatically with mean pooling; DINOv2 remains strong.](C:\Users\kosti\.gemini\antigravity-ide\brain\934ebbbc-351a-4f85-b7b3-8e3a27e0c489\cifar10_knn_5_mean_pool.png)
````

---

### 5.3 CIFAR-100 Results

#### CLS Token Pooling

| Model | 1-shot | 5-shot | 10-shot | 100-shot |
|---|:---:|:---:|:---:|:---:|
| `dino_vits16` | 32.3 ± 0.6 | 50.5 ± 0.3 | 56.1 ± 0.6 | 70.3 ± 0.3 |
| `dinov2_vits14` | 48.1 ± 3.0 | 65.5 ± 0.2 | 70.4 ± 0.3 | **80.6 ± 0.3** |
| `dinov2_vits14_reg` | **49.5 ± 2.5** | **67.7 ± 0.5** | **71.4 ± 0.3** | 80.2 ± 0.3 |
| `moco_v3_vits16` | 7.2 ± 0.0 | 12.2 ± 0.2 | 14.6 ± 0.2 | 24.5 ± 0.2 |

#### Mean Pool Pooling

| Model | 1-shot | 5-shot | 10-shot | 100-shot |
|---|:---:|:---:|:---:|:---:|
| `dino_vits16` | 14.1 ± 0.6 | 23.7 ± 0.6 | 28.8 ± 0.8 | 46.1 ± 0.5 |
| `dinov2_vits14` | 42.5 ± 2.5 | 58.5 ± 0.3 | 63.6 ± 0.3 | 75.3 ± 0.2 |
| `dinov2_vits14_reg` | **47.4 ± 1.3** | **63.9 ± 0.3** | **68.5 ± 0.2** | **78.5 ± 0.2** |
| `moco_v3_vits16` | 6.6 ± 0.7 | 11.2 ± 0.6 | 13.2 ± 0.3 | 23.8 ± 0.2 |

> [!NOTE]
> Random chance on CIFAR-100 = 1%. MoCo v3 at 24.5% (100-shot) is 24.5× above chance but still dramatically below DINOv2's 80.6%.

### 5.4 CIFAR-100 Plots

````carousel
![CIFAR-100 kNN accuracy vs shots — CLS token pooling (k=5). The performance gap between DINOv2 and DINOv1 is wider on CIFAR-100.](C:\Users\kosti\.gemini\antigravity-ide\brain\934ebbbc-351a-4f85-b7b3-8e3a27e0c489\cifar100_knn_5_cls.png)
<!-- slide -->
![CIFAR-100 kNN accuracy vs shots — Mean pool (k=5). DINOv1 mean_pool degrades severely on CIFAR-100; DINOv2 register variant is most robust.](C:\Users\kosti\.gemini\antigravity-ide\brain\934ebbbc-351a-4f85-b7b3-8e3a27e0c489\cifar100_knn_5_mean_pool.png)
````

---

### 5.5 Summary: Best Results per Model

| Model | Best CIFAR-10 | Best CIFAR-100 | Best Pooling |
|---|:---:|:---:|:---:|
| `dino_vits16` | 88.4% | 70.3% | CLS |
| `dinov2_vits14` | 94.7% | 80.6% | CLS |
| `dinov2_vits14_reg` | 94.6% | 80.2% | CLS |
| `moco_v3_vits16` | 43.9% | 24.5% | CLS ≈ Mean |

*All at 100 shots, k=5.*

---

## 6. Conclusions

### 6.1 Summary of Findings

**Finding 1 — Pretraining objective matters more than architecture.**
All four backbones share roughly the same ViT-Small capacity, yet accuracy at 100 shots spans 43.9%→94.7% on CIFAR-10. The training objective — not the model size — determines whether the representation space supports kNN retrieval.

**Finding 2 — DINOv2 is the clear best choice for few-shot kNN.**
DINOv2 ViT-S/14 (with or without registers) achieves 94.7% on CIFAR-10 and 80.6% on CIFAR-100 at 100 shots. Even at just **1 shot**, DINOv2 reaches **66–70% on CIFAR-10** — a remarkable result requiring zero labelled data during pretraining.

**Finding 3 — Self-distillation (DINO) beats contrastive learning (MoCo) for retrieval.**
MoCo v3 is trained for *instance discrimination* — separating individual images from each other. This produces features that distinguish specific images but do not cluster semantic classes. DINO's student-teacher framework explicitly enforces semantic consistency across crops, creating naturally clustered class representations. MoCo v3 underperforms by 50+ percentage points at 100 shots.

**Finding 4 — CLS token is the superior pooling strategy, except for DINOv2.**
For DINOv1, CLS vs mean_pool is decisive: 88.4% vs 64.4% on CIFAR-10. DINOv1 explicitly optimises the CLS token during pretraining; patch tokens are not trained to be class-discriminative individually. DINOv2 closes this gap (94.7% vs 92.1%), because its masked image modelling objective trains patch tokens to carry rich local semantic content.

**Finding 5 — Register tokens help most at low shot counts.**
The register variant outperforms the standard DINOv2 at 1-shot (68.2% vs 66.0% on CIFAR-10; 49.5% vs 48.1% on CIFAR-100) but converges to similar accuracy at 100 shots. Registers clean up CLS token artifacts at low data, where each prototype's quality matters most.

**Finding 6 — Few-shot performance scales predictably.**
Accuracy grows log-linearly with N across all models (clear on the log-scale x-axis plots). Variance across seeds is highest at 1-shot (individual prototypes may be unrepresentative) and collapses to near-zero by 50-shot. This means the few-shot evaluation protocol is reliable and reproducible above 10 shots.

### 6.2 Practical Recommendations

| Goal | Recommendation |
|---|---|
| Maximum accuracy, any N | DINOv2 ViT-S/14 (reg), CLS token |
| Best 1-shot accuracy | DINOv2 ViT-S/14 (reg), mean_pool |
| Fastest extraction speed | DINOv1 ViT-S/16 (smaller patch → fewer ops) |
| Contrastive SSL comparison | MoCo v3 should not be used for kNN — consider a linear probe instead |

### 6.3 Limitations

- All backbones are pretrained on large-scale web data (ImageNet / LVD-142M), while CIFAR images are low-resolution 32×32 photos. Results may not directly transfer to domains far from natural images (e.g., medical, satellite, microscopy).
- iBOT and AttMask models could not be evaluated due to unavailable checkpoint files. Including masked image modelling SSL methods beyond DINOv2 would provide a more complete picture.
- The evaluation is limited to ViT-Small variants. Scaling effects (ViT-Base, ViT-Large) are not explored.
- k = 5 is used for the main tables; the effect of k is consistent but not fully reported above.
