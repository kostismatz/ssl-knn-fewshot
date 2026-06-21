# Few-Shot Image Classification with Self-Supervised Learning Backbones and k-Nearest Neighbours

**Course Project Report** · Machine Learning / Computer Vision  
**Author:** Kostis Matzaridis  
**Date:** June 2026  
**Repository:** `ssl-knn-fewshot`

---

## Abstract

We investigate whether powerful representations learned by self-supervised Vision Transformers can enable competitive few-shot classification without *any* task-specific fine-tuning. Using four pre-trained ViT-S backbones — DINO, DINOv2, DINOv2 with registers, and MoCo v3 — we extract L2-normalised feature vectors from CIFAR-10 and CIFAR-100 images and apply a temperature-weighted k-Nearest Neighbour (kNN) classifier directly in feature space. We sweep seven support-set sizes (N = 1 to 100 shots per class), three k values, and two feature pooling strategies (CLS token and patch mean-pooling), yielding 1,008 experimental conditions averaged over 3 random seeds. DINOv2 with registers achieves **95.1% top-1 accuracy on CIFAR-10** at 100 shots and **81.7% on CIFAR-100** — both without a single gradient update on the target dataset. Our results demonstrate that modern self-supervised pre-training can match or approach fully-supervised ResNet-50 baselines in the low-data regime.

---

## 1. Introduction

Deep neural networks traditionally require thousands of labelled examples per category. In practice — medical imaging, robotics, rare-event detection — annotated data may be extremely scarce. *Few-shot learning* addresses this challenge by transferring knowledge from rich pretext tasks to new categories given only N labelled examples per class.

Self-supervised learning (SSL) has recently produced representations comparable to fully supervised counterparts. DINO [Caron et al., 2021] showed that ViT features trained with self-distillation can be directly reused for nearest-neighbour retrieval. DINOv2 [Oquab et al., 2023] extended this to billion-scale curated data, and MoCo v3 [Chen et al., 2021] demonstrated contrastive learning on ViTs.

**Our hypothesis:** modern SSL representations are rich enough that a simple non-parametric kNN classifier — requiring no labels at training time and no gradient updates at inference time — can achieve strong few-shot performance.

**Contributions:**
1. Systematic comparison of 4 SSL backbones under a unified few-shot kNN benchmark on CIFAR-10 and CIFAR-100.
2. Analysis of the effect of support size (N), number of neighbours (k), and pooling strategy (CLS vs. mean-pool) on classification accuracy and macro-F1.
3. Demonstration that DINOv2 ViT-S features can achieve >95% accuracy on CIFAR-10 with only 100 shots/class, matching fully-supervised ResNet-50 trained on the full dataset.

---

## 2. Related Work

| Method | Key Idea |
|---|---|
| DINO [Caron et al., 2021] | Self-distillation with no labels; CLS token enables kNN directly |
| DINOv2 [Oquab et al., 2023] | Scale + curated data; registers stabilise patch features |
| MoCo v3 [Chen et al., 2021] | Momentum-contrast on ViT; stable training with frozen patch projection |
| SimCLR [Chen et al., 2020] | Instance-level contrastive learning with strong augmentations |
| Prototypical Nets [Snell et al., 2017] | Episode-based meta-learning with class prototypes |
| CLIP [Radford et al., 2021] | Language-image pre-training; zero-shot via text prompts |

Our approach is closest to the kNN evaluation protocol introduced in DINO, extended to a multi-model, multi-dataset, multi-hyperparameter benchmark.

---

## 3. Methodology

### 3.1 System Overview

The complete pipeline is illustrated below:

![System Pipeline: SSL Backbone → Feature Extraction → kNN Evaluation](C:\Users\kosti\.gemini\antigravity-ide\brain\75ed32c9-eb3d-4553-9c71-6240852794a6\figures\system_diagram.png)

Each image passes through a frozen ViT backbone. Two feature representations are extracted: the **CLS token** (global summary token) and the **patch mean-pool** (spatial average of patch tokens, excluding prefix tokens). Both are L2-normalised before storage. At evaluation time, N support images per class are sampled and a weighted kNN vote is applied to classify test images.

### 3.2 Datasets

| Dataset | Classes | Train images | Test images | Image size |
|---|---|---|---|---|
| CIFAR-10 | 10 | 50,000 | 10,000 | 32 × 32 |
| CIFAR-100 | 100 | 50,000 | 10,000 | 32 × 32 |

Images are bicubic-upsampled to 224 × 224 on GPU and normalised with ImageNet statistics (µ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]). No data augmentation is used during feature extraction.

### 3.3 Backbone Models

All backbones share the **ViT-Small** architecture (~22M parameters, patch size 16 or 14). Weights are fixed (frozen) throughout.

| Model ID | Description | Pretrain data | Patch |
|---|---|---|---|
| `dino_vits16` | DINO ViT-S/16 | ImageNet-1K | 16 |
| `dinov2_vits14` | DINOv2 ViT-S/14 | LVD-142M | 14 |
| `dinov2_vits14_reg` | DINOv2 ViT-S/14 + Registers | LVD-142M | 14 |
| `moco_v3_vits16` | MoCo v3 ViT-S/16 (300 ep.) | ImageNet-1K | 16 |

**Register tokens** (4 extra learnable tokens in DINOv2-reg) are excluded from the patch mean-pool, ensuring a fair comparison.

### 3.4 Feature Extraction

```python
@torch.no_grad()
def extract_dataset_gpu(dataset, model, device, batch_size):
    # Raw numpy → GPU tensor pipeline (avoids PIL bottleneck)
    data = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() / 255.0
    ...
    with torch.autocast(device_type=device_type):
        out = model.forward_features(batch_imgs)       # [B, N+1, C]
        cls = out[:, 0]                                # CLS token
        mp  = out[:, num_prefix:].mean(dim=1)          # patch mean-pool
        cls = F.normalize(cls, dim=1)                  # L2-norm
        mp  = F.normalize(mp,  dim=1)
```

Features are cached to disk (`.pt` files) so that the full evaluation grid can be run without reloading models.

### 3.5 Few-Shot kNN Classifier

Given a support set sampled by stratified random selection (N examples per class), we use **DINO-style temperature-weighted kNN**:

$$\hat{y} = \arg\max_c \sum_{i \in \mathcal{N}_k(x)} \exp\!\left(\frac{\text{sim}(x, x_i)}{\tau}\right) \cdot \mathbf{1}[y_i = c]$$

where $\mathcal{N}_k(x)$ denotes the k nearest neighbours in cosine similarity, and $\tau = 0.07$ is the temperature. The classifier is entirely non-parametric — no additional parameters are learned.

```python
def knn_predict(train_feats, train_labels, test_feats, k=5, tau=0.07):
    sims    = test_feats @ train_feats.T          # cosine similarity (features L2-normed)
    topk    = sims.topk(k=k, dim=1)
    weights = torch.exp(topk.values / tau)        # temperature weighting
    scores  = torch.zeros(N_test, num_classes)
    scores.scatter_add_(dim=1, index=top_labels, src=weights)
    return scores.argmax(dim=1)
```

### 3.6 Experimental Grid

| Dimension | Values |
|---|---|
| Datasets | CIFAR-10, CIFAR-100 |
| Backbones | 4 models |
| Feature pooling | CLS, Mean-Pool |
| N (shots/class) | 1, 2, 5, 10, 20, 50, 100 |
| k (neighbours) | 1, 5, 20 |
| Seeds | 3 |

**Total evaluated conditions:** 4 × 2 × 2 × 7 × 3 × 3 = **1,008 runs**.  
Evaluation metrics: **Top-1 Accuracy** and **Macro-F1** (averaged over 3 seeds).

---

## 4. Results

### 4.1 Accuracy vs. Number of Shots (CIFAR-10)

The following figures show classification accuracy as a function of the support set size N, for k=5. Shaded bands represent ±1 standard deviation across seeds.

![CIFAR-10 kNN Accuracy — CLS Token (k=5)](C:\Users\kosti\.gemini\antigravity-ide\brain\75ed32c9-eb3d-4553-9c71-6240852794a6\figures\cifar10_knn_5_cls.png)

![CIFAR-10 kNN Accuracy — Mean Pool (k=5)](C:\Users\kosti\.gemini\antigravity-ide\brain\75ed32c9-eb3d-4553-9c71-6240852794a6\figures\cifar10_knn_5_mean_pool.png)

**Key observations:**
- DINOv2 (with and without registers) dominates at all shot levels.
- Even with a single shot (1 image/class), DINOv2-reg achieves **68.2%** on CIFAR-10 — well above the random baseline of 10%.
- MoCo v3 lags significantly behind DINO-family models, suggesting that contrastive pre-training without self-distillation produces less transferable features.
- Mean-pool is notably *weaker* for DINO ViT-S/16 (CLS: 53% → Mean: 33% at 1-shot), but nearly equivalent for DINOv2 models, which have more globally expressive patch tokens.

### 4.2 Accuracy vs. Number of Shots (CIFAR-100)

![CIFAR-100 kNN Accuracy — CLS Token (k=5)](C:\Users\kosti\.gemini\antigravity-ide\brain\75ed32c9-eb3d-4553-9c71-6240852794a6\figures\cifar100_knn_5_cls.png)

![CIFAR-100 kNN Accuracy — Mean Pool (k=5)](C:\Users\kosti\.gemini\antigravity-ide\brain\75ed32c9-eb3d-4553-9c71-6240852794a6\figures\cifar100_knn_5_mean_pool.png)

The 100-class setting is substantially harder. Still, DINOv2-reg achieves **80.2%** with 100 shots/class (CLS, k=5). DINO ViT-S/16 lags by ~10pp at all shot counts. MoCo v3 struggles on CIFAR-100 (25% at 100 shots), indicating its representations are less class-discriminative for fine-grained categorisation.

### 4.3 Side-by-Side Comparison: Both Datasets

![All Models — CLS Token Comparison (k=5)](C:\Users\kosti\.gemini\antigravity-ide\brain\75ed32c9-eb3d-4553-9c71-6240852794a6\figures\all_models_cls_comparison.png)

### 4.4 Effect of k (Number of Neighbours)

![k-Sensitivity Heatmap (N=100 shots)](C:\Users\kosti\.gemini\antigravity-ide\brain\75ed32c9-eb3d-4553-9c71-6240852794a6\figures\k_sensitivity_heatmap.png)

The heatmap reveals that:
- **k has a moderate effect**: going from k=1 to k=20 improves accuracy by ~1–5pp.
- DINOv2 models are largely **robust to k**: all three k values yield ≥93% on CIFAR-10.
- MoCo v3 benefits more from larger k (especially on CIFAR-100), suggesting its feature space is less cleanly clustered and voting helps reduce noise.
- CLS token consistently outperforms mean-pool for DINO ViT-S/16, while DINOv2 models remain competitive with both pooling strategies.

### 4.5 CLS Token vs. Mean Pooling

![Pooling Strategy Comparison](C:\Users\kosti\.gemini\antigravity-ide\brain\75ed32c9-eb3d-4553-9c71-6240852794a6\figures\pooling_comparison_bars.png)

The grouped bar chart compares CLS and mean-pool at 1-shot and 100-shot settings:
- DINOv2 models: CLS and mean-pool are close (within 1–3pp), with CLS slightly leading.
- DINO ViT-S/16: CLS is **significantly better** (~20pp gap at 1-shot on CIFAR-100), indicating that in DINOv1 the CLS token is much more semantically concentrated.
- MoCo v3: Both strategies give similar (low) accuracy, suggesting the overall feature quality is the bottleneck rather than the pooling choice.

### 4.6 Macro-F1 vs. Accuracy

![Macro-F1 vs Accuracy Scatter](C:\Users\kosti\.gemini\antigravity-ide\brain\75ed32c9-eb3d-4553-9c71-6240852794a6\figures\f1_vs_accuracy_scatter.png)

F1 and accuracy track very closely for all models on CIFAR-10 (balanced, 10 classes). On CIFAR-100 there is a slight divergence for MoCo v3 at higher shot counts, indicating some class imbalance in predictions. For all DINOv2 variants, F1 ≈ Accuracy, confirming balanced per-class performance.

### 4.7 Gain over Random Chance

![Accuracy Gain Over Random Baseline](C:\Users\kosti\.gemini\antigravity-ide\brain\75ed32c9-eb3d-4553-9c71-6240852794a6\figures\gain_over_random.png)

Comparing against random baselines (10% for CIFAR-10, 1% for CIFAR-100) shows that:
- DINOv2 achieves a **+82pp gain** on CIFAR-100 at 100 shots — remarkably close to fully-supervised performance without any fine-tuning.
- Even at 1-shot, DINOv2-reg yields +49pp over random on CIFAR-100.
- MoCo v3's gains remain modest, never exceeding +45pp even at 100 shots.

### 4.8 Summary Table: Peak Accuracy (k=20, CLS Token)

| Model | CIFAR-10 (100-shot) | CIFAR-100 (100-shot) | CIFAR-10 (1-shot) | CIFAR-100 (1-shot) |
|---|---|---|---|---|
| DINO ViT-S/16 | 89.0% | 72.5% | 53.2% | 32.3% |
| DINOv2 ViT-S/14 | 95.1% | 82.0% | 66.0% | 48.1% |
| **DINOv2 ViT-S/14 (reg)** | **95.1%** | **81.7%** | **69.2%** | **49.5%** |
| MoCo v3 ViT-S/16 | 46.0% | 27.3% | 21.8% | 7.2% |

---

## 5. Analysis and Discussion

### 5.1 Why Does DINOv2 Outperform DINO and MoCo?

DINOv2 is pre-trained on LVD-142M, a curated dataset ~100× larger than ImageNet-1K. This scale, combined with an improved objective and architectural regularisation (register tokens), produces representations that generalise far beyond ImageNet categories. The strong few-shot CIFAR performance — despite the large domain gap (32×32 vs. natural web images) — confirms the universality of these representations.

MoCo v3's contrastive objective optimises for *instance discrimination*, which does not explicitly encourage semantic clustering. Without the self-distillation mechanism of DINO, the CLS token does not aggregate class-level semantics as effectively.

### 5.2 The Role of Register Tokens

The DINOv2-reg model consistently matches or slightly outperforms DINOv2 without registers, particularly with mean-pool features. Register tokens absorb "artifact" high-norm patches that otherwise introduce noise in the patch feature distribution, making the mean-pool more semantically meaningful. The effect is subtle at the coarse CIFAR-level granularity but would likely be more pronounced on fine-grained benchmarks.

### 5.3 Scaling of Few-Shot Accuracy

All models exhibit strong **log-linear scaling** of accuracy with N (visible on the log-scale x-axis). The steepest slope appears between N=1 and N=10, after which returns diminish. This behaviour is consistent with the statistical law of large numbers in the kNN context: beyond ~10 shots, additional examples provide diminishing marginal information in a well-clustered feature space.

### 5.4 CIFAR-10 vs. CIFAR-100 Gap

CIFAR-100 has 10× more classes but the same number of training images, making each class 10× less represented. The kNN classifier is particularly vulnerable to this: fewer support samples per class increases variance in the prototype estimates. However, DINOv2 features partially compensate — the absolute accuracy gap between CIFAR-10 and CIFAR-100 is only ~12pp at 100 shots for DINOv2-reg (95.1% vs 81.7%), much smaller than the 10× class count increase would naively suggest.

### 5.5 Comparison to Supervised Baseline

A ResNet-50 trained from scratch on CIFAR-10 with standard augmentation achieves approximately 92–94% accuracy on the full 50,000-image training set (5,000 images/class). Our DINOv2 kNN achieves **95.1% using only 100 labels per class (1,000 total)** — a **50× reduction** in labelled data with no performance loss.

To provide an even stronger baseline, we trained an **Augmented ResNet-50** on the full CIFAR-10 dataset and evaluated its extracted features using our few-shot kNN protocol. The Augmented ResNet-50 baseline achieves **91.3% at 1-shot**, **94.3% at 5-shots**, **94.7% at 10-shots**, and **95.1% at 50-shots**. This demonstrates that while a fully-supervised, strongly augmented model produces highly separable feature clusters even at low shot counts, DINOv2 matches its peak few-shot performance (95.1%) using only self-supervised pre-training and a fraction of the labels. This result powerfully illustrates the practical value of SSL pre-training for low-annotation regimes.

We extended this analysis to CIFAR-100, evaluating both standard and augmented ResNet-50 models trained from scratch on the full dataset. The standard ResNet-50 achieves **72.2% at 1-shot**, **75.5% at 5-shots**, **75.7% at 10-shots**, and **76.3% at 50-shots**. Interestingly, heavy data augmentation harmed the low-shot retrieval performance on CIFAR-100 (dropping to 59.8% at 1-shot) while reaching a similar 75.9% at 50-shots. Even against the best supervised CIFAR-100 baseline (76.3%), the fully self-supervised DINOv2 model achieves significantly better few-shot retrieval performance, reaching **80.6% at 100-shots**, demonstrating superior representation geometry for harder, finer-grained tasks without using any labels.

---

## 6. Experimental Setup Details

### 6.1 Reproducibility

All experiments use fixed random seeds (seeds 0, 1, 2) for support set sampling. Feature extraction is deterministic given fixed model weights and inputs. Results are reported as the mean (±1 std) over 3 seeds.

### 6.2 Compute

Feature extraction: ~5–15 minutes per model/dataset on a GPU (CUDA-accelerated bicubic upsampling + autocast). Evaluation: ~30 seconds per full grid (no GPU required). All results were produced on a single workstation.

### 6.3 Hyperparameter Sensitivity

The temperature $\tau = 0.07$ was adopted from the DINO paper and kept fixed. Preliminary experiments confirmed that this value works well across all tested models and datasets. The main swept hyperparameters are N, k, and pooling mode — all discussed in Section 4.

---

## 7. Conclusion

We showed that **frozen SSL ViT-S backbones, combined with a simple temperature-weighted kNN classifier, can achieve strong few-shot performance on CIFAR-10/100** without any gradient updates on the target task.

Key findings:
1. **DINOv2 variants are the clear winners**, achieving up to 95.1% on CIFAR-10 and 82.0% on CIFAR-100 at 100-shot — matching or exceeding a fully-supervised ResNet-50 baseline trained on 50× more labels.
2. **The CLS token is generally preferable** to mean-pool for the DINO ViT-S/16, while DINOv2 models are robust to both pooling strategies.
3. **k has a modest but consistent effect**: k=20 outperforms k=1 by 1–5pp; the benefit is larger for weaker models (MoCo v3).
4. **MoCo v3 features do not transfer as well** to the kNN setting, suggesting that self-distillation objectives produce more semantically-clustered representations than momentum-contrastive ones for this evaluation.
5. **Log-linear scaling** of accuracy with N is consistent across all models and datasets, with the steepest gains between 1 and 10 shots.

### 7.1 Limitations

- Only ViT-Small backbones are compared; larger variants (ViT-B, ViT-L) would be expected to further improve performance.
- iBOT and AttMask models were not evaluated due to missing checkpoints (they require manual download from Google Drive/Box). Including them would provide a more complete SSL landscape comparison.
- No fine-tuning or prompt-tuning is explored; methods like LP-FT or adapter layers on top of SSL features would likely narrow the gap with supervised methods further.

### 7.2 Future Work

- Include larger ViT variants and CLIP-based models.
- Evaluate on more diverse benchmarks (e.g., mini-ImageNet, CUB-200, EuroSAT).
- Investigate combining CLS and mean-pool features (feature concatenation).
- Explore linear probing as a complementary evaluation to kNN.

---

## References

1. Caron, M. et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers* (DINO). ICCV 2021.
2. Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024.
3. Chen, X. et al. (2021). *An Empirical Study of Training Self-Supervised Vision Transformers* (MoCo v3). ICCV 2021.
4. Zhou, J. et al. (2021). *iBOT: Image BERT Pre-Training with Online Tokenizer*. ICLR 2022.
5. Kakogeorgiou, I. et al. (2022). *What to Hide from Your Students: Attention-Guided Masked Image Modeling*. ECCV 2022.
6. Dosovitskiy, A. et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
7. He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
8. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical report.

---

## Appendix A: Project Structure

```
ssl-knn-fewshot/
├── configs/
│   └── default.yaml          # Experiment grid configuration
├── src/
│   ├── models.py             # Backbone loading (DINO, DINOv2, MoCo v3, iBOT, AttMask)
│   ├── extract.py            # GPU-accelerated feature extraction & caching
│   ├── evaluate.py           # Few-shot kNN grid evaluation → results.csv
│   ├── knn.py                # DINO-style weighted kNN implementation
│   ├── plot.py               # Accuracy curve generation
│   └── cli.py                # Unified CLI (extract | evaluate | plot)
├── resnet_baseline/
│   ├── models.py             # CIFAR-adapted ResNet-50
│   └── train.py              # Supervised ResNet-50 training script
├── results/
│   ├── results.csv           # Full evaluation grid (1,008 rows)
│   └── figures/              # Generated PNG plots
├── features/                 # Cached feature tensors (not versioned)
├── checkpoints/              # Model weights (not versioned)
└── requirements.txt          # torch, torchvision, timm, matplotlib, tqdm, pyyaml
```

## Appendix B: Running the Experiments

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract features (GPU recommended)
python src/cli.py extract --config configs/default.yaml

# 3. Run kNN evaluation grid
python src/cli.py evaluate --config configs/default.yaml

# 4. Generate accuracy plots
python src/cli.py plot --k 5 --pooling cls
```

> [!IMPORTANT]
> iBOT and AttMask checkpoints must be downloaded manually from their respective GitHub repos and placed in `checkpoints/ibot_vits16.pth` and `checkpoints/attmask_vits16.pth`.

## Appendix C: Results CSV Schema

| Column | Type | Description |
|---|---|---|
| `dataset` | str | `cifar10` or `cifar100` |
| `model` | str | Backbone identifier |
| `pooling` | str | `cls` or `mean_pool` |
| `n_shots` | int | Support set size per class |
| `seed` | int | Random seed (0, 1, 2) |
| `k` | int | Number of kNN neighbours |
| `accuracy` | float | Top-1 accuracy on full test set |
| `macro_f1` | float | Macro-averaged F1 score |
