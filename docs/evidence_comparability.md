# Evidence Comparability Framework & Decision Theory

This document defines the mathematical comparability criteria, normalization schemes, and multi-attribute decision rules used to evaluate model candidates across heterogeneous vision benchmarks.

---

## 1. Direct vs. Indirect Comparability Matrix

```
                        ResNet-50  ViT-Base  ConvNeXt  Scratch-CNN  Scratch-ViT  MViTv2
ResNet-50 (CIFAR-10)       [ Direct ] [ Direct ] [ Direct ] [ Indirect*] [ Indirect*] [ Protocol**]
ViT-Base/16 (CIFAR-10)     [ Direct ] [ Direct ] [ Direct ] [ Indirect*] [ Indirect*] [ Protocol**]
ConvNeXt-Tiny (CIFAR-10)   [ Direct ] [ Direct ] [ Direct ] [ Indirect*] [ Indirect*] [ Protocol**]
Scratch CNN (CIFAR-100)    [ Indirect*] [ Indirect*] [ Indirect*] [ Direct ] [ Direct ] [ Incomparable ]
Scratch ViT (CIFAR-100)    [ Indirect*] [ Indirect*] [ Indirect*] [ Direct ] [ Direct ] [ Incomparable ]
MViTv2 (ImageNet-1K)       [ Protocol**] [ Protocol**] [ Protocol**] [ Incomparable ] [ Incomparable ] [ Direct ]
```

- **[ Direct ]**: Identical dataset, resolution, and evaluation script. Raw accuracy and corruption curves are directly comparable.
- **[ Indirect* ]**: Different class counts (10 vs 100), but relative corruption degradation ($\% \text{ drop}$) and calibration properties can be compared architecturally.
- **[ Protocol** ]**: Different benchmark scale (ImageNet-1K vs CIFAR). FLOPs, parameter efficiency, and top-level capacity are contrasted.
- **[ Incomparable ]**: Random-init scratch models on 32×32 cannot be meaningfully compared to 50M parameter hierarchical models trained on 1.28M full-resolution images.

---

## 2. Mathematical Normalization Framework

To enable principled ranking across candidates without falling into naive cross-dataset comparisons, the system computes normalized dimensionless indicators:

### 2.1 Relative Robustness Retention ($R_{ret}$)
$$R_{ret} = \frac{\text{Acc}_{\text{corrupted}}}{\text{Acc}_{\text{clean}}} \times 100\%$$
- **Interpretation**: Measures the percentage of clean performance preserved under severe corruption.
- Independent of whether baseline accuracy is 60% (CIFAR-100 scratch) or 97% (CIFAR-10 fine-tuned).

### 2.2 Expected Calibration Error (ECE)
$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- Lower is better. Measures reliability of prediction probabilities for safety-critical deployment.

### 2.3 Hardware Efficiency Index (HEI)
$$HEI = \frac{\text{Throughput (img/sec)}}{\text{Parameters (M)} \times \text{FLOPs (G)}}$$
- Higher is better. Rewards compact, high-speed architectures suitable for edge devices.

---

## 3. Multi-Attribute Utility Theory (MAUT) Decision Function

The recommendation engine implements a weighted additive utility score:

$$S(m) = \sum_{k \in \mathcal{K}} w_k \cdot \phi_k(m)$$

Where:
- $\mathcal{K} = \{\text{Accuracy}, \text{Robustness}, \text{Latency}, \text{DataEfficiency}, \text{Calibration}\}$
- $w_k \ge 0$ is the user preference weight ($\sum w_k = 1.0$)
- $\phi_k(m) \in [0, 1]$ is the min-max normalized utility score for model $m$ on attribute $k$ within the filtered candidate pool:

$$\phi_k(m) = \frac{x_k(m) - \min_{j} x_k(j)}{\max_j x_k(j) - \min_j x_k(j)}$$

For inverse metrics (latency, ECE, corruption drop):
$$\phi_k(m) = \frac{\max_j x_k(j) - x_k(m)}{\max_j x_k(j) - \min_j x_k(j)}$$

---

## 4. Confidence Gating Rules

Each recommendation output produces an empirical **Evidence Confidence Score** ($C \in [0\%, 100\%]$):
1. **Direct Empirical Benchmark Match**: Base confidence = $95\%$.
2. **Estimated / Cross-Referenced Evidence (e.g. MViT corruptions)**: Subtract $15\%$.
3. **Cross-Resolution Transfer Inference**: Subtract $10\%$.
4. **Hard Constraint Margin < 5%**: Subtract $5\%$ (indicating sensitivity to hardware threshold).
