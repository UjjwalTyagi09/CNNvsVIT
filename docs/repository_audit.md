# Comprehensive Repository & Asset Audit

This document details the inspection and architectural audit of the target computer vision repositories informing our **Evidence-Based Vision Model Recommendation Engine**.

---

## 1. Repository 1: `ayo-aa/cnn-vs-vit-model-selector`

- **URL**: [https://github.com/ayo-aa/cnn-vs-vit-model-selector](https://github.com/ayo-aa/cnn-vs-vit-model-selector)
- **License**: MIT
- **Primary Focus**: Controlled CIFAR-100 study comparing a custom Residual CNN and a compact Vision Transformer under strictly matched parameter budgets (~960K weights).

### Key Findings & Assets
1. **Model Matching**:
   - `Scratch Residual CNN`: 959,452 parameters.
   - `Scratch Compact ViT`: 955,236 parameters (4×4 patch size, 7 transformer layers, 4 attention heads).
2. **Empirical Outcomes**:
   - The CNN outperforms the scratch ViT in clean Top-1 accuracy (60.96% ± 0.18pp vs. 55.62% ± 0.37pp).
   - ViT exhibits significantly superior calibration: Expected Calibration Error (ECE) is 0.086 vs. 0.142 for the CNN.
   - Under Gaussian noise ($\sigma = 0.1$), the scratch ViT retains **29.85%** accuracy compared to **24.12%** for the CNN, demonstrating that ViT self-attention retains spatial representations better under high-frequency degradation.
   - Inference latency: CNN is ~2.2× faster (1.84 ms vs 4.12 ms on Tesla T4 for batch size 64).
3. **Reused Architecture**:
   - Parameter-matched candidate registry concepts (`catalog.py` and `schema.py`).
   - Weighted multi-attribute utility theory (MAUT) decision logic (`selector.py`).
4. **Current Status & Pending Studies**:
   - Scratch study complete with 3 random seeds (42, 43, 44).
   - Transfer learning study (ResNet-10t vs DeiT-Tiny) is designated as pending in the repository.

---

## 2. Repository 2: `seanzero7/Cifar10c-Robustness-Study`

- **URL**: [https://github.com/seanzero7/Cifar10c-Robustness-Study](https://github.com/seanzero7/Cifar10c-Robustness-Study)
- **License**: Code open; model weights Apache-2.0 via `timm` and HuggingFace.
- **Primary Focus**: Inference-only systematic corruption robustness evaluation across CIFAR-10 and CIFAR-10-C (19 corruptions × 5 severities = 95 test conditions).

### Key Findings & Assets
1. **Models Evaluated**:
   - `ResNet-50` (pretrained on ImageNet-1K, fine-tuned on CIFAR-10, 25.6M parameters, 4.1G FLOPs).
   - `ViT-Base/16` (pretrained on ImageNet-21K, fine-tuned on CIFAR-10, 86.6M parameters, 17.6G FLOPs).
   - `ConvNeXt-Tiny` (modernized depthwise CNN, 28.6M parameters, 4.5G FLOPs).
2. **Preprocessing Details**:
   - All models upscale native 32×32 CIFAR images to 224×224 with standard ImageNet normalization.
3. **Empirical Outcomes**:
   - ViT-Base/16 is the most robust overall (mean corrupted accuracy: 84.62%, drop: 13.26pp).
   - ConvNeXt-Tiny achieves competitive robustness (mean corrupted accuracy: 79.82%, drop: 16.90pp) at roughly a third of ViT-Base's parameter and FLOP footprint.
   - ResNet-50 suffers greater degradation under high severities (mean corrupted accuracy: 74.24%, drop: 21.07pp).
   - Noise corruptions cause the highest drop across all architectures; ViT is substantially more resilient to impulse and speckle noise.

---

## 3. Repository 3: `facebookresearch/mvit` (Meta AI)

- **URL**: [https://github.com/facebookresearch/mvit](https://github.com/facebookresearch/mvit)
- **License**: Apache-2.0
- **Primary Focus**: Multiscale Vision Transformers v2 for Image Recognition (CVPR 2022).

### Key Findings & Assets
1. **Core Innovation**:
   - Pooling attention mechanism that scales spatial resolutions while increasing channel dimensions across stages, mirroring the hierarchical inductive bias of CNNs while leveraging self-attention.
2. **Official Model Checkpoints**:
   - `MViTv2-T` (Tiny): 24M parameters, 4.7G FLOPs, 82.3% ImageNet-1K Top-1.
   - `MViTv2-S` (Small): 35M parameters, 7.0G FLOPs, 83.6% ImageNet-1K Top-1.
   - `MViTv2-B` (Base): 52M parameters, 10.2G FLOPs, 84.4% ImageNet-1K Top-1.
3. **Hardware & Deployment Profile**:
   - PyTorch native implementation with optimized CUDA kernels. Checkpoints available via Facebook Public Files.
   - Highly suitable for dense downstream tasks (detection, segmentation) and high-resolution classification.

---

## 4. Repository 4: `hendrycks/robustness`

- **URL**: [https://github.com/hendrycks/robustness](https://github.com/hendrycks/robustness)
- **License**: Apache-2.0
- **Primary Focus**: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (ICLR 2019).

### Key Findings & Assets
1. **Taxonomy of Corruptions**:
   - **Noise**: Gaussian, Shot, Impulse, Speckle.
   - **Blur**: Defocus, Glass, Motion, Zoom, Gaussian Blur.
   - **Weather**: Snow, Frost, Fog, Brightness, Spatter.
   - **Digital**: Contrast, Elastic Transform, Pixelate, JPEG Compression, Saturate.
2. **Evaluation Standard**:
   - Defines Mean Corruption Error (mCE) normalized against AlexNet.
   - Validates that corruptions must be evaluated across all 5 severity levels to observe monotonic breakdown behavior.
