# Research Gaps & Evidence Mitigation Strategy

This document articulates the empirical and methodological gaps uncovered during the multi-repository audit, along with the mitigation and confidence-gating strategies integrated into the recommendation engine.

---

## 1. Identified Research Gaps

### Gap 1: Absence of Direct Corruption Robustness for MViT Checkpoints
- **Description**: The official `facebookresearch/mvit` repository publishes ImageNet-1K clean accuracy (82.3% to 84.4%) and FLOPs, but does not include CIFAR-10-C or ImageNet-C benchmark runs.
- **Impact**: We cannot plot exact per-corruption curves (e.g. glass blur at severity 3) for MViTv2 alongside ResNet-50 or ViT-Base.
- **Mitigation**:
  1. Robustness for MViT is flagged with `estimated` provenance rather than `directly_measured`.
  2. The recommendation engine applies an architectural confidence penalty (-15% confidence factor) when a user specifically demands granular corruption-type queries for MViT.
  3. Qualitative properties derived from the CVPR 2022 paper (multiscale attention resilience to structural deformations) are exposed in the trade-off explanations.

### Gap 2: Incomplete Transfer Learning Study in Repo 1
- **Description**: In `ayo-aa/cnn-vs-vit-model-selector`, the random initialization scratch study on CIFAR-100 is complete across 3 seeds, but the transfer learning study (`ResNet-10t` vs `DeiT-Tiny`) remains pending.
- **Impact**: Scratch vs transfer comparisons cannot be directly derived within that specific parameter-matched repository.
- **Mitigation**:
  1. The engine clearly distinguishes "Scratch vs Scratch" (Repo 1) from "Pretrained Transfer" (Repo 2).
  2. When users query for small dataset fine-tuning, the engine routes to Pretrained Transfer evidence while noting the inductive bias advantages demonstrated in the scratch study.

### Gap 3: Absence of Standardized Hardware Latency Profiling in Repo 2
- **Description**: `seanzero7/Cifar10c-Robustness-Study` measured Top-1, Top-5, and softmax confidence across 95 corruptions, but did not measure wall-clock latency (ms) or memory footprint (MB) across standard devices.
- **Impact**: Hardware latency comparisons between ResNet-50, ViT-Base, and ConvNeXt-Tiny required complementary profiling data.
- **Mitigation**:
  1. Latency and memory benchmarks are calculated and cross-referenced from standard PyTorch/timm benchmarking runs on CPU and GPU (Tesla T4 / modern x86 CPU).
  2. Latency values in `data/evidence.json` explicitly indicate the hardware target (CPU ms vs GPU ms).

### Gap 4: Cross-Dataset and Cross-Resolution Incommensurability
- **Description**: Raw accuracy cannot be directly contrasted across CIFAR-10 (10 classes), CIFAR-100 (100 classes), and ImageNet-1K (1000 classes), nor between 32×32 and 224×224 resolutions.
- **Mitigation**:
  1. Strict protocol scoping in the constraint filter: candidates are grouped into peer cohorts.
  2. Relative metrics (e.g., % retention under noise, calibration ECE, relative corruption drop $\Delta_{corr}$) are prioritized over raw Top-1 accuracy when evaluating across architectural families.

---

## 2. Summary of Mitigation Rules in the Recommendation Engine

| Scenario | Default Fallback | Confidence Adjustment | UI Presentation |
| :--- | :--- | :--- | :--- |
| User specifies high noise tolerance with small dataset | Recommends Compact ViT or ConvNeXt depending on scratch vs transfer need | High (Verified on CIFAR-100 & CIFAR-10-C) | Verified Evidence Badge |
| User requests MViT on noisy camera feed | Recommends MViT with architectural justification | Medium (Flagged: Paper-derived estimation) | "Estimated Robustness" warning tag |
| User asks for edge CPU deployment (<20ms latency) | Recommends Scratch CNN or ConvNeXt-Tiny | High (Verified latency profiles) | Real-time budget satisfied indicator |
