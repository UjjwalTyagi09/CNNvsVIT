# License, Data Provenance, and Governance Audit

This document establishes the licensing terms, redistribution compliance, and provenance of models, evaluation datasets, and codebases included in the Vision Model Recommendation System.

---

## 1. Codebase & Repository Licenses

| Repository / Asset | Code License | Weight License | Commercial Use Allowed | Attribution Requirements |
| :--- | :--- | :--- | :--- | :--- |
| `ayo-aa/cnn-vs-vit-model-selector` | MIT | MIT | Yes | Retain copyright notice & permission notice |
| `seanzero7/Cifar10c-Robustness-Study` | Open (Unlicensed/Academic) | Apache-2.0 (`timm`, HuggingFace) | Yes (Code academic, weights commercial) | Standard model card citations |
| `facebookresearch/mvit` | Apache-2.0 | Apache-2.0 | Yes | Include Apache 2.0 license & NOTICE file |
| `hendrycks/robustness` | Apache-2.0 | Apache-2.0 | Yes | Include Apache 2.0 notice & paper citation |

---

## 2. Model Checkpoint Provenance & Weights Accessibility

1. **`resnet50_cifar10_timm`**:
   - Checkpoint: `timm` repository / HuggingFace Hub (`edadaltocg/resnet50_cifar10`).
   - Distribution: Publicly hosted, zero-friction automated download via `timm.create_model`.
   - Size: ~98 MB.
2. **`vit_base_16_cifar10_hf`**:
   - Checkpoint: HuggingFace Hub (`aaraki/vit-base-patch16-224-in21k-finetuned-cifar10`).
   - Distribution: Publicly hosted, standard HuggingFace pipeline.
   - Size: ~330 MB.
3. **`convnext_tiny_cifar10_timm`**:
   - Checkpoint: `timm` / HuggingFace model cache.
   - Distribution: Publicly hosted.
   - Size: ~110 MB.
4. **`scratch_cnn_cifar100` & `scratch_vit_cifar100`**:
   - Checkpoint: Stored externally on Google Drive by author; not hosted in Git LFS.
   - Evidence Integrity: Full summary evaluation results JSON (76 KB) is checked directly into the git repo (`reports/cifar100-random-init-measured.summary.json`). Recommendation engine utilizes verified empirical outputs without requiring checkpoint redownload.
5. **`mvitv2_tiny_in1k`, `mvitv2_small_in1k`, `mvitv2_base_in1k`**:
   - Checkpoint: Official Meta AI public file servers (`dl.fbaipublicfiles.com/mvit/mvitv2_models/...`).
   - Size: 96 MB, 140 MB, 200 MB.
   - Checkpoint verification: SHA-256 sums published in official repository.

---

## 3. Benchmark Dataset Provenance

1. **CIFAR-10 & CIFAR-100** (Krizhevsky, 2009):
   - License: Open research/educational use.
   - Size: 50,000 training images, 10,000 test images across 10 / 100 classes.
2. **CIFAR-10-C & CIFAR-100-C** (Hendrycks & Dietterich, 2019):
   - License: Creative Commons Attribution 4.0 International (CC BY 4.0).
   - Storage: Zenodo open repository.
3. **ImageNet-1K (ILSVRC 2012)**:
   - License: Custom non-commercial research license (Standard vision academic benchmark).
   - Validation Set: 50,000 images, 1,000 classes.

---

## 4. Architectural Separation Guarantee

The recommendation engine strictly consumes **empirical metadata and evaluation logs** as input, decoupling the decision pipeline from model training or direct weight execution. This ensures:
- 100% operational reliability on resource-constrained environments (CPU, laptop, cloud container).
- Zero dependency on massive dataset storage or GPU compute for running recommendations.
- Clean separation between evidence provenance and recommendation logic.
