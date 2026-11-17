# Evidence-Based Vision Model Recommendation System

> **Research Project**: CNN vs ViT vs MViT, Dataset Suitability, and Robustness-Aware Model Selection  
> **Core Objective**: Select and recommend the optimal existing computer vision architecture for any given operational problem statement using empirical benchmark evidence, without requiring redundant model training.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54%2B-FF4B4B.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-22%20Passed-emerald.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20Apache--2.0-indigo.svg)](docs/license_and_data_audit.md)

---

## 🎯 Executive Summary & Research Questions

Practitioners frequently struggle with critical architectural decisions when deploying computer vision models:
1. **Which architecture should I choose: CNN, ViT, or MViT?**
2. **Which model performs better with limited data vs large datasets?**
3. **Which model is more robust to sensor noise, adverse weather, and real-world corruptions?**
4. **What are the concrete trade-offs between accuracy, latency, memory, and model complexity?**

Rather than training another classifier, this system operates as an **evidence-based decision engine**. It synthesizes empirical evaluation logs, corruption robustness curves, and hardware efficiency measurements from verified research repositories to provide mathematically principled recommendations backed by transparent explanations.

---

## 🏛️ Architecture Paradigm Comparison

| Dimension | Residual CNN (e.g. ResNet, ConvNeXt) | Vision Transformer (ViT-Base) | Multiscale ViT (MViTv2) |
| :--- | :--- | :--- | :--- |
| **Inductive Bias** | **Strong**: Spatial locality & translation invariance | **Minimal**: Token patch relations learned globally | **Hybrid**: Multiscale hierarchical pooling + self-attention |
| **Small Data / Scratch** | **Superior**: Fast convergence on limited samples | **Poor**: Requires massive pretraining or transfer | **Moderate**: Optimal with transfer learning |
| **Noise & Corruptions** | **Moderate**: Vulnerable to high-frequency sensor noise | **Superior**: Global attention preserves shape semantics | **High**: Multiscale features resist multi-frequency blur |
| **Calibration (ECE)** | **Fair**: Tends toward overconfidence | **High**: Superior probability calibration | **Good**: Well-calibrated prediction distributions |
| **Hardware Efficiency** | **High**: Low latency on CPUs and edge devices | **Moderate**: Higher FLOPs and memory footprint | **Balanced**: Pooling attention reduces FLOPs on GPUs |

---

## 📊 Empirical Findings from Verified Benchmark Repositories

### 1. CIFAR-100 Parameter-Matched Scratch Study (`ayo-aa/cnn-vs-vit-model-selector`)
- **Strict Matching**: Custom Residual CNN (959,452 params) vs Compact ViT (955,236 params), evaluated over 3 random seeds.
- **Clean Accuracy**: CNN achieves **60.96% ± 0.18pp** vs ViT's **55.62% ± 0.37pp** (+5.34pp CNN advantage due to local convolutional priors).
- **Gaussian Noise Robustness**: Under noise ($\sigma = 0.1$), Compact ViT retains **29.85%** accuracy vs the CNN's **24.12%** (+5.73pp ViT advantage).
- **Calibration**: ViT achieves an Expected Calibration Error of **0.086 vs 0.142** for the CNN.
- **Latency**: CNN is **~2.2× faster** (1.84ms vs 4.12ms on Tesla T4 for batch size 64).

### 2. CIFAR-10-C Systematic Corruption Study (`seanzero7/Cifar10c-Robustness-Study`)
- **Full Benchmark**: 3 models evaluated across **19 corruption types × 5 severities = 95 test conditions**.
- **ViT-Base/16**: Lowest degradation (**84.62% mean corrupted accuracy**, drop: 13.26pp).
- **ConvNeXt-Tiny**: Highly competitive modern CNN (**79.82% mean corrupted accuracy**, drop: 16.90pp) with only 28.6M parameters and 4.5G FLOPs.
- **ResNet-50**: Drops to **59.5% at Severity 5** (21.07pp mean drop), showing vulnerability to severe noise and blur.

### 3. Official MViTv2 Benchmark (`facebookresearch/mvit`)
- **Multiscale Pooling Attention**: MViTv2-T (24M, 4.7G FLOPs, 82.3% ImageNet-1K), MViTv2-S (35M, 7.0G FLOPs, 83.6%), MViTv2-B (52M, 10.2G FLOPs, 84.4%).
- Bridges hierarchical feature representation and transformer attention for dense, high-resolution perception.

---

## 🔬 System Architecture

```
reserch/
├── app/                           # Streamlit Web Application (Dark Glassmorphism)
│   ├── app.py                     # 6-view interactive dashboard & decision center
│   ├── charts.py                  # Plotly radar, severity curves, taxonomy charts
│   └── style.css                  # Modern CSS design tokens & animations
├── data/                          # Evidence Registry
│   ├── models.json                # Model specifications & checkpoints metadata
│   ├── evidence.json              # Verified quantitative evaluation metrics
│   ├── sources.json               # Provenance, licenses, and repository links
│   └── evaluation_protocols.json  # Benchmark protocol definitions & comparability rules
├── docs/                          # In-depth Research Documentation
│   ├── repository_audit.md        # Detailed audit of the 4 benchmark repositories
│   ├── license_and_data_audit.md  # Redistribution licenses and weights provenance
│   ├── research_gaps.md           # Research gaps, caveats, and mitigation strategies
│   └── evidence_comparability.md  # Comparability matrix & MAUT mathematical framework
├── src/                           # Core Recommendation Engine
│   ├── registry.py                # Model & evidence data access layer
│   ├── requirements_parser.py     # Natural language query and constraint extractor
│   ├── constraint_filter.py       # Hard operational constraint filtering
│   ├── scorer.py                  # Multi-attribute utility (MAUT) ranking
│   ├── explainer.py               # Natural language rationale & trade-off generator
│   ├── report_generator.py        # Markdown and JSON report generator
│   └── utils.py                   # Normalization and formatting utilities
├── tests/                         # Comprehensive pytest test suite (22 tests)
│   ├── test_registry.py           # Registry integrity and queries
│   ├── test_parser.py             # Query parsing and constraint extraction
│   ├── test_filter.py             # Operational filtering and fallback behavior
│   ├── test_scorer.py             # MAUT utility scoring and normalization
│   ├── test_recommender.py        # End-to-end pipeline and exports
│   └── test_scenarios.py          # Real-world benchmark scenarios
├── scripts/
│   └── ingest_evidence.py         # Evidence registry validation and diagnostics
└── requirements.txt
```

---

## 📐 Mathematical Decision Framework (MAUT)

Candidate models are scored using Multi-Attribute Utility Theory (MAUT):

$$S(m) = \sum_{k \in \mathcal{K}} w_k \cdot \phi_k(m)$$

Where:
- $\mathcal{K} = \{\text{Accuracy}, \text{Robustness}, \text{Latency}, \text{DataEfficiency}, \text{Calibration}\}$
- $w_k \ge 0$ is the user preference weight ($\sum w_k = 1.0$)
- $\phi_k(m) \in [0, 1]$ is the min-max normalized utility score for model $m$ on attribute $k$.

For inverse metrics (latency, ECE, corruption drop):
$$\phi_k(m) = \frac{\max_j x_k(j) - x_k(m)}{\max_j x_k(j) - \min_j x_k(j)}$$

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
# Clone the repository and enter directory
cd reserch

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Launch the Interactive Web Application
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501` to explore:
- **Interactive Problem Builder**: Enter queries like *"Obstacle detection on drone CPU under 25ms"* or 1-click presets.
- **Winner Scorecard**: Inspect MAUT scores, radar comparison, and margin over runner-up.
- **Head-to-Head Comparison**: Select any 2 models for side-by-side metric breakdown.
- **Robustness Deep Dive**: Plot interactive CIFAR-10-C degradation curves across severities 1 to 5.
- **Export Center**: Download generated audit reports as Markdown or JSON.

### 3. Run Automated Tests
```bash
python -m pytest tests/ -v
```
Runs 22 unit and scenario integration tests validating:
- Edge drone camera deployment under strict latency constraints.
- Adverse weather outdoor surveillance with heavy fog and sensor noise.
- Small dataset scratch training with inductive bias validation.
- High-throughput hierarchical classification.

### 4. Programmatic Python SDK Usage
```python
from src.recommender import VisionModelRecommender

# Initialize recommender
recommender = VisionModelRecommender()

# Run recommendation
result = recommender.recommend(
    query="Autonomous drone camera with sensor noise and real-time CPU latency requirements",
    overrides={"max_latency_ms": 25.0, "hardware_target": "cpu"}
)

winner = result["top_recommendation"]
print(f"Recommended Model: {winner['display_name']} ({winner['architecture_family']})")
print(f"Score: {winner['score']} / 100")
print(f"Rationale: {result['explanation']['core_rationale']}")
```

---

## 📜 Licenses & Attribution

- Benchmark data and models derived from public research repositories:
  - `ayo-aa/cnn-vs-vit-model-selector`: MIT License.
  - `seanzero7/Cifar10c-Robustness-Study`: Open Research (Weights: Apache-2.0 via `timm` / HuggingFace).
  - `facebookresearch/mvit`: Apache-2.0 License (Meta AI).
  - `hendrycks/robustness`: Apache-2.0 License.
- Detailed audit and redistribution governance documented in [docs/license_and_data_audit.md](docs/license_and_data_audit.md).
