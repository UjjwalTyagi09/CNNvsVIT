"""
Streamlit Web Application for Evidence-Based Vision Model Recommendation System.
Architectures: CNN vs ViT vs MViT.
"""

import sys
from pathlib import Path

# Add project root and app directory to path
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
for p in [str(ROOT_DIR), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
from src.recommender import VisionModelRecommender
from src.utils import format_latency, format_params, format_percentage

try:
    from charts import (
        create_radar_chart,
        create_score_contribution_bar,
        create_severity_curve_chart,
        create_category_bar_chart,
        create_per_corruption_bar_chart
    )
except (ModuleNotFoundError, ImportError):
    from app.charts import (
        create_radar_chart,
        create_score_contribution_bar,
        create_severity_curve_chart,
        create_category_bar_chart,
        create_per_corruption_bar_chart
    )

# 1. Page Configuration
st.set_page_config(
    page_title="Vision Model Recommender: CNN vs ViT vs MViT",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom CSS
css_path = Path(__file__).resolve().parent / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3. Initialize Engine in Session State
@st.cache_resource
def get_recommender():
    return VisionModelRecommender()

recommender = get_recommender()

# 4. Sidebar Navigation
st.sidebar.markdown(
    """
    <div style="padding: 0.5rem 0 1rem 0;">
        <span style="font-size: 1.4rem; font-weight: 800; background: linear-gradient(135deg, #818cf8, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            👁️ VisionRecommender
        </span>
        <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.2rem;">
            Evidence-Based Model Selection
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

nav_page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview & Dashboard",
        "🎯 Problem Builder & Recommender",
        "⚔️ Head-to-Head Comparison",
        "🛡️ Robustness & Corruptions",
        "📚 Evidence Explorer & Provenance",
        "📄 Export & Audit Report"
    ],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
st.sidebar.markdown(
    """
    - **Models Registered**: `8 architectures`
    - **Benchmark Corruptions**: `19 types × 5 severities`
    - **Verified Repos**: `4 audited`
    - **Engine Status**: `🟢 Active & Verified`
    """
)

st.sidebar.markdown("---")
st.sidebar.caption("Google DeepMind Vision Research Pair Programming")


# ==============================================================================
# PAGE 1: OVERVIEW & DASHBOARD
# ==============================================================================
if nav_page == "🏠 Overview & Dashboard":
    st.markdown('<div class="gradient-title">Vision Architecture Decision Paradigm</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">An evidence-driven framework comparing CNNs, Vision Transformers (ViT), '
        'and Multiscale Vision Transformers (MViT) across real-world datasets, corruption robustness, and hardware constraints.</div>',
        unsafe_allow_html=True
    )

    # 4 Quick Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="metric-glass-card">
                <div class="metric-label">Audited Checkpoints</div>
                <div class="metric-val">8 Models</div>
                <div class="metric-sub">CNN • ViT • MViTv2</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="metric-glass-card">
                <div class="metric-label">CIFAR-10-C Corruptions</div>
                <div class="metric-val">95 Evals</div>
                <div class="metric-sub">19 Corruptions × 5 Severities</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            """
            <div class="metric-glass-card">
                <div class="metric-label">Scratch Study Baseline</div>
                <div class="metric-val">~960K Params</div>
                <div class="metric-sub">Strictly Matched CIFAR-100</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            """
            <div class="metric-glass-card">
                <div class="metric-label">Decision Methodology</div>
                <div class="metric-val">MAUT</div>
                <div class="metric-sub">Multi-Attribute Utility Theory</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🏛️ Architecture Family Comparison Matrix")
    st.markdown(
        "A rigorous synthesis of empirical research findings regarding structural priors, inductive biases, and failure modes:"
    )

    matrix_data = {
        "Property": [
            "Primary Inductive Bias",
            "Attention Receptive Field",
            "Small Dataset / Scratch Training",
            "High-Frequency Noise Resilience",
            "Confidence Calibration (ECE)",
            "Inference Throughput & Latency",
            "Memory & Hardware Footprint",
            "Best Suited Deployment"
        ],
        "Residual CNN": [
            "Strong (Spatial locality & translation invariance)",
            "Local receptive field expanding with depth",
            "Excellent (Quick convergence with few samples)",
            "Moderate (Sensitive to sensor noise & texture shifts)",
            "Fair (Tendency toward overconfidence)",
            "High (Optimized hardware kernels, low FLOPs)",
            "Minimal memory requirement (<500MB)",
            "Edge devices, microcontrollers, real-time robotics"
        ],
        "Vision Transformer (ViT)": [
            "Minimal (Learns spatial relations from patch tokens)",
            "Global multi-head self-attention across all patches",
            "Poor from scratch (Requires massive pretraining / transfer)",
            "Superior (Global context retains shape under corruption)",
            "Excellent (Low Expected Calibration Error)",
            "Moderate to Slow on CPU (Higher FLOPs and attention scaling)",
            "Moderate to High (>1GB memory for ViT-Base)",
            "High-accuracy cloud servers, adverse weather perception"
        ],
        "Multiscale ViT (MViTv2)": [
            "Hybrid (Hierarchical resolution scaling + Attention)",
            "Multiscale pooling attention across temporal/spatial stages",
            "Moderate (Benefits strongly from ImageNet pretraining)",
            "High (Preserves structural semantics across scales)",
            "Good (Strong confidence bounds)",
            "High throughput on GPUs (Pooling attention reduces FLOPs)",
            "Moderate memory footprint (~500MB)",
            "High-resolution classification, object detection backbones"
        ]
    }
    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)

    st.markdown("### 🛡️ Empirical Benchmark Insights from Verified Repositories")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            #### 1. The Inductive Bias Dilemma (CIFAR-100 Scratch Study)
            - **Finding**: When restricted to ~960k parameters and trained from scratch with random initialization, **Scratch Residual CNN achieves 60.96% clean accuracy vs 55.62% for Compact ViT** (+5.34pp advantage).
            - **The Twist**: Under severe Gaussian noise, **Compact ViT retains 29.85% accuracy vs 24.12% for the CNN**, demonstrating that even small transformers learn representations more resistant to high-frequency degradation.
            - **Calibration**: Compact ViT achieves an **ECE of 0.086 vs 0.142 for the CNN**, making it far more trustworthy for safety-critical classification.
            """
        )
    with col_b:
        st.markdown(
            """
            #### 2. Systematic Corruption Degradation (CIFAR-10-C Study)
            - **Finding**: Evaluated across 19 corruptions at 5 severities, **ViT-Base/16 dominates overall robustness (84.6% mean corrupted Top-1)** with only 13.2pp drop.
            - **ConvNeXt-Tiny Competitiveness**: ConvNeXt-Tiny achieves **79.8% mean corrupted accuracy** while consuming only 28.6M parameters and 4.5G FLOPs—providing ViT-like robustness at a CNN latency budget.
            - **ResNet-50 Breakdown**: ResNet-50 degrades to **59.5% at Severity 5** (21.1pp mean drop), proving that classic convolutions suffer heavy vulnerability to extreme sensor noise and blur.
            """
        )


# ==============================================================================
# PAGE 2: PROBLEM BUILDER & RECOMMENDATION
# ==============================================================================
elif nav_page == "🎯 Problem Builder & Recommender":
    st.markdown('<div class="gradient-title">Problem Builder & Model Selection Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Specify your operational problem statement in natural language or customize '
        'hardware, latency, and corruption constraints below.</div>',
        unsafe_allow_html=True
    )

    # Preset Templates
    st.markdown("#### ⚡ 1-Click Problem Presets")
    p1, p2, p3, p4 = st.columns(4)
    preset_query = ""
    if p1.button("🤖 Edge Robot Drone", help="Real-time CPU inference under 25ms"):
        preset_query = "Real-time obstacle detection for an autonomous drone on embedded CPU with latency under 25ms."
    if p2.button("🌧️ Harsh Outdoor Weather", help="Severe fog, snow, and sensor noise"):
        preset_query = "Outdoor camera surveillance in adverse weather dealing with severe fog, snow, and sensor noise. Accuracy and robustness are essential."
    if p3.button("🧪 Small Medical Dataset", help="1,500 samples, training from scratch with calibration"):
        preset_query = "Training from scratch on limited medical imaging dataset with only 1500 images. Calibration and reliability are critical."
    if p4.button("🚀 Cloud High-Throughput", help="Server GPU, high resolution, multiscale hierarchy"):
        preset_query = "High-accuracy multiscale image classification on GPU server with large dataset and high resolution."

    # Problem Statement Input
    user_query = st.text_area(
        "Enter your computer vision problem statement or requirements:",
        value=preset_query if preset_query else "Select an optimal model for outdoor security cameras with sensor noise, running on edge hardware with low latency.",
        height=85
    )

    # Advanced Constraints & Overrides
    with st.expander("🛠️ Advanced Operational Constraints & Priority Tuning", expanded=False):
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("##### 📦 Dataset & Hardware Constraints")
            override_dataset_size = st.selectbox(
                "Dataset Size Regime:",
                ["Auto-Detect from Query", "small (< 5,000 samples)", "medium (5,000 - 50,000 samples)", "large (> 50,000 samples)"],
                index=0
            )
            override_regime = st.selectbox(
                "Training Regime:",
                ["Auto-Detect", "transfer (Fine-tuning Pretrained Weights)", "scratch (Random Initialization / No Pretraining)"],
                index=0
            )
            override_hardware = st.selectbox(
                "Hardware Deployment Target:",
                ["any", "cpu", "edge", "gpu"],
                index=0
            )
            override_latency = st.slider("Max Inference Latency Budget (ms):", min_value=5, max_value=100, value=50, step=5)
            override_params = st.slider("Max Parameter Budget (Millions):", min_value=1, max_value=100, value=90, step=5)

        with c_right:
            st.markdown("##### ⚖️ Decision Priority Weights (MAUT)")
            w_acc = st.slider("Accuracy Weight:", 0.0, 1.0, 0.30, 0.05)
            w_rob = st.slider("Robustness Weight:", 0.0, 1.0, 0.30, 0.05)
            w_lat = st.slider("Latency Speed Weight:", 0.0, 1.0, 0.20, 0.05)
            w_data = st.slider("Data Efficiency Weight:", 0.0, 1.0, 0.10, 0.05)
            w_cal = st.slider("Calibration (ECE) Weight:", 0.0, 1.0, 0.10, 0.05)

            st.markdown("##### 🌪️ Corruptions of Primary Interest")
            corr_noise = st.checkbox("Sensor Noise (Gaussian, Shot, Impulse, Speckle)", value=True)
            corr_blur = st.checkbox("Blur (Defocus, Motion, Zoom, Glass)", value=False)
            corr_weather = st.checkbox("Weather (Fog, Snow, Frost, Brightness)", value=False)

    # Prepare Overrides Dict
    overrides = {}
    if override_dataset_size != "Auto-Detect from Query":
        overrides["dataset_size"] = override_dataset_size.split()[0]
    if override_regime != "Auto-Detect":
        overrides["training_regime"] = override_regime.split()[0]
    if override_hardware != "any":
        overrides["hardware_target"] = override_hardware
    if override_latency < 100:
        overrides["max_latency_ms"] = float(override_latency)
    if override_params < 90:
        overrides["max_params"] = int(override_params * 1_000_000)

    # Weights override
    overrides["weight_accuracy"] = w_acc
    overrides["weight_robustness"] = w_rob
    overrides["weight_latency"] = w_lat
    overrides["weight_data_efficiency"] = w_data
    overrides["weight_calibration"] = w_cal

    # Target corruptions
    corrs = []
    if corr_noise:
        corrs.extend(["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"])
    if corr_blur:
        corrs.extend(["defocus_blur", "motion_blur", "zoom_blur", "gaussian_blur"])
    if corr_weather:
        corrs.extend(["fog", "snow", "frost", "brightness"])
    if corrs:
        overrides["corruption_types"] = corrs

    # Execute Recommendation
    result = recommender.recommend(user_query, overrides=overrides)
    top = result.get("top_recommendation")
    runner_up = result.get("runner_up")
    ranked = result.get("ranked_candidates", [])
    excluded = result.get("excluded_candidates", [])
    explanation = result.get("explanation", {})

    st.markdown("---")

    if top:
        top_fam = top.get("architecture_family", "CNN")
        badge_class = f"badge-{top_fam.lower()}"
        
        # Winner Hero Display
        st.markdown(
            f"""
            <div class="winner-hero-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.5rem;">
                    <div>
                        <span class="badge-tag {badge_class}">{top_fam} Architecture</span>
                        <span class="badge-tag badge-confidence">Confidence: {explanation.get('confidence_level')} ({explanation.get('confidence_score')}%)</span>
                        <h2 style="margin: 0.4rem 0 0.2rem 0; font-size: 2rem; color: #ffffff;">{top.get('display_name')}</h2>
                        <div style="color: #94a3b8; font-size: 0.95rem;">Checkpoint Source: <code>{top.get('checkpoint_source')}</code> | License: <code>{top.get('weights_license')}</code></div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #a5b4fc;">MAUT Score</div>
                        <div style="font-size: 2.8rem; font-weight: 800; color: #38bdf8; line-height: 1;">{top.get('score')} <span style="font-size: 1.1rem; color: #94a3b8;">/ 100</span></div>
                        <div style="font-size: 0.85rem; color: #34d399; margin-top: 0.2rem;">+ {top.get('margin_over_runner_up', 0.0):.2f} pts margin over runner-up</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Visual Score Analysis (Radar + Score Contributions)
        v_col1, v_col2 = st.columns([1, 1])
        with v_col1:
            st.plotly_chart(create_radar_chart(ranked), use_container_width=True)
        with v_col2:
            st.plotly_chart(create_score_contribution_bar(top), use_container_width=True)

        # Tabbed Explanations
        t1, t2, t3, t4 = st.tabs(["💡 Decision Rationale", "🏛️ Architecture Fit", "⚖️ Trade-Offs", "🔄 Sensitivity"])
        with t1:
            st.markdown(f"**Core Winning Driver**: {explanation.get('core_rationale')}")
            if explanation.get("runner_up_comparison"):
                st.info(explanation.get("runner_up_comparison"))
        with t2:
            st.markdown(explanation.get("architecture_analysis", ""))
        with t3:
            st.markdown("##### Key Engineering Trade-Offs:")
            for to in explanation.get("trade_offs", []):
                st.markdown(f"- {to}")
        with t4:
            st.markdown(f"**Sensitivity Analysis**: {explanation.get('sensitivity_analysis')}")

        # Ranked Candidates Table
        st.markdown("### 📋 Full Candidate Evaluation Leaderboard")
        table_rows = []
        for c in ranked:
            raw = c.get("raw_metrics", {})
            table_rows.append({
                "Rank": f"#{c.get('rank')}",
                "Model": c.get("display_name"),
                "Family": c.get("architecture_family"),
                "MAUT Score": c.get("score"),
                "Clean Top-1": f"{raw.get('clean_top1', 0):.1f}%",
                "Robustness": f"{raw.get('robustness_metric', 0):.1f}%",
                "Latency": f"{raw.get('latency_ms', 0):.1f} ms",
                "Parameters": format_params(c.get("parameter_count", 0)),
                "FLOPs": c.get("flops", "N/A"),
                "Status": "✅ Passed"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # Excluded Models
        if excluded:
            with st.expander(f"⚠️ Excluded Candidates ({len(excluded)} models failed hard constraints)"):
                ex_rows = []
                for ex in excluded:
                    ex_rows.append({
                        "Model": ex.get("display_name"),
                        "Family": ex.get("architecture_family"),
                        "Parameters": format_params(ex.get("parameter_count", 0)),
                        "Reasons for Exclusion": "; ".join(ex.get("exclusion_reasons", []))
                    })
                st.dataframe(pd.DataFrame(ex_rows), use_container_width=True, hide_index=True)
    else:
        st.error("No candidate models passed your constraints. Try relaxing latency or parameter bounds.")


# ==============================================================================
# PAGE 3: HEAD-TO-HEAD COMPARISON
# ==============================================================================
elif nav_page == "⚔️ Head-to-Head Comparison":
    st.markdown('<div class="gradient-title">Head-to-Head Architectural Face-off</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Select any two vision models to inspect pairwise metrics, FLOP efficiency, '
        'and architectural trade-offs.</div>',
        unsafe_allow_html=True
    )

    models_list = recommender.registry.get_all_models()
    model_opts = {m["model_id"]: f"{m['display_name']} ({m['architecture_family']})" for m in models_list}
    
    col_sel_a, col_sel_b = st.columns(2)
    with col_sel_a:
        mid_a = st.selectbox("Select Model A:", list(model_opts.keys()), index=0, format_func=lambda x: model_opts[x])
    with col_sel_b:
        mid_b = st.selectbox("Select Model B:", list(model_opts.keys()), index=1, format_func=lambda x: model_opts[x])

    cmp = recommender.compare_models(mid_a, mid_b)
    if "error" not in cmp:
        cand_a = cmp["model_a"]
        cand_b = cmp["model_b"]
        ev_a = cand_a.get("evidence", {})
        ev_b = cand_b.get("evidence", {})
        eff_a = ev_a.get("efficiency", {})
        eff_b = ev_b.get("efficiency", {})
        rob_a = ev_a.get("robustness_summary", {})
        rob_b = ev_b.get("robustness_summary", {})

        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(
                f"""
                <div class="metric-glass-card">
                    <span class="badge-tag badge-{cand_a.get('architecture_family', 'cnn').lower()}">{cand_a.get('architecture_family')}</span>
                    <h3 style="margin: 0.3rem 0; color: #fff;">{cand_a.get('display_name')}</h3>
                    <p style="color: #94a3b8; font-size: 0.85rem;">{cand_a.get('notes')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_b:
            st.markdown(
                f"""
                <div class="metric-glass-card">
                    <span class="badge-tag badge-{cand_b.get('architecture_family', 'cnn').lower()}">{cand_b.get('architecture_family')}</span>
                    <h3 style="margin: 0.3rem 0; color: #fff;">{cand_b.get('display_name')}</h3>
                    <p style="color: #94a3b8; font-size: 0.85rem;">{cand_b.get('notes')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Comparative Table
        comp_data = {
            "Metric": [
                "Clean Top-1 Accuracy",
                "Mean Corrupted Accuracy",
                "Trainable Parameters",
                "Compute (FLOPs)",
                "CPU Latency",
                "GPU Latency",
                "Expected Calibration Error (ECE)",
                "Primary Training Regime",
                "Code License"
            ],
            cand_a.get("display_name"): [
                f"{ev_a.get('clean_metrics', {}).get('top1_accuracy', 'N/A')}%",
                f"{rob_a.get('mean_corrupted_top1', rob_a.get('gaussian_noise_accuracy', 'N/A'))}%",
                format_params(cand_a.get("parameter_count", 0)),
                str(cand_a.get("flops", "N/A")),
                format_latency(eff_a.get("cpu_latency_ms", 0)),
                format_latency(eff_a.get("gpu_latency_ms", 0)),
                str(ev_a.get("clean_metrics", {}).get("expected_calibration_error", "N/A")),
                cand_a.get("pretrained_dataset", "N/A"),
                cand_a.get("code_license", "N/A")
            ],
            cand_b.get("display_name"): [
                f"{ev_b.get('clean_metrics', {}).get('top1_accuracy', 'N/A')}%",
                f"{rob_b.get('mean_corrupted_top1', rob_b.get('gaussian_noise_accuracy', 'N/A'))}%",
                format_params(cand_b.get("parameter_count", 0)),
                str(cand_b.get("flops", "N/A")),
                format_latency(eff_b.get("cpu_latency_ms", 0)),
                format_latency(eff_b.get("gpu_latency_ms", 0)),
                str(ev_b.get("clean_metrics", {}).get("expected_calibration_error", "N/A")),
                cand_b.get("pretrained_dataset", "N/A"),
                cand_b.get("code_license", "N/A")
            ]
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 4: ROBUSTNESS & CORRUPTIONS
# ==============================================================================
elif nav_page == "🛡️ Robustness & Corruptions":
    st.markdown('<div class="gradient-title">CIFAR-10-C & Hendrycks Robustness Suite</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Systematic evaluation across 19 corruption types, 5 severity levels, '
        'and 4 distortion categories (Noise, Blur, Weather, Digital).</div>',
        unsafe_allow_html=True
    )

    ev_dict = recommender.registry.get_all_evidence()

    # Severity Degradation Curves
    st.plotly_chart(create_severity_curve_chart(ev_dict), use_container_width=True)

    # Category Bar Chart
    c_cat, c_per = st.columns([1, 1])
    with c_cat:
        st.plotly_chart(create_category_bar_chart(ev_dict), use_container_width=True)
    with c_per:
        target_model = st.selectbox(
            "Select model to view all 19 corruptions (Severity 3):",
            ["vit_base_16_cifar10_hf", "convnext_tiny_cifar10_timm", "resnet50_cifar10_timm"],
            format_func=lambda x: recommender.registry.get_model(x).get("display_name", x)
        )
        st.plotly_chart(create_per_corruption_bar_chart(ev_dict, target_model), use_container_width=True)

    st.markdown("### 🔬 Empirical Robustness Breakdown")
    st.markdown(
        """
        - **Why ViTs Dominate Noise**: Vision Transformers compute all-to-all patch attention without local kernel constraints. When high-frequency Gaussian or shot noise masks pixel-level textures, the self-attention mechanism relies on global semantic relationships, whereas standard CNN kernels suffer immediate filter activation degradation.
        - **ConvNeXt's Receptive Field Advantage**: ConvNeXt-Tiny modernizes CNNs using 7×7 depthwise convolutions and inverted bottleneck blocks. This expanded receptive field mimics ViT attention properties while preserving fast convolutional runtime, achieving **79.8% mean corrupted accuracy** compared to ResNet-50's **74.2%**.
        - **Weather & Digital Resilience**: All architectures maintain relatively high accuracy (>85%) under gentle brightness and saturation shifts, but contrast degradation and glass blur cause steep non-linear accuracy drops across all three models.
        """
    )


# ==============================================================================
# PAGE 5: EVIDENCE EXPLORER & PROVENANCE
# ==============================================================================
elif nav_page == "📚 Evidence Explorer & Provenance":
    st.markdown('<div class="gradient-title">Evidence Explorer & Provenance Registry</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Explore raw empirical benchmark data, repository licenses, author attributions, '
        'and protocol comparability standards.</div>',
        unsafe_allow_html=True
    )

    st.markdown("#### 📂 Verified Source Repositories")
    sources = recommender.registry.get_all_sources()
    for s in sources:
        with st.expander(f"🏛️ {s.get('name')} (License: {s.get('license')})", expanded=False):
            st.markdown(f"**Repository URL**: [{s.get('url')}]({s.get('url')})")
            st.markdown(f"**Authors / Organization**: {s.get('authors')}")
            st.markdown(f"**Description**: {s.get('description')}")
            st.markdown(f"**Evidence Types**: `{', '.join(s.get('evidence_types', []))}`")
            st.markdown(f"**Dataset**: `{s.get('dataset')}` | **Verified**: ✅ {s.get('verified')}")

    st.markdown("#### 📑 Standardized Evaluation Protocols")
    protocols = recommender.registry.get_all_protocols()
    proto_df = []
    for p in protocols:
        proto_df.append({
            "Protocol ID": p.get("protocol_id"),
            "Name": p.get("name"),
            "Dataset": p.get("dataset"),
            "Classes": p.get("classes"),
            "Resolution": p.get("input_resolution"),
            "Hardware": p.get("hardware", "N/A"),
            "Direct Comparables": ", ".join(p.get("valid_direct_comparisons", []))
        })
    st.dataframe(pd.DataFrame(proto_df), use_container_width=True, hide_index=True)

    st.markdown("#### ⚠️ Transparent Research Gap Disclosures")
    st.markdown(
        """
        1. **MViTv2 Corruption Benchmark**: MViTv2 checkpoints from Facebook Research are evaluated on ImageNet-1K. Corruption metrics are estimated based on published multiscale pooling attention literature rather than an in-repo CIFAR-10-C script.
        2. **Transfer Study Pending in Repo 1**: In `cnn-vs-vit-model-selector`, the transfer learning benchmark between ResNet-10t and DeiT-Tiny remains marked as pending. The engine relies on the verified scratch study data.
        3. **Hardware Latency Normalization**: Hardware latencies for Repo 2 models are measured across standard PyTorch/timm CPU and GPU profiles.
        """
    )


# ==============================================================================
# PAGE 6: EXPORT & AUDIT REPORT
# ==============================================================================
elif nav_page == "📄 Export & Audit Report":
    st.markdown('<div class="gradient-title">Export & Audit Report Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Generate, audit, and download research-grade Markdown and JSON reports '
        'documenting the model selection decision trail.</div>',
        unsafe_allow_html=True
    )

    sample_query = "Outdoor drone obstacle perception with severe sensor noise and real-time latency requirements."
    res = recommender.recommend(sample_query)
    
    md_content = recommender.export_report(res, fmt="markdown")
    json_content = recommender.export_report(res, fmt="json")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button(
            label="📥 Download Markdown Report (.md)",
            data=md_content,
            file_name="vision_model_recommendation_report.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_btn2:
        st.download_button(
            label="📥 Download JSON Payload (.json)",
            data=json_content,
            file_name="vision_model_recommendation.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("### 📄 Markdown Report Preview")
    st.markdown(md_content)

    with st.expander("🔍 View Raw JSON Audit Payload"):
        st.code(json_content, language="json")

    st.markdown("### 💻 Programmatic Python SDK Example")
    st.code(
        """from src.recommender import VisionModelRecommender

# Initialize the recommendation engine
recommender = VisionModelRecommender()

# Run recommendation with natural language and optional overrides
result = recommender.recommend(
    query="Autonomous drone with sensor noise and real-time inference on CPU",
    overrides={"max_latency_ms": 25.0}
)

winner = result["top_recommendation"]
print(f"Recommended Model: {winner['display_name']} ({winner['architecture_family']})")
print(f"Score: {winner['score']} / 100")
print(f"Rationale: {result['explanation']['core_rationale']}")
""",
        language="python"
    )
