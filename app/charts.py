"""
Plotly visualization helpers for the Vision Model Recommendation UI.
Uses a curated, cohesive dark aesthetic matching the application design system.
"""

from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px


COLOR_PALETTE = {
    "CNN": "#10b981",       # Emerald
    "ViT": "#6366f1",       # Indigo / Purple
    "MViT": "#f43f5e",      # Rose
    "accent_cyan": "#06b6d4",
    "accent_amber": "#f59e0b",
    "bg_dark": "#0e131f",
    "grid_color": "rgba(255, 255, 255, 0.08)"
}


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply unified dark styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14, 19, 31, 0.6)",
        font=dict(family="Outfit, sans-serif", color="#f8fafc", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor="rgba(18, 24, 38, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(size=11)
        )
    )
    fig.update_xaxes(
        gridcolor=COLOR_PALETTE["grid_color"],
        zerolinecolor=COLOR_PALETTE["grid_color"],
        tickfont=dict(color="#94a3b8")
    )
    fig.update_yaxes(
        gridcolor=COLOR_PALETTE["grid_color"],
        zerolinecolor=COLOR_PALETTE["grid_color"],
        tickfont=dict(color="#94a3b8")
    )
    return fig


def create_radar_chart(candidates: List[Dict[str, Any]]) -> go.Figure:
    """Create multi-candidate radar chart comparing normalized utilities."""
    categories = ["Accuracy", "Robustness", "Latency Speed", "Data Efficiency", "Calibration"]
    key_mapping = {
        "Accuracy": "accuracy",
        "Robustness": "robustness",
        "Latency Speed": "latency",
        "Data Efficiency": "data_efficiency",
        "Calibration": "calibration"
    }

    fig = go.Figure()

    # Show at most top 4 candidates to keep chart clean
    top_cands = candidates[:4]
    for c in top_cands:
        utils = c.get("utilities", {})
        fam = c.get("architecture_family", "CNN")
        color = COLOR_PALETTE.get(fam, "#6366f1")
        name = c.get("display_name", c["model_id"])

        r_values = [utils.get(key_mapping[cat], 0.5) * 100 for cat in categories]
        # Close the polygon
        r_values.append(r_values[0])
        theta_cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_cats,
            name=name,
            fill='toself',
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
            line=dict(color=color, width=2.5),
            hoverinfo='r+name'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=COLOR_PALETTE["grid_color"],
                tickfont=dict(color="#64748b", size=10)
            ),
            angularaxis=dict(
                gridcolor=COLOR_PALETTE["grid_color"],
                tickfont=dict(color="#f8fafc", size=11, family="Outfit, sans-serif")
            ),
            bgcolor="rgba(14, 19, 31, 0.4)"
        ),
        showlegend=True,
        title=dict(text="<b>Multi-Attribute Utility Radar</b>", font=dict(size=15, color="#f8fafc"))
    )
    return apply_dark_theme(fig)


def create_score_contribution_bar(candidate: Dict[str, Any]) -> go.Figure:
    """Create stacked or grouped bar chart showing score contribution per dimension."""
    contribs = candidate.get("score_contributions", {})
    labels = ["Accuracy", "Robustness", "Latency", "Data Eff.", "Calibration"]
    keys = ["accuracy", "robustness", "latency", "data_efficiency", "calibration"]
    values = [contribs.get(k, 0.0) for k in keys]
    colors = ["#38bdf8", "#818cf8", "#34d399", "#f59e0b", "#c084fc"]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            marker_line=dict(width=1, color="rgba(255, 255, 255, 0.2)"),
            text=[f"{v:.1f} pts" for v in values],
            textposition='auto',
            hoverinfo='x+y'
        )
    ])
    fig.update_layout(
        title=dict(text=f"<b>Score Contributions for {candidate.get('display_name')}</b>", font=dict(size=14)),
        yaxis=dict(title="Points Contributed (Max 100)", range=[0, max(values) * 1.3 if values else 50]),
        height=300
    )
    return apply_dark_theme(fig)


def create_severity_curve_chart(evidence_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plot CIFAR-10-C severity degradation curves (Severities 1-5)."""
    fig = go.Figure()
    models_to_plot = ["vit_base_16_cifar10_hf", "convnext_tiny_cifar10_timm", "resnet50_cifar10_timm"]

    for mid in models_to_plot:
        data = evidence_dict.get(mid)
        if not data:
            continue
        rob = data.get("robustness_summary", {})
        by_sev = rob.get("accuracy_by_severity", {})
        clean = data.get("clean_metrics", {}).get("top1_accuracy", 100.0)

        fam = data.get("architecture_family", "CNN")
        color = COLOR_PALETTE.get(fam, "#6366f1")
        name = data.get("display_name", mid)

        x_vals = [0, 1, 2, 3, 4, 5]
        y_vals = [clean] + [by_sev.get(str(i), 0) for i in range(1, 6)]

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate="Severity %{x}: %{y:.1f}%<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text="<b>CIFAR-10-C Accuracy Degradation across Severity Levels (0 to 5)</b>", font=dict(size=15)),
        xaxis=dict(title="Corruption Severity (0 = Clean Test Set)", tickvals=[0, 1, 2, 3, 4, 5]),
        yaxis=dict(title="Top-1 Accuracy (%)", range=[50, 100]),
        height=380
    )
    return apply_dark_theme(fig)


def create_category_bar_chart(evidence_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing accuracy across 4 corruption categories (Noise, Blur, Weather, Digital)."""
    categories = ["Noise", "Blur", "Weather", "Digital"]
    cat_keys = ["noise", "blur", "weather", "digital"]
    models = ["resnet50_cifar10_timm", "convnext_tiny_cifar10_timm", "vit_base_16_cifar10_hf"]

    fig = go.Figure()

    for mid in models:
        data = evidence_dict.get(mid)
        if not data:
            continue
        rob = data.get("robustness_summary", {})
        by_cat = rob.get("accuracy_by_category", {})
        fam = data.get("architecture_family", "CNN")
        color = COLOR_PALETTE.get(fam, "#6366f1")
        name = data.get("display_name", mid)

        vals = [by_cat.get(k, 0) for k in cat_keys]

        fig.add_trace(go.Bar(
            name=name,
            x=categories,
            y=vals,
            marker_color=color,
            marker_line=dict(width=1, color="rgba(255,255,255,0.15)"),
            text=[f"{v:.1f}%" for v in vals],
            textposition='outside'
        ))

    fig.update_layout(
        barmode='group',
        title=dict(text="<b>Resilience by Corruption Taxonomy (CIFAR-10-C)</b>", font=dict(size=15)),
        yaxis=dict(title="Mean Corrupted Top-1 (%)", range=[50, 100]),
        height=380
    )
    return apply_dark_theme(fig)


def create_per_corruption_bar_chart(evidence_dict: Dict[str, Dict[str, Any]], model_id: str) -> go.Figure:
    """Bar chart of all 19 corruptions at severity 3 for a specific model."""
    data = evidence_dict.get(model_id, {})
    per_corr = data.get("per_corruption_sev3", {})
    if not per_corr:
        return go.Figure()

    sorted_items = sorted(per_corr.items(), key=lambda x: x[1])
    labels = [k.replace("_", " ").title() for k, v in sorted_items]
    vals = [v for k, v in sorted_items]

    # Color code by value
    colors = [
        "#f43f5e" if v < 70 else ("#f59e0b" if v < 80 else "#10b981")
        for v in vals
    ]

    fig = go.Figure(data=[
        go.Bar(
            y=labels,
            x=vals,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.1f}%" for v in vals],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title=dict(text=f"<b>19 Corruptions at Severity 3: {data.get('display_name', model_id)}</b>", font=dict(size=14)),
        xaxis=dict(title="Top-1 Accuracy (%)", range=[0, 105]),
        height=520,
        margin=dict(l=140, r=40, t=50, b=40)
    )
    return apply_dark_theme(fig)
