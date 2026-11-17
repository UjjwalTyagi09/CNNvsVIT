"""
Report Generator.
Produces research-grade Markdown and JSON recommendation reports for export and auditing.
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from src.requirements_parser import ProblemRequirements
from src.utils import format_latency, format_params, format_percentage


class ReportGenerator:
    """Generates structured Markdown and JSON reports."""

    def generate_markdown(
        self,
        recommendation_result: Dict[str, Any]
    ) -> str:
        """Generate structured markdown recommendation report."""
        req: ProblemRequirements = recommendation_result["requirements"]
        top = recommendation_result.get("top_recommendation")
        ranked = recommendation_result.get("ranked_candidates", [])
        explanation = recommendation_result.get("explanation", {})

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = []
        md.append("# Vision Model Selection & Recommendation Report")
        md.append(f"*Generated on {now} by Evidence-Based Vision Model Recommendation Engine*\n")

        # 1. Executive Summary
        md.append("## 1. Executive Summary")
        if top:
            md.append(f"- **Recommended Model**: `{top.get('display_name')}` ({top.get('architecture_family')})")
            md.append(f"- **Final MAUT Score**: **{top.get('score')} / 100** (Margin over runner-up: `+{top.get('margin_over_runner_up', 0.0):.2f}` pts)")
            md.append(f"- **Evidence Confidence**: **{explanation.get('confidence_level')}** ({explanation.get('confidence_score')}% empirical grounding)")
            md.append(f"- **Primary Architecture Advantage**: {explanation.get('core_rationale')}\n")
        else:
            md.append("- **Status**: No candidates satisfied the active constraints.\n")

        # 2. Problem Statement & Constraints
        md.append("## 2. Problem Definition & Operational Constraints")
        md.append(f"- **User Query / Objective**: *\"{req.raw_query}\"*")
        md.append(f"- **Dataset Regime**: `{req.dataset_size.upper()}` (Regime: `{req.training_regime}`)")
        md.append(f"- **Hardware Target**: `{req.hardware_target.upper()}`")
        if req.max_latency_ms:
            md.append(f"- **Latency Ceiling**: `{req.max_latency_ms} ms`")
        if req.max_params:
            md.append(f"- **Parameter Ceiling**: `{format_params(req.max_params)}`")
        if req.noise_tolerance_critical:
            md.append(f"- **Corruption Robustness**: **Critical** (Targeting: {', '.join(req.corruption_types) if req.corruption_types else 'General corruptions'})")
        md.append("")

        # 3. Decision Weights
        md.append("### Decision Priority Weights (MAUT)")
        md.append("| Dimension | Priority Weight | Focus |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| Clean Accuracy | {req.weight_accuracy * 100:.1f}% | Top-1 validation accuracy |")
        md.append(f"| Corruption Robustness | {req.weight_robustness * 100:.1f}% | Degradation retention under noise/blur/weather |")
        md.append(f"| Latency & Throughput | {req.weight_latency * 100:.1f}% | Inference wall-clock speed |")
        md.append(f"| Data Efficiency | {req.weight_data_efficiency * 100:.1f}% | Sample requirements & scratch trainability |")
        md.append(f"| Calibration (ECE) | {req.weight_calibration * 100:.1f}% | Probability reliability & error calibration |")
        md.append("")

        # 4. Ranked Model Scorecard
        md.append("## 3. Candidate Scorecard & Ranking")
        md.append("| Rank | Model | Family | Score | Top-1 Acc | Robustness | Latency | Parameters |")
        md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for c in ranked:
            raw = c.get("raw_metrics", {})
            md.append(
                f"| #{c.get('rank')} | **{c.get('display_name')}** | `{c.get('architecture_family')}` | "
                f"**{c.get('score')}** | {raw.get('clean_top1', 0):.1f}% | {raw.get('robustness_metric', 0):.1f}% | "
                f"{raw.get('latency_ms', 0):.1f} ms | {format_params(c.get('parameter_count', 0))} |"
            )
        md.append("")

        # 5. Architecture Analysis & Trade-offs
        md.append("## 4. Architectural Analysis: CNN vs. ViT vs. MViT")
        md.append(f"{explanation.get('architecture_analysis', '')}\n")
        
        md.append("### Key Engineering Trade-Offs")
        for to in explanation.get("trade_offs", []):
            md.append(f"- {to}")
        md.append("")

        md.append("### Sensitivity Analysis")
        md.append(f"{explanation.get('sensitivity_analysis', '')}\n")

        # 6. Provenance & Citations
        md.append("## 5. Evidence Provenance & Citations")
        for cite in explanation.get("verified_citations", []):
            if cite:
                md.append(f"- Source: [{cite}]({cite})")
        if explanation.get("evidence_notes"):
            md.append("\n**Gaps & Caveats:**")
            for note in explanation.get("evidence_notes", []):
                md.append(f"- ⚠️ {note}")

        return "\n".join(md)

    def generate_json(
        self,
        recommendation_result: Dict[str, Any]
    ) -> str:
        """Generate machine-readable JSON recommendation report."""
        req: ProblemRequirements = recommendation_result["requirements"]
        payload = {
            "timestamp": datetime.now().isoformat(),
            "query": req.raw_query,
            "parsed_requirements": {
                "dataset_size": req.dataset_size,
                "training_regime": req.training_regime,
                "hardware_target": req.hardware_target,
                "max_latency_ms": req.max_latency_ms,
                "max_params": req.max_params,
                "noise_tolerance_critical": req.noise_tolerance_critical,
                "weights": {
                    "accuracy": req.weight_accuracy,
                    "robustness": req.weight_robustness,
                    "latency": req.weight_latency,
                    "data_efficiency": req.weight_data_efficiency,
                    "calibration": req.weight_calibration
                }
            },
            "top_recommendation": recommendation_result.get("top_recommendation"),
            "ranked_candidates": recommendation_result.get("ranked_candidates", []),
            "explanation": recommendation_result.get("explanation", {})
        }
        return json.dumps(payload, indent=2, default=str)
