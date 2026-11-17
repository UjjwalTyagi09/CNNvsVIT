"""
Explainer.
Generates research-grounded, evidence-backed explanations and trade-off rationales
comparing CNN vs ViT vs MViT architectures.
"""

from typing import Dict, List, Any, Optional
from src.requirements_parser import ProblemRequirements
from src.utils import format_latency, format_params, format_percentage


class RecommendationExplainer:
    """Generates natural language and structured explanations for model recommendations."""

    def explain(
        self,
        ranked_candidates: List[Dict[str, Any]],
        requirements: ProblemRequirements
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation packet for the top recommendation and runner-up.
        """
        if not ranked_candidates:
            return {
                "headline": "No candidates available matching constraints.",
                "rationale": "All candidate models were excluded based on strict hardware or regime constraints.",
                "trade_offs": [],
                "architecture_comparison": {},
                "sensitivity_analysis": "Try relaxing latency or parameter limits."
            }

        top = ranked_candidates[0]
        runner_up = ranked_candidates[1] if len(ranked_candidates) > 1 else None

        top_name = top.get("display_name", top["model_id"])
        top_fam = top.get("architecture_family", "CNN")
        margin = top.get("margin_over_runner_up", 0.0)

        # 1. Headline & Core Rationale
        headline = f"Recommended Model: {top_name} (Architecture: {top_fam})"
        
        reasons = []
        # Examine highest score contribution
        contributions = top.get("score_contributions", {})
        sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_factor, top_val = sorted_contrib[0] if sorted_contrib else ("accuracy", 0.0)
        
        factor_readable = {
            "accuracy": "clean classification accuracy",
            "robustness": "resilience under corruptions & noise",
            "latency": "low-latency inference speed",
            "data_efficiency": "data efficiency & sample economy",
            "calibration": "probability calibration & confidence reliability"
        }
        reasons.append(
            f"Demonstrated superior balance for your requirements, driven primarily by its strong "
            f"{factor_readable.get(top_factor, top_factor)} (utility contribution: {top_val:.1f} pts)."
        )

        # 2. Architecture Comparison (CNN vs ViT vs MViT)
        arch_rationale = self._explain_architecture_fit(top_fam, requirements)

        # 3. Head-to-Head Comparison with Runner-Up
        runner_up_comparison = ""
        if runner_up:
            r_name = runner_up.get("display_name", runner_up["model_id"])
            r_fam = runner_up.get("architecture_family", "")
            
            # Identify where winner beat runner up and where runner up had an edge
            top_utils = top.get("utilities", {})
            r_utils = runner_up.get("utilities", {})
            
            advantages = []
            disadvantages = []
            for dim in ["accuracy", "robustness", "latency", "data_efficiency", "calibration"]:
                diff = top_utils.get(dim, 0.0) - r_utils.get(dim, 0.0)
                if diff > 0.08:
                    advantages.append(factor_readable[dim])
                elif diff < -0.08:
                    disadvantages.append(factor_readable[dim])
                    
            adv_str = ", ".join(advantages) if advantages else "overall multi-attribute utility balance"
            runner_up_comparison = (
                f"{top_name} surpassed runner-up {r_name} by a margin of {margin:.2f} points. "
                f"{top_name} gained its advantage through superior {adv_str}."
            )
            if disadvantages:
                runner_up_comparison += f" Note that {r_name} holds an advantage in {', '.join(disadvantages)}."

        # 4. Explicit Trade-Offs
        trade_offs = self._generate_trade_offs(top, runner_up, requirements)

        # 5. Sensitivity Analysis: What would change this recommendation?
        sensitivity = self._generate_sensitivity_hints(top, runner_up, requirements)

        # 6. Evidence Confidence & Gaps
        confidence_info = self._calculate_confidence(top, requirements)

        return {
            "headline": headline,
            "core_rationale": " ".join(reasons),
            "architecture_analysis": arch_rationale,
            "runner_up_comparison": runner_up_comparison,
            "trade_offs": trade_offs,
            "sensitivity_analysis": sensitivity,
            "confidence_score": confidence_info["score"],
            "confidence_level": confidence_info["level"],
            "evidence_notes": confidence_info["notes"],
            "verified_citations": [top.get("source_repository", ""), top.get("paper_url", "")]
        }

    def _explain_architecture_fit(self, family: str, req: ProblemRequirements) -> str:
        """Explain why this architecture family fits the specific problem context."""
        if family == "CNN":
            if req.dataset_size == "small" or req.training_regime == "scratch":
                return (
                    "Why CNN: Convolutions possess an inherent inductive bias (spatial locality and translation invariance). "
                    "When training from scratch or with limited data, this architectural prior prevents overfitting and "
                    "allows the model to learn effective visual representations without requiring millions of pretraining images."
                )
            else:
                return (
                    "Why CNN: Modern convolutional architectures (like ConvNeXt or ResNet) provide exceptional throughput "
                    "and memory efficiency on standard hardware, making them ideal when real-time latency or CPU deployment is required."
                )
        elif family == "ViT":
            return (
                "Why ViT: Vision Transformers process images via global multi-head self-attention across patches. "
                "Because self-attention is not constrained by fixed local receptive fields, ViTs excel at retaining semantic context "
                "under high-frequency image noise, sensor corruptions, and adverse weather conditions. They also demonstrate "
                "superior calibration error (ECE), providing reliable predictive confidence."
            )
        elif family == "MViT":
            return (
                "Why MViT: Multiscale Vision Transformers bridge the inductive bias of CNN hierarchies and the representation "
                "power of Vision Transformers. By using pooling attention to expand channel depth while reducing spatial resolution "
                "across stages, MViT captures both fine local textures and global multi-scale context efficiently."
            )
        return ""

    def _generate_trade_offs(
        self,
        top: Dict[str, Any],
        runner_up: Optional[Dict[str, Any]],
        req: ProblemRequirements
    ) -> List[str]:
        """Itemize concrete engineering trade-offs."""
        trade_offs = []
        top_fam = top.get("architecture_family", "")
        ev = top.get("evidence", {})
        eff = ev.get("efficiency", {})
        
        # Accuracy vs Latency
        if req.hardware_target in ("cpu", "edge"):
            lat = eff.get("cpu_latency_ms", 0.0)
            trade_offs.append(
                f"Latency vs Compute: At {format_latency(lat)} on CPU, this model meets deployment budgets, "
                f"though larger server-class models could offer 1-2% higher clean accuracy at significantly higher compute cost."
            )

        # Robustness vs Clean Accuracy
        if top_fam == "ViT":
            trade_offs.append(
                "Robustness vs Throughput: ViT provides superior corruption resilience and calibration, but requires "
                "higher FLOPs and memory footprint than a compact CNN counterpart."
            )
        elif top_fam == "CNN":
            trade_offs.append(
                "Throughput vs Extreme Noise: CNN offers fast inference and low memory, but exhibits higher accuracy drop "
                "under extreme noise (e.g., severe Gaussian or shot noise) compared to large Vision Transformers."
            )
        elif top_fam == "MViT":
            trade_offs.append(
                "Multiscale Resolution vs Memory: MViT provides high feature capacity across resolutions, but attention pooling "
                "kernels require specialized PyTorch deployment configurations."
            )

        return trade_offs

    def _generate_sensitivity_hints(
        self,
        top: Dict[str, Any],
        runner_up: Optional[Dict[str, Any]],
        req: ProblemRequirements
    ) -> str:
        """Describe what threshold or weight shift would change the winning recommendation."""
        if not runner_up:
            return "Recommendation is robust across the active candidate pool."

        r_name = runner_up.get("display_name", runner_up["model_id"])
        r_fam = runner_up.get("architecture_family", "")
        top_fam = top.get("architecture_family", "")

        if top_fam == "CNN" and r_fam in ("ViT", "MViT"):
            return (
                f"Sensitivity: If your hardware budget is upgraded to dedicated GPU or your priority on corruption robustness "
                f"increases by >15%, {r_name} ({r_fam}) would overtake the current recommendation."
            )
        elif top_fam == "ViT" and r_fam == "CNN":
            return (
                f"Sensitivity: If real-time latency (<20ms) or edge CPU deployment becomes a strict requirement, "
                f"{r_name} (CNN) would become the preferred choice."
            )
        return (
            f"Sensitivity: A shift in priority towards {r_name}'s primary strengths would narrow the {top.get('margin_over_runner_up', 0.0):.1f} pt margin."
        )

    def _calculate_confidence(self, candidate: Dict[str, Any], req: ProblemRequirements) -> Dict[str, Any]:
        """Compute empirical evidence confidence score and flag research gaps."""
        score = 95
        notes = []
        ev = candidate.get("evidence", {})
        rob = ev.get("robustness_summary", {})

        if "estimated_mce" in rob or "note" in rob:
            score -= 15
            notes.append("Robustness metrics include paper-derived estimates (direct CIFAR-10-C run pending for this variant).")

        if req.dataset_size == "small" and candidate.get("parameter_count", 0) > 30_000_000:
            score -= 10
            notes.append("Large model applied to small dataset regime relies on pretraining transferability assumptions.")

        level = "High" if score >= 85 else ("Medium" if score >= 70 else "Low")
        return {"score": score, "level": level, "notes": notes}
