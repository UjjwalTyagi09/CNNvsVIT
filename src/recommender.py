"""
Vision Model Recommender.
Master orchestration pipeline uniting registry, parser, filter, scorer, and explainer.
"""

from typing import Dict, Any, List, Optional
from src.registry import EvidenceRegistry
from src.requirements_parser import RequirementsParser, ProblemRequirements
from src.constraint_filter import ConstraintFilter
from src.scorer import CandidateScorer
from src.explainer import RecommendationExplainer
from src.report_generator import ReportGenerator


class VisionModelRecommender:
    """Master recommendation engine for computer vision model selection."""

    def __init__(self, data_dir: Optional[str] = None):
        self.registry = EvidenceRegistry(data_dir=data_dir)
        self.parser = RequirementsParser()
        self.filter = ConstraintFilter()
        self.scorer = CandidateScorer()
        self.explainer = RecommendationExplainer()
        self.report_gen = ReportGenerator()

    def recommend(
        self,
        query: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute full recommendation pipeline.
        
        Args:
            query: Natural language problem statement or description.
            overrides: Optional structured overrides for constraints or weights.
            
        Returns:
            Dictionary containing requirements, top recommendation, runner-up,
            all ranked candidates, excluded models, and evidence-grounded explanation.
        """
        # 1. Parse requirements
        req: ProblemRequirements = self.parser.parse(query, overrides=overrides)

        # 2. Retrieve unified candidates from registry
        all_candidates = self.registry.get_all_unified_candidates()

        # 3. Apply hard constraint filters
        passed, excluded = self.filter.filter(all_candidates, req)

        # 4. Score and rank passed candidates
        ranked = self.scorer.score(passed, req)

        # 5. Generate explanations and trade-offs
        explanation = self.explainer.explain(ranked, req)

        top = ranked[0] if ranked else None
        runner_up = ranked[1] if len(ranked) > 1 else None

        result = {
            "requirements": req,
            "top_recommendation": top,
            "runner_up": runner_up,
            "ranked_candidates": ranked,
            "excluded_candidates": excluded,
            "explanation": explanation,
            "total_candidates_evaluated": len(all_candidates),
            "candidates_passed": len(passed)
        }
        return result

    def compare_models(self, model_a_id: str, model_b_id: str) -> Dict[str, Any]:
        """Perform head-to-head empirical comparison between two specific models."""
        a = self.registry.get_unified_candidate(model_a_id)
        b = self.registry.get_unified_candidate(model_b_id)
        if not a or not b:
            return {"error": "One or both model IDs not found in registry."}

        ev_a = a.get("evidence", {})
        ev_b = b.get("evidence", {})

        clean_a = ev_a.get("clean_metrics", {})
        clean_b = ev_b.get("clean_metrics", {})

        eff_a = ev_a.get("efficiency", {})
        eff_b = ev_b.get("efficiency", {})

        rob_a = ev_a.get("robustness_summary", {})
        rob_b = ev_b.get("robustness_summary", {})

        return {
            "model_a": a,
            "model_b": b,
            "metrics_comparison": {
                "top1_accuracy": {
                    "model_a": clean_a.get("top1_accuracy"),
                    "model_b": clean_b.get("top1_accuracy"),
                    "diff": round(float(clean_a.get("top1_accuracy", 0)) - float(clean_b.get("top1_accuracy", 0)), 2)
                },
                "parameters": {
                    "model_a": a.get("parameter_count"),
                    "model_b": b.get("parameter_count"),
                    "ratio": round(a.get("parameter_count", 1) / max(1, b.get("parameter_count", 1)), 2)
                },
                "cpu_latency_ms": {
                    "model_a": eff_a.get("cpu_latency_ms"),
                    "model_b": eff_b.get("cpu_latency_ms")
                },
                "gpu_latency_ms": {
                    "model_a": eff_a.get("gpu_latency_ms"),
                    "model_b": eff_b.get("gpu_latency_ms")
                },
                "mean_corrupted_top1": {
                    "model_a": rob_a.get("mean_corrupted_top1"),
                    "model_b": rob_b.get("mean_corrupted_top1")
                }
            }
        }

    def export_report(self, recommendation_result: Dict[str, Any], fmt: str = "markdown") -> str:
        """Export recommendation as markdown or JSON."""
        if fmt.lower() == "json":
            return self.report_gen.generate_json(recommendation_result)
        return self.report_gen.generate_markdown(recommendation_result)


if __name__ == "__main__":
    import sys
    default_q = "Deploying camera surveillance in adverse weather with severe fog, snow, and sensor noise. Accuracy and robustness are essential."
    user_q = sys.argv[1] if len(sys.argv) > 1 else default_q
    rec = VisionModelRecommender()
    output = rec.recommend(user_q)
    print(rec.export_report(output, fmt="markdown"))
