"""
Constraint Filter.
Applies hard operational constraints (hardware, parameter limits, latency limits, scratch feasibility)
and tracks exclusion justifications.
"""

from typing import List, Dict, Any, Tuple
from src.requirements_parser import ProblemRequirements


class ConstraintFilter:
    """Applies hard constraints to candidate models."""

    def filter(
        self,
        candidates: List[Dict[str, Any]],
        requirements: ProblemRequirements
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter candidate models based on requirements.
        Returns:
            passed_candidates: Models satisfying all hard constraints.
            excluded_candidates: Models that failed at least one constraint, annotated with reasons.
        """
        passed: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []

        for cand in candidates:
            reasons = []
            ev = cand.get("evidence", {})
            eff = ev.get("efficiency", {})
            params = cand.get("parameter_count", eff.get("parameters", 0))
            
            # Determine appropriate latency metric based on hardware target
            if requirements.hardware_target in ("cpu", "edge"):
                latency = eff.get("cpu_latency_ms", 999.0)
            else:
                latency = eff.get("gpu_latency_ms", eff.get("cpu_latency_ms", 999.0))

            # 1. Parameter constraint
            if requirements.max_params is not None and params > requirements.max_params:
                reasons.append(
                    f"Parameter count ({params:,}) exceeds maximum budget ({requirements.max_params:,})"
                )

            # 2. Latency constraint
            if requirements.max_latency_ms is not None and latency > requirements.max_latency_ms:
                reasons.append(
                    f"Inference latency ({latency:.1f}ms) exceeds max latency limit ({requirements.max_latency_ms:.1f}ms)"
                )

            # 3. Hardware compatibility
            if requirements.hardware_target == "cpu" and not cand.get("cpu_friendly", True):
                reasons.append("Model is not optimized for CPU inference (high compute complexity)")

            # 4. Scratch feasibility constraint
            if requirements.training_regime == "scratch":
                pretrain_data = str(cand.get("pretrained_dataset", "")).lower()
                suitability = ev.get("dataset_suitability", {})
                if not suitability.get("scratch_feasible", True) or not pretrain_data.startswith("none"):
                    reasons.append(
                        "Model requires massive pretraining and is not evaluated/feasible for scratch training"
                    )

            if not reasons:
                passed.append(cand)
            else:
                annotated = dict(cand)
                annotated["exclusion_reasons"] = reasons
                excluded.append(annotated)

        # Graceful fallback: If hard constraints filter out 100% of candidates,
        # return all candidates with warnings so the user is never left with an empty result.
        if not passed:
            for item in excluded:
                item["fallback_warning"] = "Relaxed constraint: candidate included despite constraint violation."
            return excluded, []

        return passed, excluded
