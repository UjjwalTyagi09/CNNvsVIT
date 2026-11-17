"""
Scorer.
Calculates multi-attribute utility scores (MAUT) across accuracy, robustness,
latency, data efficiency, and calibration.
"""

from typing import List, Dict, Any, Tuple
from src.requirements_parser import ProblemRequirements
from src.utils import normalize_series, compute_relative_retention


class CandidateScorer:
    """Scores candidate models using multi-attribute utility theory."""

    def score(
        self,
        candidates: List[Dict[str, Any]],
        requirements: ProblemRequirements
    ) -> List[Dict[str, Any]]:
        """
        Score and rank filtered candidates.
        Returns candidates enriched with raw metrics, normalized dimension utilities,
        total utility score, and attribute breakdown.
        """
        if not candidates:
            return []

        # 1. Extract raw attribute values
        raw_accuracy: Dict[str, float] = {}
        raw_robustness: Dict[str, float] = {}
        raw_latency: Dict[str, float] = {}
        raw_data_efficiency: Dict[str, float] = {}
        raw_calibration: Dict[str, float] = {}

        for cand in candidates:
            cid = cand["model_id"]
            ev = cand.get("evidence", {})
            clean = ev.get("clean_metrics", {})
            rob = ev.get("robustness_summary", {})
            eff = ev.get("efficiency", {})
            suit = ev.get("dataset_suitability", {})
            per_corr = ev.get("per_corruption_sev3", {})

            # A. Clean Accuracy
            raw_accuracy[cid] = float(clean.get("top1_accuracy", 50.0))

            # B. Robustness
            # If specific corruptions requested and present in evidence, compute specific mean
            if requirements.corruption_types and any(c in per_corr for c in requirements.corruption_types):
                corr_accs = [per_corr[c] for c in requirements.corruption_types if c in per_corr]
                score_rob = sum(corr_accs) / len(corr_accs)
            elif "mean_corrupted_top1" in rob:
                score_rob = float(rob["mean_corrupted_top1"])
            elif "gaussian_noise_accuracy" in rob:
                score_rob = float(rob["gaussian_noise_accuracy"])
            elif "estimated_mce" in rob:
                # Lower mCE is better robustness; invert to positive score scale (100 - mCE)
                score_rob = 100.0 - float(rob["estimated_mce"])
            else:
                score_rob = float(clean.get("top1_accuracy", 50.0)) * 0.75
            raw_robustness[cid] = score_rob

            # C. Latency (invert: lower ms is better)
            if requirements.hardware_target in ("cpu", "edge"):
                lat = float(eff.get("cpu_latency_ms", 50.0))
            else:
                lat = float(eff.get("gpu_latency_ms", eff.get("cpu_latency_ms", 50.0)))
            raw_latency[cid] = lat

            # D. Data Efficiency
            # Inductive bias matters on small datasets; scale capacity matters on large datasets
            min_samples = suit.get("min_recommended_samples", 10000)
            transfer_eff = suit.get("transfer_efficiency", 0.8)
            family = cand.get("architecture_family", "CNN")

            if requirements.dataset_size == "small":
                if requirements.training_regime == "scratch":
                    # CNNs with local inductive bias win on small scratch
                    eff_score = 90.0 if family == "CNN" else 60.0
                else:
                    eff_score = transfer_eff * 100.0 - (min_samples / 1000.0)
            elif requirements.dataset_size == "large":
                # ViT and MViT scale better with large data
                eff_score = 95.0 if family in ("ViT", "MViT") else 80.0
            else:  # medium
                eff_score = transfer_eff * 90.0
            raw_data_efficiency[cid] = max(10.0, min(100.0, eff_score))

            # E. Calibration (ECE lower is better; confidence stability)
            ece = clean.get("expected_calibration_error")
            if ece is not None:
                cal_score = max(0.0, 1.0 - (float(ece) * 5.0)) * 100.0
            elif "mean_softmax_confidence" in clean:
                conf = float(clean.get("mean_softmax_confidence", 0.9))
                cal_score = conf * 90.0
            else:
                cal_score = 75.0
            raw_calibration[cid] = cal_score

        # 2. Normalize attributes into [0.0, 1.0]
        norm_acc = normalize_series(raw_accuracy, higher_is_better=True)
        norm_rob = normalize_series(raw_robustness, higher_is_better=True)
        norm_lat = normalize_series(raw_latency, higher_is_better=False)  # lower ms -> higher utility
        norm_data_eff = normalize_series(raw_data_efficiency, higher_is_better=True)
        norm_cal = normalize_series(raw_calibration, higher_is_better=True)

        # 3. Calculate weighted MAUT score
        scored_candidates = []
        for cand in candidates:
            cid = cand["model_id"]
            u_acc = norm_acc.get(cid, 0.5)
            u_rob = norm_rob.get(cid, 0.5)
            u_lat = norm_lat.get(cid, 0.5)
            u_data = norm_data_eff.get(cid, 0.5)
            u_cal = norm_cal.get(cid, 0.5)

            total_score = (
                requirements.weight_accuracy * u_acc
                + requirements.weight_robustness * u_rob
                + requirements.weight_latency * u_lat
                + requirements.weight_data_efficiency * u_data
                + requirements.weight_calibration * u_cal
            ) * 100.0

            result = dict(cand)
            result["score"] = round(total_score, 2)
            result["utilities"] = {
                "accuracy": round(u_acc, 3),
                "robustness": round(u_rob, 3),
                "latency": round(u_lat, 3),
                "data_efficiency": round(u_data, 3),
                "calibration": round(u_cal, 3)
            }
            result["raw_metrics"] = {
                "clean_top1": raw_accuracy[cid],
                "robustness_metric": raw_robustness[cid],
                "latency_ms": raw_latency[cid],
                "data_efficiency_index": raw_data_efficiency[cid],
                "calibration_index": raw_calibration[cid]
            }
            result["score_contributions"] = {
                "accuracy": round(requirements.weight_accuracy * u_acc * 100.0, 2),
                "robustness": round(requirements.weight_robustness * u_rob * 100.0, 2),
                "latency": round(requirements.weight_latency * u_lat * 100.0, 2),
                "data_efficiency": round(requirements.weight_data_efficiency * u_data * 100.0, 2),
                "calibration": round(requirements.weight_calibration * u_cal * 100.0, 2)
            }
            scored_candidates.append(result)

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Assign rank and calculate margin relative to runner up
        for i, c in enumerate(scored_candidates):
            c["rank"] = i + 1
            if i == 0 and len(scored_candidates) > 1:
                c["margin_over_runner_up"] = round(scored_candidates[0]["score"] - scored_candidates[1]["score"], 2)
            elif i > 0:
                c["margin_from_winner"] = round(c["score"] - scored_candidates[0]["score"], 2)

        return scored_candidates
