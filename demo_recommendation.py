"""
Demonstration script for Evidence-Based Vision Model Recommendation Engine.
Executes an end-to-end recommendation workflow from a realistic problem statement,
displaying ranked candidate models, MAUT scores, evidence citations, and trade-off rationales.
"""

import sys
from pathlib import Path
from src.recommender import VisionModelRecommender
from src.utils import format_params, format_latency


def run_demo():
    print("=" * 80)
    print("       EVIDENCE-BASED VISION MODEL RECOMMENDATION ENGINE — DEMO")
    print("=" * 80)

    recommender = VisionModelRecommender()

    # Problem statement
    problem_statement = (
        "We are deploying an outdoor surveillance camera system that encounters severe fog, "
        "winter snow, and low-light sensor noise. High classification accuracy and corruption "
        "robustness are essential."
    )

    print(f"\n[INPUT PROBLEM STATEMENT]:\n\"{problem_statement}\"\n")

    # Run recommendation
    result = recommender.recommend(problem_statement)
    req = result["requirements"]
    winner = result["top_recommendation"]
    runner_up = result["runner_up"]
    ranked = result["ranked_candidates"]
    excluded = result["excluded_candidates"]
    explanation = result["explanation"]

    print("-" * 80)
    print("1. PARSED OPERATIONAL REQUIREMENTS & DECISION WEIGHTS")
    print("-" * 80)
    print(f" - Detected Dataset Regime : {req.dataset_size.upper()} (Regime: {req.training_regime})")
    print(f" - Hardware Deployment     : {req.hardware_target.upper()}")
    print(f" - Noise Tolerance Needed  : {'CRITICAL' if req.noise_tolerance_critical else 'Standard'}")
    print(f" - Targeted Corruptions    : {', '.join(req.corruption_types)}")
    print(f" - Decision Weights (MAUT) : Accuracy={req.weight_accuracy*100:.1f}%, Robustness={req.weight_robustness*100:.1f}%, "
          f"Latency={req.weight_latency*100:.1f}%, DataEff={req.weight_data_efficiency*100:.1f}%, Calibration={req.weight_calibration*100:.1f}%")

    print("\n" + "-" * 80)
    print("2. CANDIDATE EVALUATION LEADERBOARD (RANKED BY MAUT SCORE)")
    print("-" * 80)
    print(f"{'Rank':<6} {'Model':<30} {'Family':<8} {'Score':<8} {'Clean Acc':<10} {'Robustness':<11} {'Latency':<10} {'Params'}")
    print("-" * 80)
    for c in ranked:
        raw = c["raw_metrics"]
        print(f"#{c['rank']:<5} {c['display_name']:<30} {c['architecture_family']:<8} "
              f"{c['score']:<8.2f} {raw['clean_top1']:<9.1f}% {raw['robustness_metric']:<10.1f}% "
              f"{raw['latency_ms']:<9.1f}ms {format_params(c.get('parameter_count', 0))}")

    if excluded:
        print(f"\n[Excluded Candidates ({len(excluded)} models)]:")
        for ex in excluded:
            print(f" - {ex['display_name']}: {'; '.join(ex.get('exclusion_reasons', []))}")

    print("\n" + "-" * 80)
    print("3. RECOMMENDATION RATIONALE & TRADE-OFF ANALYSIS")
    print("-" * 80)
    print(f" WINNER: {winner['display_name']} ({winner['architecture_family']})")
    print(f" MAUT Score: {winner['score']:.2f}/100 (Margin over runner-up: +{winner.get('margin_over_runner_up', 0.0):.2f} pts)")
    print(f" Evidence Confidence: {explanation.get('confidence_level')} ({explanation.get('confidence_score')}%)")
    print(f"\n [Core Driver]: {explanation.get('core_rationale')}")
    print(f"\n [Architectural Analysis]:\n {explanation.get('architecture_analysis')}")
    print(f"\n [Comparison with Runner-Up]:\n {explanation.get('runner_up_comparison')}")

    print("\n [Engineering Trade-Offs]:")
    for to in explanation.get("trade_offs", []):
        print(f"  • {to}")

    print(f"\n [Sensitivity Insight]:\n  {explanation.get('sensitivity_analysis')}")

    if explanation.get("evidence_notes"):
        print("\n [Research Limitations & Caveats]:")
        for note in explanation.get("evidence_notes", []):
            print(f"  ⚠️ {note}")

    print("\n" + "=" * 80)
    print("                      DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
