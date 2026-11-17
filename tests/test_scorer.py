"""
Unit tests for CandidateScorer.
"""

import pytest
from src.registry import EvidenceRegistry
from src.requirements_parser import ProblemRequirements
from src.scorer import CandidateScorer


@pytest.fixture
def scorer_env():
    reg = EvidenceRegistry()
    candidates = reg.get_all_unified_candidates()
    scorer = CandidateScorer()
    return scorer, candidates


def test_scorer_outputs_valid_range(scorer_env):
    scorer, candidates = scorer_env
    req = ProblemRequirements()
    scored = scorer.score(candidates, req)

    assert len(scored) == len(candidates)
    for c in scored:
        assert 0.0 <= c["score"] <= 100.0
        assert "utilities" in c
        assert "score_contributions" in c
        for u in c["utilities"].values():
            assert 0.0 <= u <= 1.0


def test_scorer_sorts_descending(scorer_env):
    scorer, candidates = scorer_env
    req = ProblemRequirements()
    scored = scorer.score(candidates, req)

    scores = [c["score"] for c in scored]
    assert scores == sorted(scores, reverse=True)
    assert scored[0]["rank"] == 1
    if len(scored) > 1:
        assert scored[0]["margin_over_runner_up"] >= 0.0


def test_scorer_respects_extreme_latency_weight(scorer_env):
    scorer, candidates = scorer_env
    # Latency weight 90%
    req = ProblemRequirements(
        weight_accuracy=0.02,
        weight_robustness=0.02,
        weight_latency=0.90,
        weight_data_efficiency=0.03,
        weight_calibration=0.03
    )
    req.normalize_weights()
    scored = scorer.score(candidates, req)

    # Fastest model on latency must rank top
    top_model = scored[0]
    raw_lat = top_model["raw_metrics"]["latency_ms"]
    assert raw_lat <= 5.0  # Ultra-fast model
