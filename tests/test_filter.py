"""
Unit tests for ConstraintFilter.
"""

import pytest
from src.registry import EvidenceRegistry
from src.requirements_parser import ProblemRequirements
from src.constraint_filter import ConstraintFilter


@pytest.fixture
def filter_env():
    reg = EvidenceRegistry()
    candidates = reg.get_all_unified_candidates()
    filt = ConstraintFilter()
    return filt, candidates


def test_parameter_filtering(filter_env):
    filt, candidates = filter_env
    req = ProblemRequirements(max_params=2_000_000)
    passed, excluded = filt.filter(candidates, req)

    # Only scratch CNN and scratch ViT are < 2M parameters (~960k)
    passed_ids = [c["model_id"] for c in passed]
    assert "scratch_cnn_cifar100" in passed_ids
    assert "scratch_vit_cifar100" in passed_ids
    assert "vit_base_16_cifar10_hf" not in passed_ids
    assert len(excluded) > 0


def test_latency_filtering(filter_env):
    filt, candidates = filter_env
    # CPU latency <= 10ms
    req = ProblemRequirements(hardware_target="cpu", max_latency_ms=10.0)
    passed, excluded = filt.filter(candidates, req)

    for c in passed:
        lat = c["evidence"]["efficiency"]["cpu_latency_ms"]
        assert lat <= 10.0


def test_scratch_feasibility_filter(filter_env):
    filt, candidates = filter_env
    req = ProblemRequirements(training_regime="scratch")
    passed, excluded = filt.filter(candidates, req)

    # Large pretrained models (like ViT-Base and MViTv2) are marked scratch_feasible=False
    passed_ids = [c["model_id"] for c in passed]
    assert "scratch_cnn_cifar100" in passed_ids
    assert "scratch_vit_cifar100" in passed_ids
    assert "vit_base_16_cifar10_hf" not in passed_ids


def test_graceful_fallback_when_all_violated(filter_env):
    filt, candidates = filter_env
    # Impossibly tight constraint: < 100k params and < 0.1ms latency
    req = ProblemRequirements(max_params=100_000, max_latency_ms=0.1)
    passed, excluded = filt.filter(candidates, req)

    # Fallback guarantees non-empty list with fallback warning
    assert len(passed) > 0
    assert "fallback_warning" in passed[0]
