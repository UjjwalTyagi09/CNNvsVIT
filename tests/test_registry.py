"""
Unit tests for the EvidenceRegistry.
"""

import pytest
from src.registry import EvidenceRegistry


@pytest.fixture
def registry():
    return EvidenceRegistry()


def test_registry_loads_models(registry):
    models = registry.get_all_models()
    assert len(models) >= 8
    model_ids = [m["model_id"] for m in models]
    assert "resnet50_cifar10_timm" in model_ids
    assert "vit_base_16_cifar10_hf" in model_ids
    assert "convnext_tiny_cifar10_timm" in model_ids
    assert "scratch_cnn_cifar100" in model_ids
    assert "scratch_vit_cifar100" in model_ids
    assert "mvitv2_tiny_in1k" in model_ids


def test_registry_evidence_and_sources(registry):
    sources = registry.get_all_sources()
    assert len(sources) >= 4
    protocols = registry.get_all_protocols()
    assert len(protocols) >= 4

    ev = registry.get_evidence("vit_base_16_cifar10_hf")
    assert ev is not None
    assert ev["clean_metrics"]["top1_accuracy"] > 95.0
    assert "per_corruption_sev3" in ev


def test_registry_family_filtering(registry):
    cnns = registry.get_models_by_family("CNN")
    vits = registry.get_models_by_family("ViT")
    mvits = registry.get_models_by_family("MViT")

    assert len(cnns) >= 3
    assert len(vits) >= 2
    assert len(mvits) >= 3


def test_unified_candidate(registry):
    cand = registry.get_unified_candidate("scratch_cnn_cifar100")
    assert cand is not None
    assert cand["display_name"] == "Scratch Residual CNN (CIFAR-100)"
    assert "evidence" in cand
    assert cand["evidence"]["efficiency"]["parameters"] == 959452
    assert "protocol" in cand
