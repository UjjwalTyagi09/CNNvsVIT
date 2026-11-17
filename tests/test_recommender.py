"""
Unit tests for the end-to-end VisionModelRecommender.
"""

import json
import pytest
from src.recommender import VisionModelRecommender


@pytest.fixture
def recommender():
    return VisionModelRecommender()


def test_end_to_end_recommendation(recommender):
    query = "Select an accurate and robust vision model for high-resolution image analysis."
    result = recommender.recommend(query)

    assert "top_recommendation" in result
    assert result["top_recommendation"] is not None
    assert "explanation" in result
    assert "headline" in result["explanation"]
    assert len(result["ranked_candidates"]) > 0


def test_compare_models(recommender):
    cmp = recommender.compare_models("resnet50_cifar10_timm", "vit_base_16_cifar10_hf")
    assert "metrics_comparison" in cmp
    metrics = cmp["metrics_comparison"]
    assert "top1_accuracy" in metrics
    assert metrics["top1_accuracy"]["diff"] < 0  # ViT has higher accuracy than ResNet-50


def test_export_markdown_and_json(recommender):
    result = recommender.recommend("Edge robot camera with real-time requirements")
    md = recommender.export_report(result, fmt="markdown")
    assert "# Vision Model Selection & Recommendation Report" in md
    assert "Executive Summary" in md

    js = recommender.export_report(result, fmt="json")
    parsed_json = json.loads(js)
    assert "top_recommendation" in parsed_json
    assert "parsed_requirements" in parsed_json
