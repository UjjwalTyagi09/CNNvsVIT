"""
Integration tests covering realistic benchmark scenarios:
- Edge robot camera with strict latency budget
- Harsh outdoor weather surveillance with heavy noise and fog
- Limited data scratch training (no pretraining)
- High-capacity hierarchical representation learning
"""

import pytest
from src.recommender import VisionModelRecommender


@pytest.fixture
def recommender():
    return VisionModelRecommender()


def test_scenario_edge_robot_camera(recommender):
    """
    Scenario 1: Embedded edge drone / robot.
    Requires low latency (<25ms on CPU) and compact parameter size (<30M).
    """
    query = "Autonomous drone obstacle detection on embedded CPU, requires real-time inference under 25ms."
    res = recommender.recommend(query)
    top = res["top_recommendation"]

    assert top is not None
    # Winner must meet latency requirements
    assert top["raw_metrics"]["latency_ms"] <= 25.0
    # CNNs (ConvNeXt-Tiny or Scratch CNN) dominate this real-time edge regime
    assert top["architecture_family"] == "CNN"


def test_scenario_harsh_weather_surveillance(recommender):
    """
    Scenario 2: Outdoor security camera in harsh conditions with fog, snow, and sensor noise.
    """
    query = "Outdoor camera system dealing with severe fog, snow, and sensor noise. Accuracy and robustness are essential."
    res = recommender.recommend(query)
    top = res["top_recommendation"]

    assert top is not None
    # Vision Transformer (ViT) excels at global corruption and noise resilience
    assert top["model_id"] == "vit_base_16_cifar10_hf"
    assert top["architecture_family"] == "ViT"
    assert "noise" in res["requirements"].corruption_types or "fog" in res["requirements"].corruption_types


def test_scenario_scratch_training_small_dataset(recommender):
    """
    Scenario 3: Training from scratch on small dataset without pretraining.
    Demonstrates inductive bias advantage: Scratch Residual CNN beats Scratch Compact ViT on clean accuracy.
    """
    query = "Training a model from scratch with random initialization on limited data (50k images), no pretraining allowed."
    res = recommender.recommend(query)
    top = res["top_recommendation"]

    assert top is not None
    # Scratch CNN should be favored over Scratch ViT due to strong spatial inductive bias
    assert top["model_id"] == "scratch_cnn_cifar100"
    assert top["architecture_family"] == "CNN"
    assert "inductive bias" in res["explanation"]["architecture_analysis"].lower()


def test_scenario_high_throughput_hierarchical(recommender):
    """
    Scenario 4: High resolution classification on GPU server with hierarchical feature requirements.
    """
    query = "Multi-scale image classification on large dataset with high resolution 224x224 and GPU server infrastructure."
    overrides = {
        "dataset_size": "large",
        "hardware_target": "gpu",
        "weight_accuracy": 0.45,
        "weight_data_efficiency": 0.10
    }
    res = recommender.recommend(query, overrides=overrides)
    top = res["top_recommendation"]

    assert top is not None
    assert top["score"] > 60.0
