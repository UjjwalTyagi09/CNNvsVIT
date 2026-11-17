"""
Unit tests for RequirementsParser.
"""

import pytest
from src.requirements_parser import RequirementsParser


@pytest.fixture
def parser():
    return RequirementsParser()


def test_edge_device_parsing(parser):
    query = "I need an edge computer vision model for a Raspberry Pi drone with low latency."
    req = parser.parse(query)
    assert req.hardware_target == "edge"
    assert req.max_latency_ms is not None
    assert req.max_latency_ms <= 25.0
    assert req.weight_latency > 0.25


def test_noise_and_weather_parsing(parser):
    query = "Deploying camera surveillance in adverse weather with heavy fog and sensor noise."
    req = parser.parse(query)
    assert req.noise_tolerance_critical is True
    assert "fog" in req.corruption_types
    assert "gaussian_noise" in req.corruption_types
    assert req.weight_robustness >= 0.30


def test_small_data_and_scratch_parsing(parser):
    query = "Training from scratch on a small dataset with 1500 medical images."
    req = parser.parse(query)
    assert req.dataset_size == "small"
    assert req.sample_count == 1500
    assert req.training_regime == "scratch"
    assert req.requires_good_calibration is True


def test_overrides(parser):
    query = "General vision model"
    overrides = {
        "max_latency_ms": 15.0,
        "dataset_size": "large",
        "weight_accuracy": 0.50
    }
    req = parser.parse(query, overrides=overrides)
    assert req.max_latency_ms == 15.0
    assert req.dataset_size == "large"
    assert req.weight_accuracy > 0.35
