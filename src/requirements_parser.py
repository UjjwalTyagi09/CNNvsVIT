"""
Requirements Parser.
Extracts operational constraints, dataset scales, robustness demands, and priority weights
from natural language problem statements and structured configuration overrides.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ProblemRequirements:
    """Structured operational and research requirements for model selection."""
    raw_query: str = ""
    dataset_size: str = "medium"  # "small" (<5k), "medium" (5k-50k), "large" (>50k)
    sample_count: Optional[int] = None
    training_regime: str = "transfer"  # "scratch" or "transfer"
    hardware_target: str = "any"  # "cpu", "gpu", "edge", "any"
    max_latency_ms: Optional[float] = None
    max_params: Optional[int] = None
    noise_tolerance_critical: bool = False
    corruption_types: List[str] = field(default_factory=list)
    requires_good_calibration: bool = False
    application_domain: str = "general"
    
    # Priority weights (normalized to sum to 1.0)
    weight_accuracy: float = 0.30
    weight_robustness: float = 0.25
    weight_latency: float = 0.20
    weight_data_efficiency: float = 0.15
    weight_calibration: float = 0.10

    def normalize_weights(self) -> None:
        """Ensure all weights sum to 1.0."""
        total = (
            self.weight_accuracy
            + self.weight_robustness
            + self.weight_latency
            + self.weight_data_efficiency
            + self.weight_calibration
        )
        if total > 0:
            self.weight_accuracy /= total
            self.weight_robustness /= total
            self.weight_latency /= total
            self.weight_data_efficiency /= total
            self.weight_calibration /= total


class RequirementsParser:
    """Parses natural language requirements and applies structured rules."""

    def __init__(self):
        # Keyword patterns for rule extraction
        self.noise_keywords = [
            "noise", "noisy", "grainy", "speckle", "gaussian", "sensor noise",
            "low-light", "adverse", "harsh", "distorted", "corrupt", "robustness", "robust"
        ]
        self.blur_keywords = ["blur", "defocus", "motion", "shake", "out of focus"]
        self.weather_keywords = ["fog", "snow", "frost", "rain", "outdoor", "weather", "lighting", "contrast"]
        self.edge_keywords = ["edge", "raspberry pi", "mobile", "iot", "embedded", "drone", "low-power", "battery"]
        self.cpu_keywords = ["cpu", "without gpu", "no gpu", "on-device cpu", "laptop"]
        self.realtime_keywords = ["real-time", "realtime", "instant", "low latency", "high fps", "fast inference"]
        self.small_data_keywords = [
            "limited data", "small dataset", "few samples", "limited samples", "rare",
            "scratch", "without pretraining", "from scratch", "500 images", "1000 images", "medical"
        ]
        self.large_data_keywords = ["huge dataset", "large dataset", "millions", "imagenet", "large scale", "web-scale"]
        self.calibration_keywords = ["calibration", "ece", "safety", "medical diagnosis", "autonomous driving", "confident"]

    def parse(self, text: str, overrides: Optional[Dict[str, Any]] = None) -> ProblemRequirements:
        """Parse natural language query and apply optional manual overrides."""
        req = ProblemRequirements(raw_query=text)
        lower = text.lower()

        # 1. Dataset size & training regime detection
        sample_match = re.search(r"(\d+[\d,]*)\s*(?:[a-zA-Z-]+\s+)?(samples|images|examples|instances)", lower)
        if sample_match:
            try:
                num = int(sample_match.group(1).replace(",", ""))
                req.sample_count = num
                if num < 5000:
                    req.dataset_size = "small"
                elif num > 50000:
                    req.dataset_size = "large"
                else:
                    req.dataset_size = "medium"
            except ValueError:
                pass
        elif any(k in lower for k in self.small_data_keywords):
            req.dataset_size = "small"
            req.weight_data_efficiency = 0.35
            req.weight_accuracy = 0.25
        elif any(k in lower for k in self.large_data_keywords):
            req.dataset_size = "large"
            req.weight_data_efficiency = 0.05
            req.weight_accuracy = 0.40

        if "from scratch" in lower or "random init" in lower or "no pretraining" in lower or "without pretraining" in lower:
            req.training_regime = "scratch"
            req.weight_data_efficiency = 0.35

        # 2. Hardware and Latency constraints
        if any(k in lower for k in self.edge_keywords):
            req.hardware_target = "edge"
            req.max_latency_ms = 25.0
            req.max_params = 15_000_000
            req.weight_latency = 0.35
            req.weight_accuracy = 0.20
        elif any(k in lower for k in self.cpu_keywords):
            req.hardware_target = "cpu"
            req.max_latency_ms = 35.0
            req.weight_latency = 0.30

        if any(k in lower for k in self.realtime_keywords):
            if req.max_latency_ms is None or req.max_latency_ms > 20.0:
                req.max_latency_ms = 20.0
            req.weight_latency = max(req.weight_latency, 0.35)

        # 3. Robustness and corruptions
        if any(k in lower for k in self.noise_keywords):
            req.noise_tolerance_critical = True
            req.weight_robustness = max(req.weight_robustness, 0.40)
            req.weight_latency = min(req.weight_latency, 0.10)
            req.corruption_types.extend(["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"])

        if any(k in lower for k in self.blur_keywords):
            req.noise_tolerance_critical = True
            req.weight_robustness = max(req.weight_robustness, 0.35)
            req.weight_latency = min(req.weight_latency, 0.10)
            req.corruption_types.extend(["defocus_blur", "motion_blur", "zoom_blur", "gaussian_blur"])

        if any(k in lower for k in self.weather_keywords):
            req.noise_tolerance_critical = True
            req.weight_robustness = max(req.weight_robustness, 0.35)
            req.weight_latency = min(req.weight_latency, 0.10)
            req.corruption_types.extend(["fog", "snow", "frost", "brightness"])

        # 4. Calibration & safety
        if any(k in lower for k in self.calibration_keywords):
            req.requires_good_calibration = True
            req.weight_calibration = 0.25

        # 5. Application domain
        if "medical" in lower:
            req.application_domain = "medical"
            req.requires_good_calibration = True
            req.weight_calibration = max(req.weight_calibration, 0.20)
        elif "autonomous" in lower or "car" in lower or "vehicle" in lower:
            req.application_domain = "autonomous_systems"
            req.noise_tolerance_critical = True
            req.weight_robustness = max(req.weight_robustness, 0.30)
        elif "security" in lower or "surveillance" in lower or "camera" in lower:
            req.application_domain = "surveillance"
            req.noise_tolerance_critical = True

        # Apply manual overrides if provided
        if overrides:
            for k, v in overrides.items():
                if hasattr(req, k) and v is not None:
                    setattr(req, k, v)

        req.normalize_weights()
        return req
