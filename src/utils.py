"""
Shared utility functions for normalization, metric formatting, and safe data access.
"""

from typing import Dict, List, Any, Optional


def safe_get(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Safely traverse a nested dictionary given a list of keys."""
    curr = data
    for key in keys:
        if isinstance(curr, dict) and key in curr:
            curr = curr[key]
        else:
            return default
    return curr


def normalize_series(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, float]:
    """
    Min-max normalize a dictionary of {id: value} to [0.0, 1.0].
    If all values are equal, returns 1.0 for all.
    """
    if not values:
        return {}
    
    val_list = list(values.values())
    min_v = min(val_list)
    max_v = max(val_list)
    
    if max_v == min_v:
        return {k: 1.0 for k in values}
    
    normalized = {}
    for k, v in values.items():
        if higher_is_better:
            normalized[k] = (v - min_v) / (max_v - min_v)
        else:
            normalized[k] = (max_v - v) / (max_v - min_v)
    return normalized


def compute_relative_retention(clean_acc: float, corrupted_acc: float) -> float:
    """Compute the percentage of clean accuracy retained under corruption."""
    if clean_acc <= 0:
        return 0.0
    return max(0.0, min(100.0, (corrupted_acc / clean_acc) * 100.0))


def format_params(count: int) -> str:
    """Format parameter count into human-readable string (e.g., 25.6M, 960K)."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def format_latency(ms: float) -> str:
    """Format millisecond latency into clean string."""
    if ms < 1.0:
        return f"{ms * 1000:.0f} µs"
    return f"{ms:.1f} ms"


def format_percentage(val: float, decimals: int = 1) -> str:
    """Format float into percentage string."""
    return f"{val:.{decimals}f}%"
