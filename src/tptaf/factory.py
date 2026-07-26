"""Construction helpers for the released TPTAF method code."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from .detector import DetectionSemanticsProvider
from .joint import JointTPTAF
from .model import TPTAF

def _required_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if value is None: raise ValueError(f"Configuration value '{key}' is unresolved.")
    return int(value)

def _required_float(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if value is None: raise ValueError(f"Configuration value '{key}' is unresolved.")
    return float(value)

def build_fusion_model(config: Mapping[str, Any]) -> TPTAF:
    m = dict(config.get("model", {}))
    return TPTAF(channels=_required_int(m,"channels"), detection_channels=_required_int(m,"detection_channels"), num_heads=_required_int(m,"num_heads"), num_points=_required_int(m,"num_points"), window_size=_required_int(m,"window_size"), prior_strength=_required_float(m,"prior_strength"))

def build_joint_model(config: Mapping[str, Any], detector: DetectionSemanticsProvider) -> JointTPTAF:
    return JointTPTAF(build_fusion_model(config), detector)
