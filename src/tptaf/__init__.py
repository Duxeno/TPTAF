"""Public API for the TPTAF core method package."""

from .data import (
    PairedFusionDetectionDataset,
    PairedSample,
    collate_paired_batch,
    discover_pairs,
)
from .detector import (
    DetectionInputPreprocessor,
    DetectionSemanticsProvider,
    DetectorOutput,
    crop_semantic_feature,
    validate_detection_losses,
)
from .inference import tiled_fusion
from .joint import JointOutput, JointTPTAF
from .losses import (
    DecouplingLoss,
    GradientLoss,
    SimilarityLoss,
    TPTAFLoss,
    UncertaintyWeightedLoss,
)
from .model import (
    DecoupledEncoder,
    EncoderOutput,
    ReconstructionDecoder,
    TPTAF,
    TPTAFOutput,
)

__all__ = [
    "DecoupledEncoder",
    "DecouplingLoss",
    "DetectionInputPreprocessor",
    "DetectionSemanticsProvider",
    "DetectorOutput",
    "EncoderOutput",
    "GradientLoss",
    "JointOutput",
    "JointTPTAF",
    "PairedFusionDetectionDataset",
    "PairedSample",
    "ReconstructionDecoder",
    "SimilarityLoss",
    "TPTAF",
    "TPTAFLoss",
    "TPTAFOutput",
    "UncertaintyWeightedLoss",
    "collate_paired_batch",
    "crop_semantic_feature",
    "discover_pairs",
    "tiled_fusion",
    "validate_detection_losses",
]
