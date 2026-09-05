"""Parser normalizers transforming raw layout outputs into Physical Document IR."""

from vlm_rag.normalizers.marker import MarkerNormalizationError, MarkerPhysicalNormalizer
from vlm_rag.normalizers.marker_v1 import MarkerPhysicalNormalizerV1
from vlm_rag.normalizers.mineru import MinerUPhysicalNormalizer, NormalizationError
from vlm_rag.normalizers.mineru_v1 import MinerUPhysicalNormalizerV1

__all__ = [
    "MarkerNormalizationError",
    "MarkerPhysicalNormalizer",
    "MarkerPhysicalNormalizerV1",
    "MinerUPhysicalNormalizer",
    "MinerUPhysicalNormalizerV1",
    "NormalizationError",
]
