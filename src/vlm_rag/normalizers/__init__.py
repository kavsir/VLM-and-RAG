"""Parser normalizers transforming raw layout outputs into Physical Document IR."""

from vlm_rag.normalizers.marker import MarkerNormalizationError, MarkerPhysicalNormalizer
from vlm_rag.normalizers.mineru import MinerUPhysicalNormalizer, NormalizationError

__all__ = [
    "MarkerNormalizationError",
    "MarkerPhysicalNormalizer",
    "MinerUPhysicalNormalizer",
    "NormalizationError",
]
