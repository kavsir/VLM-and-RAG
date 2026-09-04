"""Parser normalizers transforming raw layout outputs into Physical Document IR."""

from vlm_rag.normalizers.mineru import MinerUPhysicalNormalizer, NormalizationError

__all__ = ["MinerUPhysicalNormalizer", "NormalizationError"]
