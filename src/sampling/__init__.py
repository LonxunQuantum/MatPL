"""
Standalone DIRECT sampling for MatPL — no maml dependency.

Provides:
  - NEPStructure: descriptor encoder using the trained NEP model
  - DIRECTSampler: the full pipeline (encode → scale → PCA → cluster → select)
  - DescriptorFingerprint: training-set descriptor statistics for novelty scoring
  - NoveltyAwareDIRECT: DIRECT variant that skips low-novelty clusters
"""
from src.sampling.direct import DIRECTSampler
from src.sampling.nep_describer import NEPStructure
from src.sampling.descriptor_fingerprint import DescriptorFingerprint
from src.sampling.training_aware_direct import NoveltyAwareDIRECT, NoveltyAwareSelectK

__all__ = [
    "DIRECTSampler",
    "NEPStructure",
    "DescriptorFingerprint",
    "NoveltyAwareDIRECT",
    "NoveltyAwareSelectK",
]
