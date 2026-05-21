"""
Standalone DIRECT sampling for MatPL — no maml dependency.

Provides:
  - NEPStructure: descriptor encoder using the trained NEP model
  - DIRECTSampler: the full pipeline (encode → scale → PCA → cluster → select)
"""
from src.sampling.direct import DIRECTSampler
from src.sampling.nep_describer import NEPStructure

__all__ = ["DIRECTSampler", "NEPStructure"]
