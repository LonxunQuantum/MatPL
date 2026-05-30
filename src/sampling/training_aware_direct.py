"""
Novelty-aware DIRECT sampling — skips low-novelty clusters and selects
the highest-novelty structures within each remaining cluster.

Designed for the DESCRIPTOR uncertainty strategy: Stage 1 assigns per-structure
novelty scores, then this module uses those scores inside the DIRECT pipeline
to focus labeling budget on genuinely novel regions of descriptor space.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.sampling.direct import (
    BirchClustering,
    DIRECTSampler,
    PrincipalComponentAnalysis,
    SelectKFromClusters,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NoveltyAwareSelectK(BaseEstimator, TransformerMixin):
    """Select top-k highest-novelty structures per cluster, skipping low-novelty clusters.

    A cluster is skipped when its maximum novelty score is below
    ``cluster_skip_threshold``, meaning all members are near the
    training-set boundary with low labeling value.
    """

    def __init__(
        self,
        novelty_scores: np.ndarray,
        k: int = 1,
        cluster_skip_threshold: float = 4.5,
    ):
        self.novelty_scores = np.asarray(novelty_scores, dtype=float)
        self.k = k
        self.cluster_skip_threshold = cluster_skip_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, clustering_data: dict) -> dict:
        labels = clustering_data["labels"]
        features = clustering_data["PCAfeatures"]

        selected_indexes = []
        n_skipped = 0

        for label in sorted(set(labels)):
            idxs = np.where(labels == label)[0]
            cluster_novelty = self.novelty_scores[idxs]

            if cluster_novelty.max() < self.cluster_skip_threshold:
                n_skipped += 1
                continue

            k_actual = min(self.k, len(idxs))
            top_k = np.argsort(cluster_novelty)[-k_actual:]
            selected_indexes.extend(idxs[top_k])

        selected_indexes = list(set(selected_indexes))
        n_total = len(set(labels))
        logger.info(
            f"Novelty-aware selection: {len(selected_indexes)} structures from "
            f"{n_total - n_skipped}/{n_total} clusters "
            f"(skipped {n_skipped} clusters below threshold {self.cluster_skip_threshold:.2f})"
        )
        return {
            "PCAfeatures": features,
            "selected_indexes": selected_indexes,
        }


class NoveltyAwareDIRECT(DIRECTSampler):
    """DIRECT sampler that filters clusters by novelty and selects high-novelty members.

    Usage::

        sampler = NoveltyAwareDIRECT(
            structure_encoder=encoder,
            novelty_scores=scores,       # from DescriptorFingerprint.score_structure()
            cluster_skip_threshold=4.5,
            k=1,
        )
        result = sampler.fit_transform(candidate_structures)
    """

    def __init__(
        self,
        structure_encoder=None,
        novelty_scores: np.ndarray | None = None,
        scaler=None,
        pca=None,
        weighting_PCs: bool = True,
        clustering=None,
        k: int = 1,
        cluster_skip_threshold: float = 4.5,
    ):
        self.novelty_scores = novelty_scores if novelty_scores is not None else np.array([])
        self.k = k
        self.cluster_skip_threshold = cluster_skip_threshold

        if len(self.novelty_scores) > 0:
            select = NoveltyAwareSelectK(
                novelty_scores=self.novelty_scores,
                k=k,
                cluster_skip_threshold=cluster_skip_threshold,
            )
        else:
            select = SelectKFromClusters(k=k)

        super().__init__(
            structure_encoder=structure_encoder,
            scaler=scaler,
            pca=pca,
            weighting_PCs=weighting_PCs,
            clustering=clustering or BirchClustering(),
            select_k_from_clusters=select,
        )
