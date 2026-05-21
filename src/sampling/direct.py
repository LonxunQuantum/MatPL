"""
Standalone DIRECT sampling pipeline — replaces maml.sampling.direct.

Only depends on sklearn (StandardScaler, PCA, Birch) and numpy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PrincipalComponentAnalysis(BaseEstimator, TransformerMixin):
    """PCA with Kaiser's rule for component selection and optional variance weighting."""

    def __init__(self, weighting_PCs: bool = True):
        self.pca = PCA()
        self.weighting_PCs = weighting_PCs

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        m = int(np.sum(self.pca.explained_variance_ > 1))
        if m == 0:
            m = min(X.shape[1], 2)
        explained = self.pca.explained_variance_ratio_
        logger.info(f"Selected first {m} PCs, explaining {100 * sum(explained[:m]):.2f}% variance")
        transformed = self.pca.transform(X)[:, :m]
        if self.weighting_PCs:
            return transformed * explained[:m]
        return transformed


class BirchClustering(BaseEstimator, TransformerMixin):
    """Birch clustering step for the DIRECT pipeline."""

    def __init__(self, n: int | None = None, threshold_init: float = 0.5, **kwargs):
        self.n = n
        self.threshold_init = threshold_init
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, PCAfeatures):
        threshold = self.threshold_init
        model = Birch(n_clusters=self.n, threshold=threshold, **self.kwargs).fit(PCAfeatures)

        if self.n is not None:
            max_iter = 50
            for _ in range(max_iter):
                n_found = len(set(model.subcluster_labels_))
                if n_found >= self.n:
                    break
                threshold = threshold / self.n * n_found
                model = Birch(n_clusters=self.n, threshold=threshold, **self.kwargs).fit(PCAfeatures)

        labels = model.predict(PCAfeatures)
        self.model = model
        n_clusters = len(set(labels))
        logger.info(
            f"BirchClustering with threshold_init={self.threshold_init} and n={self.n} "
            f"gives {n_clusters} clusters."
        )
        label_centers = dict(zip(model.subcluster_labels_, model.subcluster_centers_))
        return {
            "labels": labels,
            "label_centers": label_centers,
            "PCAfeatures": PCAfeatures,
        }


class SelectKFromClusters(BaseEstimator, TransformerMixin):
    """Stratified sampling: select k structures from each cluster."""

    def __init__(self, k: int = 1, selection_criteria: str = "center"):
        self.k = k
        self.selection_criteria = selection_criteria

    def fit(self, X, y=None):
        return self

    def transform(self, clustering_data: dict):
        labels = clustering_data["labels"]
        features = clustering_data["PCAfeatures"]
        label_centers = clustering_data.get("label_centers", {})

        selected_indexes = []
        for label in set(labels):
            idxs = np.where(labels == label)[0]
            n_same = len(idxs)

            if self.selection_criteria == "center" and label_centers:
                center = label_centers.get(label)
                if center is not None:
                    dists = np.linalg.norm(features[idxs] - center, axis=1)
                    pick_positions = np.array(
                        [int(i) for i in np.linspace(0, n_same - 1, self.k)]
                    )
                    sorted_by_dist = np.argsort(dists)
                    selected_indexes.extend(idxs[sorted_by_dist[pick_positions]])
                else:
                    selected_indexes.extend(idxs[np.random.randint(n_same, size=self.k)])
            else:
                selected_indexes.extend(idxs[np.random.randint(n_same, size=self.k)])

        selected_indexes = list(set(selected_indexes))
        logger.info(f"Finally selected {len(selected_indexes)} configurations.")
        return {
            "PCAfeatures": features,
            "selected_indexes": selected_indexes,
        }


class DIRECTSampler(Pipeline):
    """
    DImensionality REduction-Clustering-sTratified (DIRECT) sampling.

    A self-contained sklearn Pipeline that encodes structures into descriptors,
    normalizes, reduces dimensionality via PCA, clusters, and selects
    representative structures.

    Reference: https://arxiv.org/abs/2307.13710
    """

    def __init__(
        self,
        structure_encoder: Any = None,
        scaler=None,
        pca=None,
        weighting_PCs: bool = True,
        clustering=None,
        select_k_from_clusters=None,
    ):
        """
        Args:
            structure_encoder: Any object with fit/transform that converts a list
                of structures into a 2-D feature array (N, D). Typically NEPStructure.
                Set to None/False to skip encoding (input is already a feature matrix).
            scaler: StandardScaler instance or None for default.
            pca: PrincipalComponentAnalysis instance or None for default.
            weighting_PCs: Weight PCs by explained variance (default True).
            clustering: BirchClustering instance or None for default.
            select_k_from_clusters: SelectKFromClusters instance or None for default.
        """
        self.structure_encoder = structure_encoder
        self.scaler = scaler or StandardScaler()
        self.pca = pca or PrincipalComponentAnalysis(weighting_PCs=weighting_PCs)
        self.weighting_PCs = weighting_PCs
        self.clustering = clustering or BirchClustering()
        self.select_k_from_clusters = select_k_from_clusters or SelectKFromClusters()

        steps = [
            (comp.__class__.__name__, comp)
            for comp in [
                self.structure_encoder,
                self.scaler,
                self.pca,
                self.clustering,
                self.select_k_from_clusters,
            ]
            if comp
        ]
        super().__init__(steps)
