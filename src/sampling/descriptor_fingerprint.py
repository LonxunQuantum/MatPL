"""
Descriptor-space fingerprint for training-set novelty scoring.

Computes per-atom NEP descriptor statistics from the training set, then scores
new structures by how far their atoms' descriptors deviate from the training
distribution. This provides a single-model uncertainty proxy that replaces the
multi-model committee (model_devi) approach.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms

from src.sampling.nep_describer import NEPStructure

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DescriptorFingerprint:
    """Training set descriptor statistics for novelty-based structure selection.

    Per-element statistics avoid cross-element contamination (e.g., O and In
    have very different descriptor distributions).
    """

    def __init__(self, nep_model_path: str, metric: str = "mahalanobis"):
        """
        Args:
            nep_model_path: Path to nep5.txt or nep_model.ckpt.
            metric: "mahalanobis" (default) or "max_zscore".
        """
        self.nep_model_path = nep_model_path
        self.metric = metric
        self._encoder = NEPStructure(nep_model_path)
        self._dim = self._encoder.dim

        self._elem_mu: dict[str, np.ndarray] = {}
        self._elem_sigma: dict[str, np.ndarray] = {}
        self._elem_cov_inv: dict[str, np.ndarray] = {}
        self._elem_count: dict[str, int] = {}
        self._fitted = False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def elements(self) -> list[str]:
        return sorted(self._elem_mu.keys())

    def fit(self, structures: list) -> "DescriptorFingerprint":
        """Build per-element descriptor statistics from training structures."""
        elem_descs: dict[str, list[np.ndarray]] = {}

        for struct in structures:
            desc, symbols = self._encoder.compute_peratom(struct)
            for i, sym in enumerate(symbols):
                elem_descs.setdefault(sym, []).append(desc[i])

        for sym, desc_list in elem_descs.items():
            arr = np.array(desc_list)
            self._elem_count[sym] = len(arr)
            self._elem_mu[sym] = arr.mean(axis=0)
            self._elem_sigma[sym] = arr.std(axis=0)
            self._elem_sigma[sym] = np.where(
                self._elem_sigma[sym] < 1e-12, 1.0, self._elem_sigma[sym]
            )
            if self.metric == "mahalanobis" and len(arr) > self._dim:
                cov = np.cov(arr, rowvar=False)
                cov += np.eye(self._dim) * 1e-8
                self._elem_cov_inv[sym] = np.linalg.inv(cov)

        total_atoms = sum(self._elem_count.values())
        logger.info(
            f"Descriptor fingerprint fitted: {len(elem_descs)} elements, "
            f"{total_atoms} atoms, dim={self._dim}"
        )
        self._fitted = True
        return self

    def score_atoms(self, structure) -> np.ndarray:
        """Return per-atom novelty scores (N,)."""
        desc, symbols = self._encoder.compute_peratom(structure)
        scores = np.zeros(len(symbols))
        for i, sym in enumerate(symbols):
            scores[i] = self._score_one_atom(desc[i], sym)
        return scores

    def score_structure(self, structure) -> float:
        """Return structure-level novelty = max(atom novelty scores)."""
        return float(self.score_atoms(structure).max())

    def score_structure_detail(self, structure) -> dict:
        """Return detailed novelty information for a structure."""
        atom_scores = self.score_atoms(structure)
        return {
            "max": float(atom_scores.max()),
            "mean": float(atom_scores.mean()),
            "median": float(np.median(atom_scores)),
            "top5_mean": float(np.sort(atom_scores)[-min(5, len(atom_scores)):].mean()),
            "n_atoms": len(atom_scores),
        }

    def _score_one_atom(self, desc: np.ndarray, element: str) -> float:
        if element not in self._elem_mu:
            return 99.0

        mu = self._elem_mu[element]
        delta = desc - mu

        if self.metric == "max_zscore":
            sigma = self._elem_sigma[element]
            z = np.abs(delta) / sigma
            return float(z.max())

        elif self.metric == "mahalanobis":
            cov_inv = self._elem_cov_inv.get(element)
            if cov_inv is not None:
                return float(np.sqrt(delta @ cov_inv @ delta))
            sigma = self._elem_sigma[element]
            z = np.abs(delta) / sigma
            return float(np.sqrt((z ** 2).mean()))

        raise ValueError(f"Unknown metric: {self.metric}")

    def classify_batch(
        self,
        structures: list,
        lower: float,
        upper: float,
    ) -> tuple[list[int], list[int], list[int], list[float]]:
        """Classify structures into accurate/candidate/failed.

        Returns:
            (accurate_idx, candidate_idx, failed_idx, all_scores)
        """
        accurate, candidate, failed = [], [], []
        scores = []
        for i, struct in enumerate(structures):
            s = self.score_structure(struct)
            scores.append(s)
            if s < lower:
                accurate.append(i)
            elif s < upper:
                candidate.append(i)
            else:
                failed.append(i)

        total = len(structures)
        logger.info(
            f"Descriptor classification: {total} structures → "
            f"accurate {len(accurate)} ({100*len(accurate)/total:.1f}%), "
            f"candidate {len(candidate)} ({100*len(candidate)/total:.1f}%), "
            f"failed {len(failed)} ({100*len(failed)/total:.1f}%)"
        )
        return accurate, candidate, failed, scores

    def adaptive_thresholds(self, structures: list) -> tuple[float, float]:
        """Compute adaptive thresholds from the training set itself.

        Returns (lower, upper) based on training set descriptor score quantiles.
        """
        all_atom_scores = []
        for struct in structures:
            all_atom_scores.extend(self.score_atoms(struct).tolist())
        arr = np.array(all_atom_scores)
        p99 = np.percentile(arr, 99)
        lower = p99
        upper = p99 * 2.5
        logger.info(
            f"Adaptive thresholds: p99={p99:.2f}, lower={lower:.2f}, upper={upper:.2f}"
        )
        return float(lower), float(upper)

    def save(self, path: str):
        """Serialize fingerprint to .npz file."""
        data = {
            "metric": np.array([self.metric]),
            "dim": np.array([self._dim]),
            "nep_model_path": np.array([self.nep_model_path]),
        }
        for sym in self._elem_mu:
            data[f"mu_{sym}"] = self._elem_mu[sym]
            data[f"sigma_{sym}"] = self._elem_sigma[sym]
            data[f"count_{sym}"] = np.array([self._elem_count[sym]])
            if sym in self._elem_cov_inv:
                data[f"cov_inv_{sym}"] = self._elem_cov_inv[sym]
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> "DescriptorFingerprint":
        """Deserialize fingerprint from .npz file."""
        data = np.load(path, allow_pickle=False)
        nep_path = str(data["nep_model_path"][0])
        metric = str(data["metric"][0])
        fp = cls(nep_path, metric=metric)
        fp._dim = int(data["dim"][0])

        elements = set()
        for key in data.files:
            if key.startswith("mu_"):
                elements.add(key[3:])

        for sym in elements:
            fp._elem_mu[sym] = data[f"mu_{sym}"]
            fp._elem_sigma[sym] = data[f"sigma_{sym}"]
            fp._elem_count[sym] = int(data[f"count_{sym}"][0])
            cov_key = f"cov_inv_{sym}"
            if cov_key in data:
                fp._elem_cov_inv[sym] = data[cov_key]

        fp._fitted = True
        return fp
