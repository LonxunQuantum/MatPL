"""
NEP descriptor as a sklearn-compatible structure encoder for DIRECT sampling.

No dependency on maml — only requires the compiled findneigh module (pybind11)
and standard scientific Python (numpy, ase).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from ase import Atoms


def _find_findneigh_module():
    """Locate and import the compiled findneigh pybind module."""
    import sysconfig
    abi_tag = sysconfig.get_config_var("EXT_SUFFIX") or ""

    candidates = []
    matpl_root = os.environ.get("MATPL_ROOT")
    if matpl_root:
        candidates.append(Path(matpl_root) / "src/feature/nep_find_neigh")
    # Relative to this file: MatPL/src/sampling/nep_describer.py → MatPL/src/feature/nep_find_neigh
    candidates.append(Path(__file__).resolve().parent.parent / "feature/nep_find_neigh")

    for base in candidates:
        base = base.resolve()
        if not base.exists():
            continue
        search_dirs = [base] + sorted(
            p for p in base.iterdir() if p.is_dir() and p.name.startswith("build")
        )
        for d in search_dirs:
            if abi_tag and (d / f"findneigh{abi_tag}").exists():
                if str(d) not in sys.path:
                    sys.path.insert(0, str(d))
                from findneigh import FindNeigh
                return FindNeigh
            if (d / "findneigh.so").exists():
                if str(d) not in sys.path:
                    sys.path.insert(0, str(d))
                try:
                    from findneigh import FindNeigh
                    return FindNeigh
                except ImportError:
                    sys.path.remove(str(d))
                    continue

    from findneigh import FindNeigh
    return FindNeigh


def _nep_ckpt_to_txt(ckpt_path: str) -> str:
    """Convert nep_model.ckpt to a temporary nep5.txt, return path."""
    matpl_root = os.environ.get("MATPL_ROOT")
    if matpl_root and matpl_root not in sys.path:
        sys.path.insert(0, matpl_root)
    parent = Path(__file__).resolve().parent.parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

    from src.utils.nep_to_gpumd import extract_model
    nep_content, _, _ = extract_model(ckpt_path)
    tmp_path = str(Path(ckpt_path).parent / "_tmp_nep_for_descriptor.txt")
    with open(tmp_path, "w") as f:
        f.write(nep_content)
    return tmp_path


class NEPStructure(BaseEstimator, TransformerMixin):
    """
    Compute per-structure NEP descriptors as fixed-length feature vectors.

    Implements the sklearn BaseEstimator/TransformerMixin interface so it can be
    used directly in the DIRECT pipeline without any maml dependency.
    """

    def __init__(self, nep_model_path: str, pool: str = "mean"):
        """
        Args:
            nep_model_path: Path to nep5.txt or nep_model.ckpt.
            pool: "mean" (default) or "sum" — how to pool per-atom descriptors.
        """
        self.nep_model_path = str(Path(nep_model_path).resolve())
        self.pool = pool
        self._tmp_txt = None

        FindNeigh = _find_findneigh_module()
        self._calc = FindNeigh()

        if self.nep_model_path.endswith(".ckpt"):
            self._tmp_txt = _nep_ckpt_to_txt(self.nep_model_path)
            self._calc.init_model(self._tmp_txt)
        else:
            self._calc.init_model(self.nep_model_path)

        self._dim = self._calc.getDim()
        self._element_order = self._read_element_order()

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, X, y=None):
        return self

    def transform(self, structures):
        """Transform a list of ASE Atoms into a (N, dim) feature array."""
        features = np.array([self._compute_one(s) for s in structures])
        return features

    def compute_peratom(self, structure) -> tuple[np.ndarray, list[str]]:
        """Return per-atom descriptors (N_atoms, dim) and element symbols."""
        atoms = self._to_ase(structure)
        type_map, box, position = self._atoms_to_nep_input(atoms)
        desc_flat = self._calc.getDescriptor(type_map, box, position)
        desc = np.array(desc_flat).reshape(-1, self._dim)
        return desc, atoms.get_chemical_symbols()

    def _compute_one(self, structure) -> np.ndarray:
        desc, _ = self.compute_peratom(structure)
        if self.pool == "mean":
            return desc.mean(axis=0)
        elif self.pool == "sum":
            return desc.sum(axis=0)
        raise ValueError(f"Unknown pool method: {self.pool}")

    def _to_ase(self, structure) -> "Atoms":
        from ase import Atoms as ASEAtoms
        if isinstance(structure, ASEAtoms):
            return structure
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            return AseAtomsAdaptor.get_atoms(structure)
        except (ImportError, AttributeError):
            raise TypeError(
                f"Cannot convert {type(structure)} to ASE Atoms. "
                "Pass ASE Atoms directly or install pymatgen."
            )

    def _atoms_to_nep_input(self, atoms: "Atoms"):
        """Convert ASE Atoms → (type_map, box, position) for FindNeigh."""
        symbols = atoms.get_chemical_symbols()
        elem_to_idx = {e: i for i, e in enumerate(self._element_order)}
        type_map = [elem_to_idx[s] for s in symbols]

        cell = np.array(atoms.cell)
        box = cell.T.flatten().tolist()

        pos = atoms.get_positions()
        position = pos[:, 0].tolist() + pos[:, 1].tolist() + pos[:, 2].tolist()

        return type_map, box, position

    def _read_element_order(self) -> list:
        txt_path = self._tmp_txt or self.nep_model_path
        with open(txt_path, "r") as f:
            first_line = f.readline().strip()
        parts = first_line.split()
        return parts[2:]

    def __del__(self):
        if self._tmp_txt and os.path.exists(self._tmp_txt):
            try:
                os.remove(self._tmp_txt)
            except OSError:
                pass
