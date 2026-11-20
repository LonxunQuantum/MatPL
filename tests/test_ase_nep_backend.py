import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.ase.calculate import MatPL_calculator


@pytest.mark.parametrize(
    ("hip_version", "expected_backend"),
    [(None, "cuda"), ("6.3.0", "hip")],
)
def test_ase_nep_loads_extension_for_torch_backend(
    monkeypatch, tmp_path, hip_version, expected_backend
):
    model = tmp_path / "nep.txt"
    model.write_text("nep4 1 Ag\n", encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", hip_version, raising=False)
    extensions = {"cuda": Mock(), "hip": Mock()}
    for backend, extension in extensions.items():
        monkeypatch.setitem(
            sys.modules,
            f"src.feature.NEP_GPU.build.{backend}.nep_gpu",
            SimpleNamespace(NEP=extension),
        )

    calculator = MatPL_calculator(model_file=str(model))

    expected_extension = extensions[expected_backend]
    assert calculator.calc.calc is expected_extension.return_value
    calculator.calc.calc.init_from_file.assert_called_once_with(str(model), 1, 0)
