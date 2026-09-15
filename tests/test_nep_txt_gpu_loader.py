import sys
import types

import pytest

from src.PWMLFF import nep_network


@pytest.mark.parametrize(
    ("hip_version", "build_backend"),
    [(None, "cuda"), ("6.3.0", "hip")],
)
def test_nep_txt_gpu_loader_uses_backend_build_directory(
        monkeypatch, hip_version, build_backend):
    module_name = (
        "src.feature.NEP_GPU.build.{}.nep_gpu".format(build_backend)
    )
    init_calls = []

    class FakeNepGpu:
        def init_from_file(self, *args):
            init_calls.append(args)

    monkeypatch.setattr(nep_network.torch.version, "hip", hip_version, raising=False)
    monkeypatch.setattr(nep_network.torch.cuda, "set_device", lambda gpu_id: None)
    monkeypatch.setitem(
        sys.modules,
        module_name,
        types.SimpleNamespace(NEP=FakeNepGpu),
    )

    calculator = nep_network._init_nep_txt_calculator(
        "/tmp/nep.txt", device_type="cuda", gpu_id=2, print_info=1
    )

    assert isinstance(calculator, FakeNepGpu)
    assert init_calls == [("/tmp/nep.txt", 1, 2)]
