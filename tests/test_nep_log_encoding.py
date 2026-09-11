import builtins

from src.PWMLFF import nep_network


def test_nep_log_writer_uses_utf8(tmp_path, monkeypatch):
    opened = {}

    def open_spy(path, mode, *, encoding=None):
        opened["encoding"] = encoding
        return builtins.open(path, mode, encoding=encoding)

    monkeypatch.setattr(nep_network, "open", open_spy, raising=False)

    path = tmp_path / "epoch_train.dat"
    with nep_network._open_nep_log(path, "w") as stream:
        stream.write("RMSE_F(eV/Å)\n")

    assert opened["encoding"] == "utf-8"
    assert path.read_bytes() == "RMSE_F(eV/Å)\n".encode("utf-8")
