"""NEP model shapes must remain loadable by the bundled LAMMPS interface."""

import pytest
from pathlib import Path
import re

from src.user.nep_param import NepParam
from src.utils.nep_limits import MAX_FEATURES, MAX_HIDDEN_NEURONS


def make_param(n_max=(4, 4), basis_size=(12, 12), l_max=(4, 2, 1), hidden=40,
               type_list=(1,)):
    param = NepParam()
    param.set_nep_param_from_json(
        {"model": {
            "descriptor": {
                "n_max": list(n_max),
                "basis_size": list(basis_size),
                "l_max": list(l_max),
            },
            "fitting_net": {"network_size": hidden},
        }},
        type_list=list(type_list),
    )
    return param


@pytest.mark.parametrize("hidden", [30, 200, [30, 1], [200, 1]])
def test_hidden_neuron_boundaries(hidden):
    assert make_param(hidden=hidden).neuron in ([30, 1], [200, 1])


@pytest.mark.parametrize("hidden", [0, 29, 201, [40, 40, 1], [40, 2]])
def test_invalid_network_shapes_fail_at_parse(hidden):
    with pytest.raises(ValueError, match="single hidden layer|30..200"):
        make_param(hidden=hidden)


@pytest.mark.parametrize("name,value", [
    ("n_max", (3, 4)),
    ("n_max", (4, 20)),
    ("basis_size", (3, 4)),
    ("basis_size", (4, 20)),
])
def test_descriptor_ranges_fail_at_parse(name, value):
    args = {name: value}
    with pytest.raises(ValueError, match=name):
        make_param(**args)


def test_feature_limit_accepts_exact_boundary():
    param = make_param(n_max=(12, 14), basis_size=(19, 19), hidden=200)
    assert param.feature_nums == 103
    assert param.three_feat_num + param.four_feat_num + param.five_feat_num == 90


def test_feature_limit_rejects_104_at_parse():
    with pytest.raises(ValueError, match="104.*103"):
        make_param(n_max=(13, 14))


def test_descriptor_limit_is_checked_before_fitting_settings():
    with pytest.raises(ValueError, match="104.*103"):
        make_param(n_max=(13, 14), hidden=201)


def test_angular_limit_rejects_unsafe_legacy_lammps_shape():
    with pytest.raises(ValueError, match="96.*90"):
        make_param(n_max=(4, 15))  # 101 total, but 96 angular


def test_four_zero_zero_lmax_is_supported():
    assert make_param(l_max=(4, 0, 0)).feature_nums == 25


def test_all_periodic_table_elements_remain_accepted():
    param = make_param(type_list=(118,))
    assert param.type == "1 Og"


def test_nep_txt_rejects_oversize_before_weight_parsing(tmp_path):
    source = tmp_path / "nep.txt"
    source.write_text(
        "nep5 1 H\n"
        "cutoff 8 4\n"
        "n_max 13 14\n"
        "basis_size 12 12\n"
        "l_max 4 2 1\n"
        "ANN 40 0\n"
    )
    with pytest.raises(ValueError, match="104.*103"):
        NepParam().set_nep_nn_c_param_from_nep_txt(str(source), [1])


def test_cpp_and_python_fitting_capacities_agree():
    header = (Path(__file__).resolve().parents[1] / "src/op/include/nep_fitting_limits.h")
    source = header.read_text(encoding="utf-8")
    assert int(re.search(r"kMaxFeatures\s*=\s*(\d+)", source).group(1)) == MAX_FEATURES
    assert int(re.search(r"kMaxHiddenNeurons\s*=\s*(\d+)", source).group(1)) == MAX_HIDDEN_NEURONS
