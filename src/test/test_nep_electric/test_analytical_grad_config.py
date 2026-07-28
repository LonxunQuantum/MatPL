from src.user.nep_param import NepParam


def test_use_analytical_nep_grad_defaults_to_false():
    param = NepParam()
    assert param.use_analytical_nep_grad is False


def test_use_analytical_nep_grad_can_be_enabled_from_fitting_net_json():
    param = NepParam()
    param.set_fixed_params({"model": {"fitting_net": {"use_analytical_nep_grad": True}}})
    assert param.use_analytical_nep_grad is True


if __name__ == "__main__":
    test_use_analytical_nep_grad_defaults_to_false()
    test_use_analytical_nep_grad_can_be_enabled_from_fitting_net_json()
