from src.user.nep_param import NepParam


def test_use_analytical_nep_grad_defaults_to_true():
    param = NepParam()
    assert param.use_analytical_nep_grad is True


def test_use_analytical_nep_grad_can_be_disabled_from_fitting_net_json():
    param = NepParam()
    param.set_fixed_params({"model": {"fitting_net": {"use_analytical_nep_grad": False}}})
    assert param.use_analytical_nep_grad is False


def test_use_analytical_nep_grad_json_defaults_to_true_when_key_is_omitted():
    training_param = NepParam()
    training_param.set_nep_param_from_json(
        {"model": {"fitting_net": {}}},
        type_list=[1],
    )
    assert training_param.use_analytical_nep_grad is True

    checkpoint_param = NepParam()
    checkpoint_param.use_analytical_nep_grad = False
    checkpoint_param.set_fixed_params({"model": {"fitting_net": {}}})
    assert checkpoint_param.use_analytical_nep_grad is True


if __name__ == "__main__":
    test_use_analytical_nep_grad_defaults_to_true()
    test_use_analytical_nep_grad_can_be_disabled_from_fitting_net_json()
    test_use_analytical_nep_grad_json_defaults_to_true_when_key_is_omitted()
