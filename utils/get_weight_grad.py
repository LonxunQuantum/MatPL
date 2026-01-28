import torch

def get_weights(model):
    params = {}
    for name, value in model.named_parameters():
        params[name] = value.cpu().detach().numpy()
    return params


def get_weights_grad(model):
    params = {}
    for name, value in model.named_parameters():
        params[name] = value.grad.cpu().detach().numpy()
    return params