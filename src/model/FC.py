import torch
import torch.nn as nn
import collections

# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup logging module
logger = logging.getLogger('train.FC')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dump(msg, *args, **kwargs):
    logger.log(logging_level_DUMP, msg, *args, **kwargs)
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)
def summary(msg, *args, **kwargs):
    logger.log(logging_level_SUMMARY, msg, *args, **kwargs)
def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)
def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs, exc_info=True)

# dmirror implementation
#
class f_linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, magic=False):
        super(f_linear, self).__init__()
        self.bias = None

        # for pesudo random number generator for float64/float32 precision test
        self.rand_a = 25214903917
        self.rand_c = 11
        self.rand_p = 2021

        if (magic == False):
            self.weight = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=True)
            if (bias == True):
                self.bias = nn.Parameter(torch.randn(out_dim), requires_grad=True)
        else:
            warmup_my_rand = self.my_rand_2d(out_dim, in_dim)
            self.weight = nn.Parameter(self.my_rand_2d(out_dim, in_dim), requires_grad=True)
            if (bias == True):
                self.bias = nn.Parameter(self.my_rand_1d(out_dim), requires_grad=True)
    # random number generator, maybe their better place is train.py
    def my_rand_core(self):
        r = (self.rand_a * self.rand_p + self.rand_c) % 10000
        self.rand_p = r
        return r

    def my_rand_2d(self, m, n):
        res = torch.randn(m, n)
        for i in range(m):
            for j in range(n):
                res[i, j] = float(self.my_rand_core() / 10000.0)
        return res

    def my_rand_1d(self, m):
        res = torch.randn(m)
        for i in range(m):
            res[i] = float(self.my_rand_core() / 10000.0)
        return res

    def forward(self, x):
        if (self.bias is not None):
            # import ipdb;ipdb.set_trace()
            return torch.matmul(x, self.weight.t()) + self.bias
        else:
            return torch.matmul(x, self.weight.t())


class f_activation(nn.Module):
    def __init__(self, func):
        super(f_activation, self).__init__()
        self.func = func
        self.k = torch.tensor([])

    def forward(self, x):
        self.k = x
        return self.func(x)

class pre_f_FC(nn.Module):
    def __init__(self, cfg, act_func, magic=False):
        super(pre_f_FC, self).__init__()
        self.cfg = cfg
        self.act_func = act_func
        self.layers = []

        # parse cfg & generating layers
        idx_linear = 1
        idx_activation = 1
        for idx, item in enumerate(self.cfg):
            layer_type = item[0]
            if (layer_type == 'linear'):
                in_dim = item[1]
                out_dim = item[2]
                bias = item[3]
                self.layers.append((
                    'f_linear_'+str(idx_linear),
                    f_linear(in_dim, out_dim, bias, magic)
                ))
                idx_linear += 1
            elif (layer_type == 'activation'):
                self.layers.append((
                    'f_activation_'+str(idx_activation),
                    f_activation(act_func)
                ))
                idx_activation += 1
            elif (layer_type == 'scale'):
                raise RuntimeError(
                    "Notimplemented for layer_type = %s" %(layer_type)
                )
            else:
                raise ValueError(
                    "Invalid for layer_type = %s" %(layer_type)
                )

        # the layer parameters will be registered to nn Module,
        # so optimizer can update the layer parameters.
        #
        self.base_net = nn.Sequential(
            collections.OrderedDict(self.layers)
        )
        info("pretraining forward FC: start of network instance dump ==============>")
        info("<----------------------- base_net ----------------------->")
        info(self.base_net)
        info("end of network instance dump ================>")

    # we can't call forward() of sequentialized module, since
    # we extened the param list of the layers' forward()
    #
    def forward(self, x):
        in_feature = x
        for name, obj in (self.layers):
            x = obj.forward(x)
        ei = x
        ei.unsqueeze(2)
        return ei


class f_FC(nn.Module):
    def __init__(self, cfg, act_func, magic=False):
        super(f_FC, self).__init__()
        self.cfg = cfg
        self.act_func = act_func
        self.layers = []

        # parse cfg & generating layers
        idx_linear = 1
        idx_activation = 1
        for idx, item in enumerate(self.cfg):
            layer_type = item[0]
            if (layer_type == 'linear'):
                in_dim = item[1]
                out_dim = item[2]
                bias = item[3]
                self.layers.append((
                    'f_linear_'+str(idx_linear),
                    f_linear(in_dim, out_dim, bias, magic)
                ))
                idx_linear += 1
            elif (layer_type == 'activation'):
                self.layers.append((
                    'f_activation_'+str(idx_activation),
                    f_activation(act_func)
                ))
                idx_activation += 1
            elif (layer_type == 'scale'):
                raise RuntimeError(
                    "Notimplemented for layer_type = %s" %(layer_type)
                )
            else:
                raise ValueError(
                    "Invalid for layer_type = %s" %(layer_type)
                )

        # the layer parameters will be registered to nn Module,
        # so optimizer can update the layer parameters.
        #
        self.base_net = nn.Sequential(
            collections.OrderedDict(self.layers)
        )
        info("forward FC: start of network instance dump ==============>")
        info("<----------------------- base_net ----------------------->")
        info(self.base_net)
        info("end of network instance dump ================>")

    # we can't call forward() of sequentialized module, since
    # we extened the param list of the layers' forward()
    #
    def forward(self, x):
        in_feature = x
        for name, obj in (self.layers):
            x = obj.forward(x)
        res0 = x
        mask = torch.ones_like(res0)
        dE = torch.autograd.grad(res0, in_feature, grad_outputs=mask, create_graph=True, retain_graph=True)
        dE = dE[0]
        # dE = torch.stack(list(dE), dim=0).squeeze(0)

        res1 = dE

        return res0, res1